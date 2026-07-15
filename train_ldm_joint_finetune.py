"""Joint LDM + VAE-decoder fine-tune.

Loads LDM ckpt (e.g. ldm_2x ep124) + VAE ckpt (vae_2x_final),
unfreezes the VAE decoder, and fine-tunes both jointly on:
  - 40% random subset of train patches (resampled every 5 epochs)
  - FULL val set patches (every epoch)
Test set is never touched.

Loss = LDM denoising (latent space) + decoder reconstruction (pixel space).
"""
import os, argparse, math, time, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.amp import autocast

from models.vae import VAE3D
from ldm_unet import LDMUNet3D
from ldm_dataset import build_train_dataset, make_balanced_sampler

# Reuse loss/sampling helpers from train_ldm
from train_ldm import edm_precond, EMA


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt', required=True)
    p.add_argument('--ldm_ckpt', required=True)
    p.add_argument('--out_dir',  required=True)
    p.add_argument('--data_root_ic',    required=True)
    p.add_argument('--data_root_astro', required=True)
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=4)  # smaller (decoder needs memory)
    p.add_argument('--lr_ldm',     type=float, default=1e-5)
    p.add_argument('--lr_dec',     type=float, default=5e-6)
    p.add_argument('--wd',         type=float, default=1e-4)
    p.add_argument('--patch_size', type=int, default=64)
    p.add_argument('--patches_per_cube', type=int, default=4)
    p.add_argument('--redshifts', nargs='+', type=int, default=[7,8,9,10,11,12,13])
    p.add_argument('--P_mean', type=float, default=-0.4)
    p.add_argument('--P_std',  type=float, default=1.2)
    p.add_argument('--drop_ic',     type=float, default=0.10)
    p.add_argument('--drop_params', type=float, default=0.10)
    p.add_argument('--drop_both',   type=float, default=0.05)
    p.add_argument('--decoder_loss_weight', type=float, default=1.0,
                   help='weight on pixel-space recon loss')
    p.add_argument('--train_subset_frac', type=float, default=0.40,
                   help='fraction of train set used per round')
    p.add_argument('--refresh_every', type=int, default=5,
                   help='resample train subset every N epochs')
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--resume', default=None)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')

    # ---------- Load VAE (encoder frozen, decoder TRAINABLE) ----------
    print(f"Loading VAE from {args.vae_ckpt}")
    vck = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    cfg = vck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
                base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(device)
    vae.load_state_dict(vck['model'])

    # Freeze encoder, train decoder
    for p_ in vae.encoder.parameters():
        p_.requires_grad_(False)
    for p_ in vae.decoder.parameters():
        p_.requires_grad_(True)
    vae.encoder.eval()
    vae.decoder.train()
    n_dec = sum(p.numel() for p in vae.decoder.parameters() if p.requires_grad)
    print(f"  decoder trainable params: {n_dec/1e6:.2f} M")

    latent_mean = vck['latent_mean'].view(1, -1, 1, 1, 1).to(device)
    latent_std  = vck['latent_std' ].view(1, -1, 1, 1, 1).to(device)

    # ---------- Load LDM ----------
    print(f"Loading LDM from {args.ldm_ckpt}")
    lck = torch.load(args.ldm_ckpt, map_location=device, weights_only=False)
    model = LDMUNet3D(latent_ch=cfg['latent_ch'],
                      ch_mults=(1, 2),
                      attn_levels=(0, 1),
                      ic_stem_downsamples=2).to(device)
    model.load_state_dict(lck['model'])
    n_ldm = sum(p.numel() for p in model.parameters())
    print(f"  LDM params: {n_ldm/1e6:.2f} M")
    sigma_data = float(lck.get('sigma_data', 1.0))
    print(f"  sigma_data: {sigma_data:.4f}")

    # ---------- Datasets ----------
    train_ds, train_w = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        redshifts=tuple(args.redshifts),
        patch_size=args.patch_size, patches_per_cube=args.patches_per_cube,
        augment=True, split='train')
    val_ds, val_w = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        redshifts=tuple(args.redshifts),
        patch_size=args.patch_size, patches_per_cube=args.patches_per_cube,
        augment=True, split='val')
    print(f"train pool: {len(train_ds)} patches | val (always included): {len(val_ds)} patches")

    # ---------- Optimizers (LDM and decoder separately, different LRs) ----------
    opt_ldm = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr_ldm, weight_decay=args.wd)
    opt_dec = torch.optim.AdamW([p for p in vae.decoder.parameters() if p.requires_grad],
                                 lr=args.lr_dec, weight_decay=args.wd)
    sched_ldm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ldm, T_max=args.epochs)
    sched_dec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dec, T_max=args.epochs)

    # ---------- Resume ----------
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        rck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(rck['ldm'])
        vae.decoder.load_state_dict(rck['decoder'])
        opt_ldm.load_state_dict(rck['opt_ldm'])
        opt_dec.load_state_dict(rck['opt_dec'])
        start_epoch = rck['epoch'] + 1

    # ---------- Training loop ----------
    train_subset_idx = None
    history = []
    for epoch in range(start_epoch, args.epochs):
        # refresh 40% train subset every refresh_every epochs
        if epoch == start_epoch or epoch % args.refresh_every == 0:
            n_keep = int(args.train_subset_frac * len(train_ds))
            train_subset_idx = sorted(random.sample(range(len(train_ds)), n_keep))
            print(f"[ep {epoch}] resampled train subset: {n_keep} of {len(train_ds)}")

        train_subset = Subset(train_ds, train_subset_idx)
        joint_ds = ConcatDataset([train_subset, val_ds])
        loader = DataLoader(joint_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)

        model.train(); vae.decoder.train()
        ldm_losses, dec_losses = [], []
        t0 = time.time()
        for batch in loader:
            x = batch['patch'].to(device, non_blocking=True)        # (B,1,64,64,64)
            ic_d = batch['ic_delta'].to(device, non_blocking=True)
            ic_v = batch['ic_vbv'].to(device, non_blocking=True)
            params = batch['params'].to(device, non_blocking=True)
            B = x.size(0)

            with autocast('cuda', dtype=torch.bfloat16):
                # Encode (frozen encoder)
                with torch.no_grad():
                    mu, _ = vae.encoder(x)
                    z = (mu - latent_mean) / latent_std

                # Sample noise level
                sigma = (torch.randn(B, device=device) * args.P_std + args.P_mean).exp()
                noise = torch.randn_like(z) * sigma.view(-1,1,1,1,1)
                z_noisy = z + noise

                # CFG dropouts
                drop_ic_mask     = (torch.rand(B, device=device) < args.drop_ic    ).float().view(-1,1,1,1,1)
                drop_params_mask = (torch.rand(B, device=device) < args.drop_params).float().view(-1,1)
                ic_d_drop = ic_d * (1 - drop_ic_mask)
                ic_v_drop = ic_v * (1 - drop_ic_mask)
                params_drop = params * (1 - drop_params_mask)

                # LDM forward (EDM preconditioned)
                pred_z = edm_precond(model, z_noisy, sigma, ic_d_drop, ic_v_drop,
                                     params_drop[:, :4], params_drop[:, 4],
                                     sigma_data=sigma_data)

                # LDM denoising loss
                w = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
                ldm_loss = (w.view(-1,1,1,1,1) * (pred_z - z)**2).mean()

                # Decoder reconstruction loss: decode predicted latent back to pixel
                z_hat_unnorm = pred_z * latent_std + latent_mean
                x_recon = vae.decoder(z_hat_unnorm)
                dec_loss = F.mse_loss(x_recon, x)

                total_loss = ldm_loss + args.decoder_loss_weight * dec_loss

            opt_ldm.zero_grad(set_to_none=True)
            opt_dec.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(vae.decoder.parameters(), 1.0)
            opt_ldm.step(); opt_dec.step()

            ldm_losses.append(float(ldm_loss.item()))
            dec_losses.append(float(dec_loss.item()))

        sched_ldm.step(); sched_dec.step()
        dt = time.time() - t0
        tr_ldm = float(np.mean(ldm_losses)); tr_dec = float(np.mean(dec_losses))
        print(f"Ep {epoch:4d} | ldm {tr_ldm:.5f} | dec {tr_dec:.5f} | lr_ldm {opt_ldm.param_groups[0]['lr']:.2e} | {dt:.0f}s", flush=True)
        history.append(dict(epoch=epoch, ldm=tr_ldm, dec=tr_dec, dt=dt))

        # Save
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.out_dir, f'joint_epoch{epoch:04d}.pt')
            torch.save(dict(
                ldm=model.state_dict(),
                decoder=vae.decoder.state_dict(),
                opt_ldm=opt_ldm.state_dict(),
                opt_dec=opt_dec.state_dict(),
                epoch=epoch,
                sigma_data=sigma_data,
                history=history,
            ), path)
            print(f"  saved {path}")


if __name__ == '__main__':
    main()
