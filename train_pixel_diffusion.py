"""
Pixel-space EDM diffusion baseline — NO VAE (compression = 1x).

Same EDM recipe / conditioning / CFG dropout as the latent LDM, but the
denoiser U-Net runs directly on the 64^3 T21 patch (1 channel):

  levels: 64^3 -> 32^3 -> 16^3 (windowed attn) -> 8^3 (global attn)
  ch_mults (1,2,4,4), base_ch 32  ->  ~35M params

Role in the paper: the compression-ablation end point (0x) — shows what
latent compression buys in memory / speed at matched quality.
"""
import os, argparse, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from ldm_dataset import build_train_dataset, make_balanced_sampler
from ldm_unet import LDMUNet3D
from train_ldm import edm_loss, EMA


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir',   default='/root/autodl-tmp/checkpoints/pixel_diff')
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--epochs',     type=int, default=250)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--wd',         type=float, default=1e-4)
    p.add_argument('--patch_size', type=int, default=64)
    p.add_argument('--patches_per_cube', type=int, default=4)
    p.add_argument('--base_ch',    type=int, default=32)
    p.add_argument('--P_mean',     type=float, default=-0.4)
    p.add_argument('--P_std',      type=float, default=1.2)
    p.add_argument('--drop_ic',    type=float, default=0.10)
    p.add_argument('--drop_params',type=float, default=0.10)
    p.add_argument('--drop_both',  type=float, default=0.05)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--resume',     default=None)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--ema_decays', nargs='+', type=float, default=[0.9999, 0.999])
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda')
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, weights = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        redshifts=tuple(args.redshifts),
        patch_size=args.patch_size,
        patches_per_cube=args.patches_per_cube,
        augment=True, split='train')
    sampler = make_balanced_sampler(weights, num_samples=len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_ds, _ = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        redshifts=tuple(args.redshifts),
        patch_size=args.patch_size,
        patches_per_cube=1, augment=False, split='val')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"train: {len(train_ds)} patches, val: {len(val_ds)} patches")

    # sigma_data: patches are z-scored, expect ~1.0; measure anyway
    accum, n_seen = 0.0, 0
    for i, b in enumerate(train_loader):
        if i >= 20: break
        x = b['patch']
        accum += x.float().pow(2).sum().item(); n_seen += x.numel()
    sigma_data = math.sqrt(accum / n_seen)
    print(f"empirical sigma_data = {sigma_data:.4f}")

    model_config = dict(latent_ch=1, ic_stem_downsamples=0,
                        base_ch=args.base_ch, ch_mults=(1, 2, 4, 4),
                        attn_levels=(2, 3), patch_size=args.patch_size,
                        sigma_data=sigma_data)
    model = LDMUNet3D(latent_ch=1, ic_stem_downsamples=0,
                      base_ch=args.base_ch, ch_mults=(1, 2, 4, 4),
                      attn_levels=(2, 3)).to(device)
    print(f"pixel U-Net params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    emas = {d: EMA(model, decay=d) for d in args.ema_decays}
    null_param = torch.zeros(1, 4, device=device)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        rc = torch.load(args.resume, map_location=device)
        model.load_state_dict(rc['model']); opt.load_state_dict(rc['opt'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.epochs - (rc['epoch'] + 1)))
        for g in opt.param_groups: g['lr'] = args.lr
        for d in args.ema_decays:
            if f'ema_{d}' in rc:
                emas[d].shadow = {k: v.to(device) for k, v in rc[f'ema_{d}'].items()}
        start_epoch = rc['epoch'] + 1
        print(f"resumed from epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,lr,sec\n')

    for epoch in range(start_epoch, args.epochs):
        model.train(); t0 = time.time()
        tr_sum, n_b = 0.0, 0
        for batch in train_loader:
            x      = batch['patch'].to(device)          # (B,1,64,64,64) — the target itself
            ic_d   = batch['ic_delta'].to(device)
            ic_v   = batch['ic_vbv'].to(device)
            params = batch['params'][:, :4].to(device)
            redshift = batch['redshift'].to(device)
            with autocast(dtype=torch.bfloat16):
                loss = edm_loss(model, x, ic_d, ic_v, params, redshift,
                                sigma_data=sigma_data,
                                P_mean=args.P_mean, P_std=args.P_std,
                                drop_ic=args.drop_ic, drop_params=args.drop_params,
                                drop_both=args.drop_both, null_param=null_param)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for e in emas.values(): e.update(model)
            tr_sum += loss.item(); n_b += 1
        scheduler.step()

        model.eval(); va_sum, n_v = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 20: break
                x      = batch['patch'].to(device)
                ic_d   = batch['ic_delta'].to(device)
                ic_v   = batch['ic_vbv'].to(device)
                params = batch['params'][:, :4].to(device)
                redshift = batch['redshift'].to(device)
                with autocast(dtype=torch.bfloat16):
                    loss = edm_loss(model, x, ic_d, ic_v, params, redshift,
                                    sigma_data=sigma_data,
                                    P_mean=args.P_mean, P_std=args.P_std,
                                    drop_ic=0, drop_params=0, drop_both=0,
                                    null_param=null_param)
                va_sum += loss.item(); n_v += 1

        dt = time.time() - t0
        lr_now = opt.param_groups[0]['lr']
        print(f"ep {epoch:04d} | train {tr_sum/max(n_b,1):.4f} | "
              f"val {va_sum/max(n_v,1):.4f} | lr {lr_now:.2e} | {dt:.0f}s", flush=True)
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{tr_sum/max(n_b,1):.6f},{va_sum/max(n_v,1):.6f},{lr_now:.6e},{dt:.1f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ck = dict(model=model.state_dict(), opt=opt.state_dict(),
                      scheduler=scheduler.state_dict(), epoch=epoch,
                      model_config=model_config, sigma_data=sigma_data)
            for d, e in emas.items():
                ck[f'ema_{d}'] = e.shadow
            torch.save(ck, os.path.join(args.out_dir, f'pixel_diff_epoch{epoch:04d}.pt'))
            print(f"  saved epoch {epoch}")


if __name__ == '__main__':
    main()
