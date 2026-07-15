"""
Stage 1 of the VQGAN+AR baseline: train 3D VQGAN on T21 patches.

Loss = MSE + 0.5*log-PS (k-weighted, same recipe as the VAE) + 0.25*commit
       + 0.1*hinge-GAN (discriminator enabled after --disc_start epochs).

Codebook health is logged (perplexity); a collapsed codebook (<10% usage)
invalidates stage 2, watch that column.
"""
import os, argparse, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from ldm_dataset import build_train_dataset, make_balanced_sampler
from models.vqgan3d import VQGAN3D, PatchDiscriminator3D


def log_ps_loss(recon, target, k_alpha=2.0):
    """Mean-subtracted, k-weighted log-PS MSE (matches VAE recipe)."""
    N = recon.shape[-1]
    r = recon - recon.mean(dim=(-3, -2, -1), keepdim=True)
    t = target - target.mean(dim=(-3, -2, -1), keepdim=True)
    Fr = torch.fft.rfftn(r.float(), dim=(-3, -2, -1))
    Ft = torch.fft.rfftn(t.float(), dim=(-3, -2, -1))
    Pr = Fr.real**2 + Fr.imag**2
    Pt = Ft.real**2 + Ft.imag**2
    k1  = torch.fft.fftfreq(N, device=recon.device) * N
    k1r = torch.fft.rfftfreq(N, device=recon.device) * N
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = (KX**2 + KY**2 + KZ**2).sqrt().clamp(min=1.0)
    w = Kmag ** k_alpha
    diff = (torch.log(Pr + 1e-12) - torch.log(Pt + 1e-12)) ** 2
    return (w * diff).sum() / w.sum() / recon.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='/root/autodl-tmp/checkpoints/vqgan')
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--epochs',     type=int, default=120)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--n_embed',    type=int, default=1024)
    p.add_argument('--embed_dim',  type=int, default=8)
    p.add_argument('--base_ch',    type=int, default=128)
    p.add_argument('--ps_weight',  type=float, default=0.5)
    p.add_argument('--commit_weight', type=float, default=0.25)
    p.add_argument('--gan_weight', type=float, default=0.1)
    p.add_argument('--disc_start', type=int, default=40,
                   help='epoch at which the discriminator turns on')
    p.add_argument('--patches_per_cube', type=int, default=4)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--resume', default=None)
    args = p.parse_args()
    dev = torch.device('cuda')
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, weights = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        redshifts=tuple(args.redshifts),
        patches_per_cube=args.patches_per_cube, augment=True, split='train')
    sampler = make_balanced_sampler(weights, num_samples=len(train_ds))
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True)
    val_ds, _ = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        redshifts=tuple(args.redshifts),
        patches_per_cube=1, augment=False, split='val')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"train {len(train_ds)} / val {len(val_ds)} patches")

    G = VQGAN3D(in_ch=1, embed_dim=args.embed_dim, n_embed=args.n_embed,
                base_ch=args.base_ch).to(dev)
    D = PatchDiscriminator3D().to(dev)
    print(f"G params {sum(p_.numel() for p_ in G.parameters())/1e6:.1f}M, "
          f"D params {sum(p_.numel() for p_ in D.parameters())/1e6:.1f}M")

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        rc = torch.load(args.resume, map_location=dev)
        G.load_state_dict(rc['G']); D.load_state_dict(rc['D'])
        optG.load_state_dict(rc['optG']); optD.load_state_dict(rc['optD'])
        start_epoch = rc['epoch'] + 1
        print(f"resumed from epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,rec,ps,commit,g_gan,d_loss,perplexity,val_rec,sec\n')

    model_config = dict(embed_dim=args.embed_dim, n_embed=args.n_embed,
                        base_ch=args.base_ch, latent_shape=[16, 16, 16])

    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train(); t0 = time.time()
        sums = dict(rec=0, ps=0, commit=0, g_gan=0, d_loss=0, ppl=0)
        n_b = 0
        use_disc = epoch >= args.disc_start
        for batch in loader:
            x = batch['patch'].to(dev)
            with autocast(dtype=torch.bfloat16):
                recon, commit, idx = G(x)
                rec = F.mse_loss(recon, x)
            ps = log_ps_loss(recon.float(), x.float())
            lossG = rec + args.ps_weight * ps + args.commit_weight * commit
            if use_disc:
                with autocast(dtype=torch.bfloat16):
                    logits_fake = D(recon)
                g_gan = -logits_fake.float().mean()
                lossG = lossG + args.gan_weight * g_gan
            else:
                g_gan = torch.tensor(0.0)
            optG.zero_grad(); lossG.backward(); optG.step()

            if use_disc:
                with autocast(dtype=torch.bfloat16):
                    lf = D(recon.detach()); lr_ = D(x)
                d_loss = (F.relu(1 - lr_.float()).mean()
                          + F.relu(1 + lf.float()).mean())
                optD.zero_grad(); d_loss.backward(); optD.step()
            else:
                d_loss = torch.tensor(0.0)

            with torch.no_grad():
                counts = torch.bincount(idx.flatten(), minlength=args.n_embed).float()
                probs = counts / counts.sum()
                ppl = float(torch.exp(-(probs * (probs + 1e-10).log()).sum()))
            sums['rec'] += rec.item(); sums['ps'] += float(ps)
            sums['commit'] += commit.item(); sums['g_gan'] += float(g_gan)
            sums['d_loss'] += float(d_loss); sums['ppl'] += ppl
            n_b += 1

        G.eval(); val_rec, n_v = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 20: break
                x = batch['patch'].to(dev)
                with autocast(dtype=torch.bfloat16):
                    recon, _, _ = G(x)
                val_rec += F.mse_loss(recon.float(), x.float()).item(); n_v += 1

        m = {k: v / max(n_b, 1) for k, v in sums.items()}
        dt = time.time() - t0
        print(f"ep {epoch:03d} | rec {m['rec']:.4f} ps {m['ps']:.3f} "
              f"commit {m['commit']:.4f} gGAN {m['g_gan']:.3f} D {m['d_loss']:.3f} "
              f"| ppl {m['ppl']:.0f}/{args.n_embed} | val {val_rec/max(n_v,1):.4f} | {dt:.0f}s",
              flush=True)
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{m['rec']:.6f},{m['ps']:.6f},{m['commit']:.6f},"
                    f"{m['g_gan']:.6f},{m['d_loss']:.6f},{m['ppl']:.1f},"
                    f"{val_rec/max(n_v,1):.6f},{dt:.1f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(dict(G=G.state_dict(), D=D.state_dict(),
                            optG=optG.state_dict(), optD=optD.state_dict(),
                            epoch=epoch, model_config=model_config),
                       os.path.join(args.out_dir, f'vqgan_epoch{epoch:04d}.pt'))
            print(f"  saved epoch {epoch}")


if __name__ == '__main__':
    main()
