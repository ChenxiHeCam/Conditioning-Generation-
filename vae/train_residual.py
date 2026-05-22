"""
Train a residual VAE on top of a frozen v3 VAE.

Goal: recover small-scale (k>0.25) power lost by the main VAE's 4x downsampling.

Architecture:
  - v3 VAE: frozen, ch_mults=(1,2), latent_ch=8  -> handles low/mid frequencies
  - Residual VAE: ch_mults=(1,), latent_ch=4     -> handles high-freq residual
  - Final reconstruction: x_low + r_hat

Loss:
  - MSE on residual:               (r_hat - (x - x_low))^2
  - High-k-weighted PS loss on final reconstruction
  - Tiny KL on residual latent

Usage:
  python train_residual.py \
    --vae_ckpt /root/autodl-tmp/checkpoints/vae_v3/vae_v3_epoch0059.pt \
    --out_dir /root/autodl-tmp/checkpoints/vae_residual \
    --epochs 60
"""
import os, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast

from dataset import T21Dataset
from models.vae import VAE3D


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt',        required=True,
                   help='Frozen low-freq VAE checkpoint (v3)')
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--out_dir',         default='/root/autodl-tmp/checkpoints/vae_residual')
    p.add_argument('--resume',          default=None)
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--patch_size',  type=int,   default=64)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=60)
    # Frozen VAE (v3) config -- must match the checkpoint
    p.add_argument('--vae_latent_ch', type=int, default=8)
    p.add_argument('--vae_base_ch',   type=int, default=64)
    # Residual VAE config
    p.add_argument('--res_latent_ch', type=int, default=4)
    p.add_argument('--res_base_ch',   type=int, default=32)
    # Loss weights
    p.add_argument('--kl_weight',    type=float, default=1e-5)
    p.add_argument('--ps_weight',    type=float, default=0.3,
                   help='weight for high-k-emphasized PS loss on final recon')
    p.add_argument('--high_k_alpha', type=float, default=2.0,
                   help='PS bin weight = k^alpha (boost high k)')
    p.add_argument('--ema_decay',    type=float, default=0.999)
    p.add_argument('--save_every',   type=int, default=10)
    p.add_argument('--num_workers',  type=int, default=0)
    p.add_argument('--patches_per_cube', type=int, default=4)
    return p.parse_args()


def ps_loss_hi_k(final, target, n_bins=30, high_k_alpha=2.0):
    """PS loss with high-k emphasis: weights ∝ k^alpha."""
    final  = final.float()
    target = target.float()
    B, _, N, _, _ = final.shape
    device = final.device

    def _ps(x):
        xm = x - x.mean(dim=(-3, -2, -1), keepdim=True)
        F3 = torch.fft.rfftn(xm, dim=(-3, -2, -1))
        return F3.abs().pow(2) / (N ** 3)

    Pk_r = _ps(final.squeeze(1))
    Pk_t = _ps(target.squeeze(1))

    k1  = torch.fft.fftfreq(N,  device=device)
    k1r = torch.fft.rfftfreq(N, device=device)
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = (KX**2 + KY**2 + KZ**2).sqrt()
    kmax = Kmag.max().item()
    edges = torch.linspace(0, kmax + 1e-6, n_bins + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # k^alpha weighting boosts high k
    raw_w = (centers + 1e-3).pow(high_k_alpha)
    weights = raw_w * n_bins / raw_w.sum()

    loss = torch.tensor(0.0, device=device)
    used = 0
    for i in range(n_bins):
        mask = (Kmag >= edges[i]) & (Kmag < edges[i + 1])
        if not mask.any():
            continue
        pr = Pk_r[:, mask].mean(-1)
        pt = Pk_t[:, mask].mean(-1)
        loss = loss + weights[i] * F.mse_loss(pr.log1p(), pt.log1p())
        used += 1
    return loss / max(used, 1)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)


def make_dataset(root, patch_size, redshifts, split, patches_per_cube=1):
    if root and os.path.isdir(root):
        return T21Dataset(root, patch_size, redshifts=redshifts, split=split,
                          patches_per_cube=patches_per_cube, load_ic=False)
    return None


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}  |  AMP: {device.type=='cuda'}")
    print(f"Residual VAE | high_k_alpha={args.high_k_alpha} ps_w={args.ps_weight}")

    # ---- Data ----
    def build_split(split):
        sets = []
        for root in [args.data_root_ic, args.data_root_astro]:
            ds = make_dataset(root, args.patch_size, args.redshifts, split,
                              args.patches_per_cube)
            if ds:
                sets.append(ds)
                print(f"  {split} {os.path.basename(root)}: {len(ds)}")
        return ConcatDataset(sets) if len(sets) > 1 else sets[0]

    train_ds = build_split('train')
    val_ds   = build_split('val')
    print(f"train={len(train_ds)} val={len(val_ds)}")

    mp_ctx = 'spawn' if args.num_workers > 0 else None
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              multiprocessing_context=mp_ctx)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              multiprocessing_context=mp_ctx)

    # ---- Frozen low-freq VAE ----
    vae_low = VAE3D(in_ch=1, latent_ch=args.vae_latent_ch,
                    base_ch=args.vae_base_ch, ch_mults=(1, 2)).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae_low.load_state_dict(ckpt['model'])
    vae_low.eval()
    for p in vae_low.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen VAE from {args.vae_ckpt} (epoch {ckpt.get('epoch','?')})")

    # ---- Residual VAE: 1x downsample only, smaller channels ----
    res_vae = VAE3D(in_ch=1, latent_ch=args.res_latent_ch,
                    base_ch=args.res_base_ch, ch_mults=(1,)).to(device)
    print(f"Residual VAE params: {sum(p.numel() for p in res_vae.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(res_vae.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    ema = EMA(res_vae, decay=args.ema_decay)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        rc = torch.load(args.resume, map_location=device)
        res_vae.load_state_dict(rc['model'])
        opt.load_state_dict(rc['opt'])
        if 'scheduler' in rc:
            scheduler.load_state_dict(rc['scheduler'])
        if 'ema' in rc:
            ema.shadow = {k: v.to(device) for k, v in rc['ema'].items()}
        start_epoch = rc['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,res_mse,ps_loss,kl,val_loss,val_res_mse\n')

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        res_vae.train()
        t0 = time.time()
        tr = dict(loss=0, res_mse=0, ps=0, kl=0)

        for batch in train_loader:
            x = batch['patch'].to(device)

            # Frozen forward (no grad)
            with torch.no_grad():
                with autocast(enabled=(device.type == 'cuda')):
                    x_low, _, _ = vae_low(x)
            r = x - x_low

            with autocast(enabled=(device.type == 'cuda')):
                r_hat, mean, logvar = res_vae(r)
                x_final = x_low + r_hat
                l_res   = F.mse_loss(r_hat, r)
                l_kl    = VAE3D.kl_loss(mean, logvar)
                l_ps    = ps_loss_hi_k(x_final, x, high_k_alpha=args.high_k_alpha)
                loss = l_res + args.ps_weight * l_ps + args.kl_weight * l_kl

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(res_vae.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(res_vae)

            tr['loss']    += loss.item()
            tr['res_mse'] += l_res.item()
            tr['ps']      += l_ps.item()
            tr['kl']      += l_kl.item()

        scheduler.step()
        n = len(train_loader)
        for k in tr: tr[k] /= n

        # ---- Val ----
        res_vae.eval()
        vl = dict(loss=0, res_mse=0)
        with torch.no_grad():
            for batch in val_loader:
                x = batch['patch'].to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    x_low, _, _ = vae_low(x)
                    r = x - x_low
                    r_hat, mean, logvar = res_vae(r)
                    l_res = F.mse_loss(r_hat, r)
                vl['loss']    += l_res.item()
                vl['res_mse'] += l_res.item()
        nv = len(val_loader)
        for k in vl: vl[k] /= nv

        print(f"Ep {epoch:4d} | tr {tr['loss']:.4f} "
              f"(res {tr['res_mse']:.4f} ps {tr['ps']:.4f} kl {tr['kl']:.4f}) | "
              f"val res {vl['res_mse']:.4f} | {time.time()-t0:.0f}s", flush=True)

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{tr['loss']:.6f},{tr['res_mse']:.6f},"
                    f"{tr['ps']:.6f},{tr['kl']:.6f},"
                    f"{vl['loss']:.6f},{vl['res_mse']:.6f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.out_dir, f'residual_epoch{epoch:04d}.pt')
            torch.save({
                'epoch':     epoch,
                'model':     res_vae.state_dict(),
                'ema':       ema.shadow,
                'opt':       opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args':      vars(args),
            }, path)
            print(f"  Saved {path}")

    print("Residual VAE training complete.")


if __name__ == '__main__':
    main()
