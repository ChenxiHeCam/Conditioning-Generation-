"""
Train a LOW-K residual VAE on top of frozen v3 + frozen res_hi.

This is the second expert of the MoE-style decomposition:
  x_low  = v3(x)              ← mid scales (frozen)
  r_hi   = res_hi(x - x_low)  ← high-k expert (frozen)
  r_lo   = res_lo(target)     ← THIS: low-k expert
  x_final = x_low + r_hi + r_lo

res_lo architecture (ch_mults=(1,2,2), latent 8^3 x 8) physically forbids
encoding k > ~0.06, so by construction it can only carry low-k info.
The PS loss uses low_k_alpha to further focus on large scales.

Usage:
  python train_residual_lo.py \
    --vae_ckpt    /root/autodl-tmp/checkpoints/vae_v3/vae_v3_epoch0059.pt \
    --res_hi_ckpt /root/autodl-tmp/checkpoints/vae_residual/residual_epoch0019.pt \
    --out_dir     /root/autodl-tmp/checkpoints/vae_residual_lo \
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
    p.add_argument('--vae_ckpt',         required=True)
    p.add_argument('--res_hi_ckpt',      required=True)
    p.add_argument('--data_root_ic',     default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro',  default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--out_dir',          default='/root/autodl-tmp/checkpoints/vae_residual_lo')
    p.add_argument('--resume',           default=None)
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--patch_size',  type=int,   default=64)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=60)
    # Frozen low-freq VAE config
    p.add_argument('--vae_latent_ch', type=int, default=8)
    p.add_argument('--vae_base_ch',   type=int, default=64)
    # Frozen high-k residual config
    p.add_argument('--res_hi_latent_ch', type=int, default=4)
    p.add_argument('--res_hi_base_ch',   type=int, default=32)
    # NEW low-k residual config (heavy compression)
    p.add_argument('--res_lo_latent_ch', type=int, default=8)
    p.add_argument('--res_lo_base_ch',   type=int, default=32)
    # Loss weights
    p.add_argument('--kl_weight',    type=float, default=1e-5)
    p.add_argument('--lk_weight',    type=float, default=1.0,
                   help='weight for phase-aware low-k complex Fourier MSE')
    p.add_argument('--k_cut',        type=float, default=0.12,
                   help='cutoff for low-k loss; modes with |k|<k_cut included')
    p.add_argument('--filter_k_cut', type=float, default=0.1,
                   help='post-filter cutoff: lowpass res_lo output below this k')
    p.add_argument('--ps_weight',    type=float, default=0.0,
                   help='[deprecated/optional] PS loss weight; default off')
    p.add_argument('--mean_weight',  type=float, default=1.0,
                   help='per-patch DC preservation: (mean(x_final) - mean(x))^2')
    p.add_argument('--low_k_alpha',  type=float, default=2.0,
                   help='PS bin weight = 1/(k+eps)^alpha')
    p.add_argument('--ema_decay',    type=float, default=0.999)
    p.add_argument('--save_every',   type=int, default=10)
    p.add_argument('--num_workers',  type=int, default=0)
    p.add_argument('--patches_per_cube', type=int, default=4)
    return p.parse_args()


def lowpass_3d(x, k_cut):
    """FFT lowpass: zero out modes with |k| >= k_cut.
    Removes ConvTranspose3d checkerboard artifacts at high k."""
    orig_dtype = x.dtype
    x32 = x.squeeze(1).float()
    X = torch.fft.rfftn(x32, dim=(-3, -2, -1))
    N = x32.shape[-1]
    device = x32.device
    k1  = torch.fft.fftfreq(N,  device=device)
    k1r = torch.fft.rfftfreq(N, device=device)
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = (KX**2 + KY**2 + KZ**2).sqrt()
    mask = (Kmag < k_cut).to(X.dtype)
    X = X * mask
    out = torch.fft.irfftn(X, s=x32.shape[-3:], dim=(-3, -2, -1))
    return out.unsqueeze(1).to(orig_dtype)


def low_k_complex_loss(final, target, k_cut=0.15):
    """
    Phase-aware low-k MSE in 3-D Fourier space.
    Penalises both magnitude and phase mismatches at |k| < k_cut.
    """
    r = final.squeeze(1).float()
    t = target.squeeze(1).float()
    # mean-subtract so DC doesn't dominate (DC handled by mean_loss term)
    r = r - r.mean(dim=(-3, -2, -1), keepdim=True)
    t = t - t.mean(dim=(-3, -2, -1), keepdim=True)
    fr = torch.fft.rfftn(r, dim=(-3, -2, -1), norm='ortho')
    ft = torch.fft.rfftn(t, dim=(-3, -2, -1), norm='ortho')
    N = r.shape[-1]
    device = r.device
    k1  = torch.fft.fftfreq(N,  device=device)
    k1r = torch.fft.rfftfreq(N, device=device)
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = (KX**2 + KY**2 + KZ**2).sqrt()
    mask = (Kmag < k_cut).float()
    diff = (fr - ft).abs().pow(2) * mask
    return diff.sum() / (mask.sum() * fr.shape[0])


def ps_loss_lo_k(final, target, n_bins=30, low_k_alpha=2.0):
    """PS loss with low-k emphasis: weights ∝ 1/(k+eps)^alpha."""
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

    eps_k = 0.5 / N
    raw_w = 1.0 / (centers + eps_k).pow(low_k_alpha)
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
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
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
    print(f"res_lo expert | low_k_alpha={args.low_k_alpha} ps_w={args.ps_weight}")

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

    # ---- Frozen v3 ----
    vae_low = VAE3D(in_ch=1, latent_ch=args.vae_latent_ch,
                    base_ch=args.vae_base_ch, ch_mults=(1, 2)).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae_low.load_state_dict(ckpt['model'])
    vae_low.eval()
    for p in vae_low.parameters(): p.requires_grad_(False)
    print(f"Loaded v3 from {args.vae_ckpt} (epoch {ckpt.get('epoch','?')})")

    # ---- Frozen res_hi ----
    res_hi = VAE3D(in_ch=1, latent_ch=args.res_hi_latent_ch,
                   base_ch=args.res_hi_base_ch, ch_mults=(1,)).to(device)
    rh = torch.load(args.res_hi_ckpt, map_location=device)
    res_hi.load_state_dict(rh['model'])
    res_hi.eval()
    for p in res_hi.parameters(): p.requires_grad_(False)
    print(f"Loaded res_hi from {args.res_hi_ckpt} (epoch {rh.get('epoch','?')})")

    # ---- Trainable res_lo: 3 downsamples -> 8^3 latent ----
    res_lo = VAE3D(in_ch=1, latent_ch=args.res_lo_latent_ch,
                   base_ch=args.res_lo_base_ch, ch_mults=(1, 2, 2)).to(device)
    print(f"res_lo params: {sum(p.numel() for p in res_lo.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(res_lo.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    ema = EMA(res_lo, decay=args.ema_decay)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        rc = torch.load(args.resume, map_location=device)
        res_lo.load_state_dict(rc['model'])
        opt.load_state_dict(rc['opt'])
        if 'scheduler' in rc: scheduler.load_state_dict(rc['scheduler'])
        if 'ema' in rc: ema.shadow = {k: v.to(device) for k, v in rc['ema'].items()}
        start_epoch = rc['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,res_mse,ps_loss,kl,val_loss,val_res_mse\n')

    for epoch in range(start_epoch, args.epochs):
        res_lo.train()
        t0 = time.time()
        tr = dict(loss=0, res_mse=0, ps=0, kl=0)

        for batch in train_loader:
            x = batch['patch'].to(device)

            with torch.no_grad():
                with autocast(enabled=(device.type == 'cuda')):
                    x_low, _, _   = vae_low(x)
                    r_hi_hat, _, _ = res_hi(x - x_low)
            target_lo = x - x_low - r_hi_hat   # residual after high-k removal

            with autocast(enabled=(device.type == 'cuda')):
                r_lo_hat, mean, logvar = res_lo(target_lo)
                # Post-filter: band-limit res_lo output to k < filter_k_cut
                # so ConvTranspose3d checkerboard artifacts at high k are removed
                r_lo_hat = lowpass_3d(r_lo_hat, args.filter_k_cut)
                x_final = x_low + r_hi_hat + r_lo_hat
                l_kl  = VAE3D.kl_loss(mean, logvar)
                # Phase-aware low-k complex Fourier MSE (drives the model)
                l_lk = low_k_complex_loss(x_final, x, k_cut=args.k_cut)
                # PS power-only loss (optional, off by default)
                l_ps = ps_loss_lo_k(x_final, x, low_k_alpha=args.low_k_alpha) \
                       if args.ps_weight > 0 else torch.tensor(0., device=device)
                # per-patch DC preservation
                mean_x       = x.mean(dim=(2, 3, 4))
                mean_x_final = x_final.mean(dim=(2, 3, 4))
                l_mean = F.mse_loss(mean_x_final, mean_x)
                loss = (args.lk_weight   * l_lk
                        + args.ps_weight * l_ps
                        + args.mean_weight * l_mean
                        + args.kl_weight * l_kl)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(res_lo.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(res_lo)

            tr['loss']    += loss.item()
            tr['res_mse'] += l_lk.item()     # repurposed: log low-k complex loss
            tr['ps']      += l_mean.item()   # repurposed: log mean-preservation loss
            tr['kl']      += l_kl.item()

        scheduler.step()
        n = len(train_loader)
        for k in tr: tr[k] /= n

        res_lo.eval()
        vl = dict(loss=0, res_mse=0)
        with torch.no_grad():
            for batch in val_loader:
                x = batch['patch'].to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    x_low, _, _    = vae_low(x)
                    r_hi_hat, _, _ = res_hi(x - x_low)
                    target_lo = x - x_low - r_hi_hat
                    r_lo_hat, _, _ = res_lo(target_lo)
                    r_lo_hat = lowpass_3d(r_lo_hat, args.filter_k_cut)
                    x_final = x_low + r_hi_hat + r_lo_hat
                    l_full = F.mse_loss(x_final, x)
                vl['loss']    += l_full.item()
                vl['res_mse'] += l_full.item()
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
            path = os.path.join(args.out_dir, f'residual_lo_epoch{epoch:04d}.pt')
            torch.save({
                'epoch':     epoch,
                'model':     res_lo.state_dict(),
                'ema':       ema.shadow,
                'opt':       opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args':      vars(args),
            }, path)
            print(f"  Saved {path}")

    print("res_lo training complete.")


if __name__ == '__main__':
    main()
