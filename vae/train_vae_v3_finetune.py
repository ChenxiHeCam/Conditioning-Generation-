"""
v3 finetune of the 3D VAE.

Goal: fix the large-scale PS deficit observed in v2 ep249.

Changes vs v2:
  1. k-weighted PS loss: low-k bins get higher weight (w_k = 1/(k+eps)),
     so the model can't trade DC/low-k accuracy for high-k detail.
  2. Free-bits KL: each latent channel has a KL floor (`kl_free_bits` nats).
     Stops KL from collapsing channels that carry DC info, but still
     regularises latent overall.
  3. No ramp - all loss terms are at full weight from epoch 0.
  4. Loads model weights from --resume but resets opt/scheduler so we
     get a fresh fixed-LR finetune.

Usage:
  python train_vae_v3.py \
    --resume /root/autodl-tmp/checkpoints/vae_v2/vae_epoch0249.pt \
    --out_dir /root/autodl-tmp/checkpoints/vae_v3 \
    --epochs 60 --lr 5e-5 --kl_free_bits 0.5 --low_k_alpha 1.0
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
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--out_dir',         default='/root/autodl-tmp/checkpoints/vae_v3')
    p.add_argument('--resume',          required=True)
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--patch_size',  type=int,   default=64)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--lr',          type=float, default=5e-5)
    p.add_argument('--epochs',      type=int,   default=60)
    p.add_argument('--kl_weight',   type=float, default=3e-5)
    p.add_argument('--kl_free_bits',type=float, default=0.5,
                   help='per-channel KL floor in nats')
    p.add_argument('--ps_weight',   type=float, default=0.15)
    p.add_argument('--spec_weight', type=float, default=0.03)
    p.add_argument('--low_k_alpha', type=float, default=1.0,
                   help='PS bin weight = 1/(k+eps)^alpha; 0 => uniform, 1 => 1/k')
    p.add_argument('--base_ch',     type=int,   default=64)
    p.add_argument('--latent_ch',   type=int,   default=8)
    p.add_argument('--ema_decay',   type=float, default=0.999)
    p.add_argument('--save_every',  type=int,   default=10)
    p.add_argument('--num_workers',      type=int, default=0)
    p.add_argument('--patches_per_cube', type=int, default=4)
    return p.parse_args()


def spectral_loss(recon, target):
    r = recon.squeeze(1).float()
    t = target.squeeze(1).float()
    fr = torch.fft.rfftn(r, dim=(-3, -2, -1), norm='ortho')
    ft = torch.fft.rfftn(t, dim=(-3, -2, -1), norm='ortho')
    return F.mse_loss(fr.abs().log1p(), ft.abs().log1p())


def ps_loss_kweighted(recon, target, n_bins=30, low_k_alpha=1.0):
    """
    k-weighted PS loss. Bin weights ∝ 1/(k_center + eps)^alpha, normalized
    so average weight = 1. With alpha=0 this reduces to uniform per-bin avg.
    """
    recon  = recon.float()
    target = target.float()
    B, _, N, _, _ = recon.shape
    device = recon.device

    def _ps(x):
        xm = x - x.mean(dim=(-3, -2, -1), keepdim=True)
        F3 = torch.fft.rfftn(xm, dim=(-3, -2, -1))
        return F3.abs().pow(2) / (N ** 3)

    Pk_r = _ps(recon.squeeze(1))
    Pk_t = _ps(target.squeeze(1))

    k1  = torch.fft.fftfreq(N,  device=device)
    k1r = torch.fft.rfftfreq(N, device=device)
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = (KX**2 + KY**2 + KZ**2).sqrt()
    kmax = Kmag.max().item()
    edges = torch.linspace(0, kmax + 1e-6, n_bins + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # bin weights: 1/(k+eps)^alpha, normalised so they average to 1
    eps_k = 0.5 / N            # ~smallest non-zero freq
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


def kl_free_bits(mean, logvar, free_bits=0.5):
    """
    Per-channel free-bits KL.
    mean, logvar: (B, C, D, H, W). KL is averaged over (B,D,H,W) per channel,
    then max'd with free_bits, then averaged across C.
    """
    # kl_per_dim: (B, C, D, H, W)
    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    # average over (B, spatial): per-channel KL
    kl_per_ch = kl.mean(dim=(0, 2, 3, 4))            # (C,)
    return torch.clamp_min(kl_per_ch, free_bits).mean()


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
    print(f"v3 finetune | low_k_alpha={args.low_k_alpha} kl_free_bits={args.kl_free_bits}")
    print(f"  kl_w={args.kl_weight} ps_w={args.ps_weight} spec_w={args.spec_weight} lr={args.lr}")

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
    print(f"Total  train={len(train_ds)}  val={len(val_ds)}")

    mp_ctx = 'spawn' if args.num_workers > 0 else None
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              multiprocessing_context=mp_ctx)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              multiprocessing_context=mp_ctx)

    model = VAE3D(in_ch=1, latent_ch=args.latent_ch,
                  base_ch=args.base_ch, ch_mults=(1, 2)).to(device)
    print(f"VAE params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Load weights from v2 ckpt, but use a fresh optimizer/scheduler
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded weights from epoch {ckpt.get('epoch', '?')}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    ema = EMA(model, decay=args.ema_decay)
    if 'ema' in ckpt:
        ema.shadow = {k: v.to(device) for k, v in ckpt['ema'].items()}

    log_path = os.path.join(args.out_dir, 'log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,recon,kl,ps,spec,val_loss,val_recon,val_kl\n')

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tr = dict(loss=0, recon=0, kl=0, ps=0, spec=0)

        for batch in train_loader:
            x = batch['patch'].to(device)
            with autocast(enabled=(device.type == 'cuda')):
                recon, mean, logvar = model(x)
                l_recon = F.mse_loss(recon, x)
                l_kl    = kl_free_bits(mean, logvar, args.kl_free_bits)
                l_ps    = ps_loss_kweighted(recon, x, low_k_alpha=args.low_k_alpha)
                l_spec  = spectral_loss(recon, x)
                loss = (l_recon
                        + args.kl_weight   * l_kl
                        + args.ps_weight   * l_ps
                        + args.spec_weight * l_spec)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            tr['loss']  += loss.item()
            tr['recon'] += l_recon.item()
            tr['kl']    += l_kl.item()
            tr['ps']    += l_ps.item()
            tr['spec']  += l_spec.item()

        scheduler.step()
        n = len(train_loader)
        for k in tr: tr[k] /= n

        model.eval()
        vl = dict(loss=0, recon=0, kl=0)
        with torch.no_grad():
            for batch in val_loader:
                x = batch['patch'].to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    recon, mean, logvar = model(x)
                    l_recon = F.mse_loss(recon, x)
                    l_kl    = kl_free_bits(mean, logvar, args.kl_free_bits)
                    loss    = l_recon + args.kl_weight * l_kl
                vl['loss']  += loss.item()
                vl['recon'] += l_recon.item()
                vl['kl']    += l_kl.item()
        nv = len(val_loader)
        for k in vl: vl[k] /= nv

        print(f"Ep {epoch:4d} | tr {tr['loss']:.4f} "
              f"(mse {tr['recon']:.4f} kl {tr['kl']:.4f} "
              f"ps {tr['ps']:.4f} sp {tr['spec']:.4f}) | "
              f"val {vl['loss']:.4f} (mse {vl['recon']:.4f}) | "
              f"{time.time()-t0:.0f}s", flush=True)

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{tr['loss']:.6f},{tr['recon']:.6f},{tr['kl']:.6f},"
                    f"{tr['ps']:.6f},{tr['spec']:.6f},"
                    f"{vl['loss']:.6f},{vl['recon']:.6f},{vl['kl']:.6f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.out_dir, f'vae_v3_epoch{epoch:04d}.pt')
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'ema':       ema.shadow,
                'opt':       opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args':      vars(args),
            }, path)
            print(f"  Saved {path}")

    print("v3 finetune complete.")


if __name__ == '__main__':
    main()
