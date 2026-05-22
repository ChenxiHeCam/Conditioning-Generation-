"""
Train the 3D VAE on T21 brightness temperature patches.

Staged loss schedule:
  Stage 1 (0 .. ps_start):          MSE + KL
  Stage 2 (ps_start .. spec_start):  + power-spectrum loss (ramp over 50 ep)
  Stage 3 (spec_start .. end):       + spectral FFT loss   (ramp over 50 ep)

Usage:
  python train_vae.py \
    --data_root_ic   /root/autodl-tmp/ASR21cm/varying_IC \
    --data_root_astro /root/autodl-tmp/ASR21cm/varying_astro \
    --out_dir /root/autodl-tmp/checkpoints/vae
"""
import os, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast

from dataset import T21Dataset
from models.vae import VAE3D


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--out_dir',         default='/root/autodl-tmp/checkpoints/vae')
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--patch_size',  type=int,   default=64)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=300)
    p.add_argument('--kl_weight',   type=float, default=1e-4)
    p.add_argument('--kl_anneal',   type=int,   default=50)
    p.add_argument('--ps_weight',   type=float, default=0.05,
                   help='max weight for PS loss (stage 2)')
    p.add_argument('--spec_weight', type=float, default=0.02,
                   help='max weight for spectral FFT loss (stage 3)')
    p.add_argument('--ps_start',    type=int,   default=100,
                   help='epoch to start ramping in PS loss')
    p.add_argument('--spec_start',  type=int,   default=150,
                   help='epoch to start ramping in spectral loss')
    p.add_argument('--ramp_epochs', type=int,   default=50,
                   help='epochs to ramp each staged loss from 0 to max')
    p.add_argument('--base_ch',     type=int,   default=64)
    p.add_argument('--latent_ch',   type=int,   default=4)
    p.add_argument('--ch_mults', nargs='+', type=int, default=[1, 2],
                   help='channel multipliers per downsample stage; '
                        'len(ch_mults) controls how many 2x downsamples')
    p.add_argument('--ema_decay',   type=float, default=0.999)
    p.add_argument('--save_every',  type=int,   default=25)
    p.add_argument('--resume',      default=None)
    p.add_argument('--num_workers',      type=int,   default=4)
    p.add_argument('--patches_per_cube', type=int,   default=1,
                   help='number of random patches drawn from each cube per epoch')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def ramp_weight(epoch, start, ramp, max_w):
    """Linear ramp: 0 before `start`, reaches `max_w` after `start+ramp`."""
    if epoch < start:
        return 0.0
    return max_w * min(1.0, (epoch - start) / max(1, ramp))


def spectral_loss(recon, target):
    """
    Log-magnitude MSE in 3-D Fourier space.
    Penalises mismatches at ALL frequencies (including high-k).
    Differentiable via torch.fft.
    """
    # Cast to fp32 — complex fp16 is not supported under AMP
    r = recon.squeeze(1).float()
    t = target.squeeze(1).float()
    fr = torch.fft.rfftn(r, dim=(-3, -2, -1), norm='ortho')
    ft = torch.fft.rfftn(t, dim=(-3, -2, -1), norm='ortho')
    return F.mse_loss(fr.abs().log1p(), ft.abs().log1p())


def ps_loss(recon, target, n_bins=30):
    """
    Differentiable spherically-averaged power spectrum loss.
    Uses log-MSE on Δ²(k) bins computed via 3-D FFT in PyTorch.
    """
    # Cast to fp32 — complex fp16 is not supported under AMP
    recon  = recon.float()
    target = target.float()
    B, _, N, _, _ = recon.shape
    device = recon.device

    def _ps(x):
        # x: (B, N, N, N)
        xm = x - x.mean(dim=(-3, -2, -1), keepdim=True)
        F3 = torch.fft.rfftn(xm, dim=(-3, -2, -1))
        Pk = F3.abs().pow(2) / (N ** 3)     # (B, N, N, N//2+1)
        return Pk

    Pk_r = _ps(recon.squeeze(1))
    Pk_t = _ps(target.squeeze(1))

    # Build |k| grid once (cache via closure captures device)
    k1  = torch.fft.fftfreq(N,  device=device)
    k1r = torch.fft.rfftfreq(N, device=device)
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = (KX**2 + KY**2 + KZ**2).sqrt()          # (N, N, N//2+1)
    kmax = Kmag.max().item()
    edges = torch.linspace(0, kmax + 1e-6, n_bins + 1, device=device)

    loss = torch.tensor(0.0, device=device)
    for i in range(n_bins):
        mask = (Kmag >= edges[i]) & (Kmag < edges[i + 1])
        if not mask.any():
            continue
        pr = Pk_r[:, mask].mean(-1)   # (B,)
        pt = Pk_t[:, mask].mean(-1)
        loss = loss + F.mse_loss(pr.log1p(), pt.log1p())

    return loss / n_bins


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model):
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # ---- Datasets ----
    def build_split(split):
        sets = []
        for root in [args.data_root_ic, args.data_root_astro]:
            ds = make_dataset(root, args.patch_size, args.redshifts, split,
                              args.patches_per_cube)
            if ds:
                sets.append(ds)
                print(f"  {split} {os.path.basename(root)}: {len(ds)}")
        assert sets, "No valid data roots found"
        return ConcatDataset(sets) if len(sets) > 1 else sets[0]

    train_ds = build_split('train')
    val_ds   = build_split('val')
    test_ds  = build_split('test')
    print(f"Total  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # Use 'spawn' context to avoid h5py deadlock in forked workers
    mp_ctx = 'spawn' if args.num_workers > 0 else None
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              multiprocessing_context=mp_ctx)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              multiprocessing_context=mp_ctx)

    # ---- Model ----
    model = VAE3D(in_ch=1, latent_ch=args.latent_ch,
                  base_ch=args.base_ch, ch_mults=tuple(args.ch_mults)).to(device)
    print(f"VAE params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler    = GradScaler(enabled=(device.type == 'cuda'))
    ema       = EMA(model, decay=args.ema_decay)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        if 'ema' in ckpt:
            ema.shadow = {k: v.to(device) for k, v in ckpt['ema'].items()}
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,recon,kl,ps,spec,val_loss,val_recon,val_kl,kl_w,ps_w,spec_w\n')

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        kl_w   = args.kl_weight * min(1.0, epoch / max(1, args.kl_anneal))
        ps_w   = ramp_weight(epoch, args.ps_start,   args.ramp_epochs, args.ps_weight)
        spec_w = ramp_weight(epoch, args.spec_start, args.ramp_epochs, args.spec_weight)

        model.train()
        t0 = time.time()
        tr = dict(loss=0, recon=0, kl=0, ps=0, spec=0)

        for batch in train_loader:
            x = batch['patch'].to(device)

            with autocast(enabled=(device.type == 'cuda')):
                recon, mean, logvar = model(x)
                l_recon = F.mse_loss(recon, x)
                l_kl    = VAE3D.kl_loss(mean, logvar)
                l_ps    = ps_loss(recon, x)    if ps_w   > 0 else torch.tensor(0.)
                l_spec  = spectral_loss(recon, x) if spec_w > 0 else torch.tensor(0.)
                loss = l_recon + kl_w * l_kl + ps_w * l_ps + spec_w * l_spec

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

        # ---- Validation ----
        model.eval()
        vl = dict(loss=0, recon=0, kl=0)
        with torch.no_grad():
            for batch in val_loader:
                x = batch['patch'].to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    recon, mean, logvar = model(x)
                    l_recon = F.mse_loss(recon, x)
                    l_kl    = VAE3D.kl_loss(mean, logvar)
                    loss    = l_recon + kl_w * l_kl
                vl['loss']  += loss.item()
                vl['recon'] += l_recon.item()
                vl['kl']    += l_kl.item()
        nv = len(val_loader)
        for k in vl: vl[k] /= nv

        print(f"Ep {epoch:4d} | "
              f"tr {tr['loss']:.4f} (mse {tr['recon']:.4f} kl {tr['kl']:.4f} "
              f"ps {tr['ps']:.4f} sp {tr['spec']:.4f}) | "
              f"val {vl['loss']:.4f} | "
              f"kl_w={kl_w:.1e} ps_w={ps_w:.1e} sp_w={spec_w:.1e} | "
              f"{time.time()-t0:.0f}s")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{tr['loss']:.6f},{tr['recon']:.6f},{tr['kl']:.6f},"
                    f"{tr['ps']:.6f},{tr['spec']:.6f},"
                    f"{vl['loss']:.6f},{vl['recon']:.6f},{vl['kl']:.6f},"
                    f"{kl_w:.4e},{ps_w:.4e},{spec_w:.4e}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.out_dir, f'vae_epoch{epoch:04d}.pt')
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'ema':       ema.shadow,
                'opt':       opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args':      vars(args),
            }, path)
            print(f"  Saved {path}")

    print("Training complete.")


if __name__ == '__main__':
    main()
