"""
Evaluate trained VAE: reconstruction quality + latent distribution + power spectrum.
Usage:
    python evaluate_vae.py --ckpt checkpoints/vae/vae_epoch0299.pt
"""
import os, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import T21Dataset
from models.vae import VAE3D
from utils.power_spectrum import power_spectrum


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',   default='/home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro')
    p.add_argument('--ckpt',        required=True)
    p.add_argument('--out_dir',     default='eval_results/vae')
    p.add_argument('--redshifts',   nargs='+', type=int, default=[10])
    p.add_argument('--patch_size',  type=int, default=64)
    p.add_argument('--n_samples',   type=int, default=32)
    p.add_argument('--latent_ch',   type=int, default=4)
    p.add_argument('--base_ch',     type=int, default=64)
    p.add_argument('--Lpix',        type=float, default=3.0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # Load VAE
    vae = VAE3D(in_ch=1, latent_ch=args.latent_ch,
                base_ch=args.base_ch, ch_mults=(1, 2)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    vae.load_state_dict(ckpt['model'])
    vae.eval()
    epoch = ckpt.get('epoch', '?')
    print(f"Loaded VAE epoch {epoch}")

    # Data
    ds = T21Dataset(args.data_root, args.patch_size,
                    redshifts=args.redshifts, split='val')
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    originals, recons, latents = [], [], []
    for batch in loader:
        x = batch['patch'].to(device)
        z, mu, logvar = vae.encode(x)
        x_hat = vae.decode(z)
        originals.append(x.cpu())
        recons.append(x_hat.cpu())
        latents.append(z.cpu())
        if sum(o.shape[0] for o in originals) >= args.n_samples:
            break

    orig = torch.cat(originals)[:args.n_samples]
    rec  = torch.cat(recons)[:args.n_samples]
    lat  = torch.cat(latents)[:args.n_samples]

    # ── 1. Reconstruction MSE ──────────────────────────────────────────
    mse = ((orig - rec) ** 2).mean().item()
    rel = (((orig - rec)**2).sum() / (orig**2).sum()).item()
    print(f"\nReconstruction MSE:          {mse:.6f}")
    print(f"Relative MSE (||x-xhat||/||x||): {rel:.4f}  ({'good <0.05' if rel<0.05 else 'high >0.05'})")

    # ── 2. Latent distribution ─────────────────────────────────────────
    lat_mean = lat.mean().item()
    lat_std  = lat.std().item()
    print(f"\nLatent mean: {lat_mean:.4f}  (should be ~0)")
    print(f"Latent std:  {lat_std:.4f}  (should be ~1)")

    # ── 3. Power spectrum ─────────────────────────────────────────────
    # Denormalise
    orig_phys = orig * ds.t21_std + ds.t21_mean
    rec_phys  = rec  * ds.t21_std + ds.t21_mean
    k, ps_orig = power_spectrum(orig_phys, Lpix=args.Lpix, kbins=30)
    k, ps_rec  = power_spectrum(rec_phys,  Lpix=args.Lpix, kbins=30)
    ratio = ps_rec.mean(0) / (ps_orig.mean(0) + 1e-30)
    print(f"\nPS ratio (recon/orig):")
    print(f"  Large scale (k<0.1):  {ratio[k<0.1].mean():.3f}")
    print(f"  Mid scale:            {ratio[(k>=0.1)&(k<0.5)].mean():.3f}")
    print(f"  Small scale (k>0.5):  {ratio[k>=0.5].mean():.3f}")

    # ── Plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Row 0: slices
    for i in range(3):
        mid = orig.shape[-1] // 2
        axes[0, i].imshow(orig[i, 0, :, :, mid], cmap='RdBu_r')
        axes[0, i].set_title(f'Original #{i}'); axes[0, i].axis('off')
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        mid = rec.shape[-1] // 2
        axes2[i].imshow(rec[i, 0, :, :, mid], cmap='RdBu_r')
        axes2[i].set_title(f'Recon #{i}'); axes2[i].axis('off')
    plt.tight_layout()
    fig2.savefig(os.path.join(args.out_dir, 'recon_slices.png'), dpi=150); plt.close(fig2)

    # Power spectrum
    ax = axes[0, 0]
    ax.loglog(k, ps_orig.mean(0), label='Original', color='C1', linestyle='--')
    ax.fill_between(k, ps_orig.mean(0)-ps_orig.std(0), ps_orig.mean(0)+ps_orig.std(0), alpha=0.2, color='C1')
    ax.loglog(k, ps_rec.mean(0),  label='Recon',    color='C0')
    ax.fill_between(k, ps_rec.mean(0)-ps_rec.std(0),  ps_rec.mean(0)+ps_rec.std(0),  alpha=0.2, color='C0')
    ax.set_xlabel('k'); ax.set_ylabel('Δ²(k)'); ax.legend(); ax.set_title('Power Spectrum')

    # PS ratio
    ax = axes[0, 1]
    ax.semilogx(k, ratio, color='C0')
    ax.axhline(1.0, color='gray', linestyle='--')
    ax.axhspan(0.95, 1.05, color='gray', alpha=0.1)
    ax.set_ylim(0.5, 1.5); ax.set_xlabel('k'); ax.set_ylabel('Recon/Orig')
    ax.set_title('PS Ratio (1.0 = perfect)')

    # Latent histogram
    ax = axes[0, 2]
    ax.hist(lat.flatten().numpy(), bins=100, density=True, alpha=0.7)
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, np.exp(-xs**2/2)/np.sqrt(2*np.pi), 'r--', label='N(0,1)')
    ax.set_title(f'Latent dist (μ={lat_mean:.2f}, σ={lat_std:.2f})')
    ax.legend()

    # Scatter: orig vs recon pixel values
    ax = axes[1, 0]
    sample = orig[:8].flatten().numpy()[::50]
    rec_s  = rec[:8].flatten().numpy()[::50]
    ax.scatter(sample, rec_s, alpha=0.1, s=1)
    lim = max(abs(sample).max(), abs(rec_s).max())
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
    ax.set_xlabel('Original'); ax.set_ylabel('Recon')
    ax.set_title(f'Pixel scatter (MSE={mse:.4f})')

    # Per-channel latent stats
    ax = axes[1, 1]
    for c in range(lat.shape[1]):
        ax.hist(lat[:, c].flatten().numpy(), bins=50, alpha=0.5, density=True, label=f'ch{c}')
    ax.set_title('Latent per-channel'); ax.legend(fontsize=7)

    # Residual map (mean absolute error per voxel)
    ax = axes[1, 2]
    err_map = (orig - rec).abs().mean(0)[0, :, :, orig.shape[-1]//2].numpy()
    im = ax.imshow(err_map, cmap='hot')
    plt.colorbar(im, ax=ax)
    ax.set_title('Mean |error| map (mid slice)'); ax.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'vae_eval.png'), dpi=150)
    plt.close(fig)

    np.savez(os.path.join(args.out_dir, 'vae_ps.npz'), k=k, ps_orig=ps_orig, ps_rec=ps_rec)
    print(f"\nPlots saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
