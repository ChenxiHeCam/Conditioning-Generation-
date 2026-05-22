"""
VAE v4 visual eval: best/worst test-set reconstructions + per-group PDF histograms.

Generates 2 figures:
  1. vae_v4_best_worst.png — 3 best + 3 worst test samples (true vs recon slices)
  2. vae_v4_pdfs.png — voxel-value PDF overlays per (suite, z) group
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import T21Dataset
from models.vae import VAE3D

CKPT = '/root/autodl-tmp/checkpoints/vae_v4/vae_v4_final.pt'
OUTDIR = '/root/autodl-tmp/vae_v4_visual_eval'
os.makedirs(OUTDIR, exist_ok=True)


def collect_test_recons(vae, root, redshifts, device, suite_tag, max_per_z=None):
    """Encode → decode all test patches. Return list of dicts."""
    out = []
    for z in redshifts:
        try:
            ds = T21Dataset(root, 64, redshifts=[z], split='test', load_ic=False)
        except Exception as e:
            continue
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        n = 0
        with torch.no_grad():
            for batch in loader:
                x = batch['patch'].to(device)
                mu, _ = vae.encoder(x)
                recon = vae.decoder(mu)
                for i in range(x.shape[0]):
                    if max_per_z and n >= max_per_z: break
                    xi = x[i].cpu().squeeze(0).numpy()
                    ri = recon[i].cpu().squeeze(0).numpy()
                    mse = float(((ri - xi)**2).mean())
                    var = float(xi.var())
                    out.append(dict(
                        suite=suite_tag, z=z, idx=n,
                        rel_mse=mse / max(var, 1e-30),
                        mse=mse,
                        true=xi, recon=ri,
                    ))
                    n += 1
                if max_per_z and n >= max_per_z: break
    return out


def fig_best_worst(samples, out_path, n_each=3):
    """Side-by-side slice plot of n_each best + n_each worst."""
    samples = sorted(samples, key=lambda s: s['rel_mse'])
    picks = samples[:n_each] + samples[-n_each:]   # best then worst
    labels = ['BEST'] * n_each + ['WORST'] * n_each
    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 2.6 * len(picks)))
    for i, (s, lab) in enumerate(zip(picks, labels)):
        mid = s['true'].shape[-1] // 2
        true_slice  = s['true'][mid, :, :]
        recon_slice = s['recon'][mid, :, :]
        diff        = recon_slice - true_slice
        vmin = min(true_slice.min(), recon_slice.min())
        vmax = max(true_slice.max(), recon_slice.max())
        dmax = max(abs(diff).max(), 1e-6)

        ax0, ax1, ax2 = axes[i]
        im0 = ax0.imshow(true_slice,  cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax0.set_title(f'{lab}  {s["suite"]} z={s["z"]} #{s["idx"]}  '
                      f'rel_MSE={s["rel_mse"]:.4f}\n(true, mid slice)', fontsize=9)
        ax0.axis('off')
        plt.colorbar(im0, ax=ax0, fraction=0.045)

        im1 = ax1.imshow(recon_slice, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax1.set_title('recon', fontsize=9); ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.045)

        im2 = ax2.imshow(diff, cmap='seismic', vmin=-dmax, vmax=dmax)
        ax2.set_title('recon − true', fontsize=9); ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.045)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def fig_pdfs(samples, out_path):
    """Pixel-value PDF overlays per (suite, z) group."""
    from collections import defaultdict
    groups = defaultdict(list)
    for s in samples:
        groups[(s['suite'], s['z'])].append(s)

    keys = sorted(groups.keys())
    ncol = 3
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // ncol, i % ncol]
        all_true  = np.concatenate([s['true'].flatten()  for s in groups[k]])
        all_recon = np.concatenate([s['recon'].flatten() for s in groups[k]])
        lo = min(all_true.min(), all_recon.min())
        hi = max(all_true.max(), all_recon.max())
        bins = np.linspace(lo, hi, 80)
        ax.hist(all_true,  bins=bins, density=True, alpha=0.55, label='true',  color='C1')
        ax.hist(all_recon, bins=bins, density=True, alpha=0.55, label='recon', color='C0')
        ax.set_yscale('log')
        ax.set_title(f'{k[0]}  z={k[1]}  n={len(groups[k])}', fontsize=10)
        ax.set_xlabel('voxel value (normalized T21)')
        ax.set_ylabel('density (log)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    # hide unused
    for j in range(i+1, nrow*ncol):
        axes[j // ncol, j % ncol].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device)
    cfg = ckpt['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
                base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(device)
    vae.load_state_dict(ckpt['model']); vae.eval()
    print(f"Loaded VAE ep {ckpt.get('epoch')}")

    all_samples = []
    print("Collecting astro test samples (z=10) ...")
    all_samples += collect_test_recons(
        vae, '/root/autodl-tmp/ASR21cm/varying_astro',
        redshifts=[10], device=device, suite_tag='astro')
    print(f"  got {len(all_samples)} so far")

    print("Collecting IC test samples (z=8-12) ...")
    all_samples += collect_test_recons(
        vae, '/root/autodl-tmp/ASR21cm/varying_IC',
        redshifts=[8, 9, 10, 11, 12], device=device, suite_tag='IC')
    print(f"  got {len(all_samples)} total")

    # Figure 1: best/worst
    fig_best_worst(all_samples, os.path.join(OUTDIR, 'vae_v4_best_worst.png'))

    # Figure 2: PDFs per group
    fig_pdfs(all_samples, os.path.join(OUTDIR, 'vae_v4_pdfs.png'))

    print("DONE")


if __name__ == '__main__':
    main()
