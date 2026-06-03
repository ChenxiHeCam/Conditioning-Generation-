"""Plot v4 PS ratio curves: ratio = P_recon(k) / P_target(k) vs k.
   One curve per (data_source, redshift) — show where ratio deviates from 1.0.
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
from utils.power_spectrum import power_spectrum


CKPT = '/root/autodl-tmp/checkpoints/vae_v4/vae_v4_final.pt'
OUT  = '/root/autodl-tmp/ps_ratio_v4.png'


@torch.no_grad()
def compute_curve(model, ds, device, max_n=100):
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    O, R = [], []
    for batch in ld:
        x = batch['patch'].to(device)
        mu, _ = model.encoder(x)
        r = model.decoder(mu)
        O.append(x.cpu()); R.append(r.cpu())
        if sum(o.shape[0] for o in O) >= max_n: break
    X = torch.cat(O)[:max_n].squeeze(1).numpy()
    R = torch.cat(R)[:max_n].squeeze(1).numpy()
    k, ps_o = power_spectrum(X)
    _, ps_r = power_spectrum(R)
    valid = ps_o.mean(0) > 1e-20
    ratio = ps_r.mean(0) / np.maximum(ps_o.mean(0), 1e-30)
    return k, ratio, valid, ps_o.mean(0), ps_r.mean(0)


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device)
    cfg = ckpt['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'], base_ch=cfg['base_ch'],
                ch_mults=tuple(cfg['ch_mults'])).to(device)
    vae.load_state_dict(ckpt['model']); vae.eval()
    print(f"Loaded v4 ep{ckpt['epoch']}")

    runs = [
        ('astro z=10', '/root/autodl-tmp/ASR21cm/varying_astro', 10, 'val'),
        ('IC z=8',  '/root/autodl-tmp/ASR21cm/varying_IC', 8,  'val'),
        ('IC z=9',  '/root/autodl-tmp/ASR21cm/varying_IC', 9,  'val'),
        ('IC z=10', '/root/autodl-tmp/ASR21cm/varying_IC', 10, 'val'),
        ('IC z=11', '/root/autodl-tmp/ASR21cm/varying_IC', 11, 'val'),
        ('IC z=12', '/root/autodl-tmp/ASR21cm/varying_IC', 12, 'val'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: PS ratio vs k
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(runs)))
    for (tag, root, z, split), c in zip(runs, colors):
        ds = T21Dataset(root, 64, redshifts=[z], split=split, load_ic=False)
        if len(ds) == 0: continue
        k, ratio, valid, _, _ = compute_curve(vae, ds, device)
        ax.semilogx(k[valid], ratio[valid], '-o', color=c, ms=4, lw=1.5, label=tag)
    ax.axhline(1.0, color='black', ls='--', lw=0.8)
    ax.axhspan(0.95, 1.05, color='gray', alpha=0.15, label='±5% target')
    ax.set_xlabel('k  [h / Mpc]')
    ax.set_ylabel('P_recon(k) / P_target(k)')
    ax.set_title('v4 power-spectrum ratio (val split, 100 samples astro / 5-8 IC)')
    ax.set_ylim(0.7, 1.1)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')
    # Annotate band boundaries
    for kk, lab in [(0.1, 'large←|→mid'), (0.5, 'mid←|→small')]:
        ax.axvline(kk, color='red', ls=':', lw=0.8, alpha=0.6)
        ax.text(kk, 0.715, lab, color='red', fontsize=8, ha='center')

    # Panel 2: absolute PS curves (orig vs recon for astro z=10 as example)
    ax = axes[1]
    ds = T21Dataset('/root/autodl-tmp/ASR21cm/varying_astro', 64, redshifts=[10], split='val', load_ic=False)
    k, ratio, valid, ps_o, ps_r = compute_curve(vae, ds, device)
    ax.loglog(k[valid], ps_o[valid], 'k-', lw=2, label='target')
    ax.loglog(k[valid], ps_r[valid], 'C0--', lw=2, label='v4 recon')
    ax.set_xlabel('k  [h / Mpc]')
    ax.set_ylabel('P(k)')
    ax.set_title('Power spectrum overlay (astro val z=10)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f"Saved {OUT}  ({os.path.getsize(OUT)/1024:.1f} KB)")


if __name__ == '__main__':
    main()
