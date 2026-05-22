"""
Evaluate frozen v3 VAE + residual VAE.

Reports: Rel MSE and PS ratios at large/mid/small scales for:
  - x_low only (v3 baseline)
  - x_low + r_hat (residual-enhanced)
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import T21Dataset
from models.vae import VAE3D
from utils.power_spectrum import power_spectrum


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt',     required=True)
    p.add_argument('--res_ckpt',     required=True)
    p.add_argument('--data_root',    default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--redshifts',    nargs='+', type=int, default=[10])
    p.add_argument('--patch_size',   type=int, default=64)
    p.add_argument('--n_samples',    type=int, default=32)
    p.add_argument('--Lpix',         type=float, default=3.0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load v3 VAE
    vae_low = VAE3D(in_ch=1, latent_ch=8, base_ch=64, ch_mults=(1, 2)).to(device)
    ck = torch.load(args.vae_ckpt, map_location=device)
    vae_low.load_state_dict(ck['model'])
    vae_low.eval()

    # Load residual VAE
    rc = torch.load(args.res_ckpt, map_location=device)
    a = rc.get('args', {})
    res_vae = VAE3D(in_ch=1,
                    latent_ch=a.get('res_latent_ch', 4),
                    base_ch=a.get('res_base_ch', 32),
                    ch_mults=(1,)).to(device)
    res_vae.load_state_dict(rc['model'])
    res_vae.eval()

    print(f"v3 VAE: epoch {ck.get('epoch','?')}  |  residual: epoch {rc.get('epoch','?')}")

    ds = T21Dataset(args.data_root, args.patch_size,
                    redshifts=args.redshifts, split='val')
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    originals, low, final = [], [], []
    for batch in loader:
        x = batch['patch'].to(device)
        x_low, _, _ = vae_low(x)
        r_hat, _, _ = res_vae(x - x_low)
        x_final = x_low + r_hat
        originals.append(x.cpu()); low.append(x_low.cpu()); final.append(x_final.cpu())
        if sum(o.shape[0] for o in originals) >= args.n_samples:
            break

    X  = torch.cat(originals)[:args.n_samples].squeeze(1).numpy()
    L  = torch.cat(low)[:args.n_samples].squeeze(1).numpy()
    Fn = torch.cat(final)[:args.n_samples].squeeze(1).numpy()

    def metrics(recon, tag):
        mse = float(((recon - X)**2).mean())
        rel = float(np.linalg.norm(recon - X) / np.linalg.norm(X))
        # Batch-averaged PS ratio (matches evaluate_vae.py)
        k, ps_o = power_spectrum(X,     Lpix=args.Lpix)   # (50,), (B,50)
        _, ps_r = power_spectrum(recon, Lpix=args.Lpix)
        ratio = ps_r.mean(0) / np.maximum(ps_o.mean(0), 1e-30)   # (50,)
        ps_lo  = float(ratio[k < 0.1].mean())              if (k < 0.1).any() else float('nan')
        ps_mid = float(ratio[(k >= 0.1) & (k < 0.5)].mean()) if ((k >= 0.1) & (k < 0.5)).any() else float('nan')
        ps_hi  = float(ratio[k >= 0.5].mean())             if (k >= 0.5).any() else float('nan')
        print(f"\n[{tag}]")
        print(f"  Recon MSE:    {mse:.6f}")
        print(f"  Rel MSE:      {rel:.4f}")
        print(f"  PS large:     {ps_lo:.3f}")
        print(f"  PS mid:       {ps_mid:.3f}")
        print(f"  PS small:     {ps_hi:.3f}")

    metrics(L,  'v3 baseline (x_low only)')
    metrics(Fn, 'v3 + residual (x_low + r_hat)')


if __name__ == '__main__':
    main()
