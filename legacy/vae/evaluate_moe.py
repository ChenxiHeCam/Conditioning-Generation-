"""
Evaluate the 3-stage MoE stack: v3 + res_hi + res_lo.
Reports Rel MSE and PS ratios for:
  - v3 only
  - v3 + res_hi
  - v3 + res_hi + res_lo
"""
import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import T21Dataset
from models.vae import VAE3D
from utils.power_spectrum import power_spectrum


def lowpass_3d(x, k_cut):
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
    out = torch.fft.irfftn(X * mask, s=x32.shape[-3:], dim=(-3, -2, -1))
    return out.unsqueeze(1).to(orig_dtype)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt',     required=True)
    p.add_argument('--res_hi_ckpt',  required=True)
    p.add_argument('--res_lo_ckpt',  required=True)
    p.add_argument('--data_root',    default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--redshifts',    nargs='+', type=int, default=[10])
    p.add_argument('--patch_size',   type=int, default=64)
    p.add_argument('--n_samples',    type=int, default=999)
    p.add_argument('--Lpix',         type=float, default=3.0)
    p.add_argument('--filter_k_cut', type=float, default=0.1,
                   help='lowpass k cutoff for res_lo output')
    return p.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # v3
    vae_low = VAE3D(in_ch=1, latent_ch=8, base_ch=64, ch_mults=(1, 2)).to(device)
    ck = torch.load(args.vae_ckpt, map_location=device)
    vae_low.load_state_dict(ck['model']); vae_low.eval()

    # res_hi (ch_mults=(1,))
    rh = torch.load(args.res_hi_ckpt, map_location=device)
    a_hi = rh.get('args', {})
    res_hi = VAE3D(in_ch=1, latent_ch=a_hi.get('res_latent_ch', 4),
                   base_ch=a_hi.get('res_base_ch', 32), ch_mults=(1,)).to(device)
    res_hi.load_state_dict(rh['model']); res_hi.eval()

    # res_lo (ch_mults=(1,2,2))
    rl = torch.load(args.res_lo_ckpt, map_location=device)
    a_lo = rl.get('args', {})
    res_lo = VAE3D(in_ch=1, latent_ch=a_lo.get('res_lo_latent_ch', 8),
                   base_ch=a_lo.get('res_lo_base_ch', 32), ch_mults=(1, 2, 2)).to(device)
    res_lo.load_state_dict(rl['model']); res_lo.eval()

    print(f"v3: ep{ck.get('epoch','?')} | res_hi: ep{rh.get('epoch','?')} | res_lo: ep{rl.get('epoch','?')}")

    ds = T21Dataset(args.data_root, args.patch_size,
                    redshifts=args.redshifts, split='val')
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    O, L, Lh, Lhl = [], [], [], []
    for batch in loader:
        x = batch['patch'].to(device)
        x_low, _, _    = vae_low(x)
        r_hi_hat, _, _ = res_hi(x - x_low)
        r_lo_hat, _, _ = res_lo(x - x_low - r_hi_hat)
        r_lo_hat = lowpass_3d(r_lo_hat, args.filter_k_cut)
        O.append(x.cpu())
        L.append(x_low.cpu())
        Lh.append((x_low + r_hi_hat).cpu())
        Lhl.append((x_low + r_hi_hat + r_lo_hat).cpu())
        if sum(o.shape[0] for o in O) >= args.n_samples:
            break

    X   = torch.cat(O)[:args.n_samples].squeeze(1).numpy()
    V3  = torch.cat(L)[:args.n_samples].squeeze(1).numpy()
    V3h = torch.cat(Lh)[:args.n_samples].squeeze(1).numpy()
    Fn  = torch.cat(Lhl)[:args.n_samples].squeeze(1).numpy()

    def metrics(recon, tag):
        mse = float(((recon - X)**2).mean())
        rel_var = mse / float(X.var())
        k, ps_o = power_spectrum(X,     Lpix=args.Lpix)
        _, ps_r = power_spectrum(recon, Lpix=args.Lpix)
        ratio = ps_r.mean(0) / np.maximum(ps_o.mean(0), 1e-30)
        m_lo  = k < 0.1
        m_hi  = k >= 0.5
        m_mid = (~m_lo) & (~m_hi)
        ps_lo  = float(ratio[m_lo].mean())  if m_lo.any()  else float('nan')
        ps_mid = float(ratio[m_mid].mean()) if m_mid.any() else float('nan')
        ps_hi  = float(ratio[m_hi].mean())  if m_hi.any()  else float('nan')
        print(f"\n[{tag}]")
        print(f"  MSE:        {mse:.6f}   (MSE/var = {rel_var:.4f})")
        print(f"  PS large:   {ps_lo:.3f}")
        print(f"  PS mid:     {ps_mid:.3f}")
        print(f"  PS small:   {ps_hi:.3f}")

    metrics(V3,  'v3 baseline')
    metrics(V3h, 'v3 + res_hi')
    metrics(Fn,  'v3 + res_hi + res_lo  (MoE final)')


if __name__ == '__main__':
    main()
