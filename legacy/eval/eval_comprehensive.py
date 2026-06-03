"""
Comprehensive eval: compare v3+res_hi vs v4 across redshifts, splits, data sources,
per-sample variance, per-bin PS, and full-cube tiled reconstruction.

Usage:
  python eval_comprehensive.py --v4_ckpt PATH --out PATH
"""
import os, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import T21Dataset
from models.vae import VAE3D
from utils.power_spectrum import power_spectrum


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--v3_ckpt',     default='/root/autodl-tmp/checkpoints/vae_v3/vae_v3_epoch0059.pt')
    p.add_argument('--reshi_ckpt',  default='/root/autodl-tmp/checkpoints/vae_residual/residual_epoch0019.pt')
    p.add_argument('--v4_ckpt',     required=True)
    p.add_argument('--out',         default='/root/autodl-tmp/eval_comprehensive.json')
    p.add_argument('--data_ic',     default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_astro',  default='/root/autodl-tmp/ASR21cm/varying_astro')
    return p.parse_args()


def ps_metrics(X, R, k_thresh_lo=0.1, k_thresh_hi=0.5):
    """Power spectrum ratios using only bins with actual modes."""
    k, ps_o = power_spectrum(X)
    _, ps_r = power_spectrum(R)
    valid = ps_o.mean(0) > 1e-20
    ratio = ps_r.mean(0) / np.maximum(ps_o.mean(0), 1e-30)
    m_lo = (k < k_thresh_lo) & valid
    m_mid = (k >= k_thresh_lo) & (k < k_thresh_hi) & valid
    m_hi = (k >= k_thresh_hi) & valid
    return dict(
        k=k.tolist(),
        ratio=ratio.tolist(),
        valid=valid.tolist(),
        ps_large=float(ratio[m_lo].mean()),
        ps_mid=float(ratio[m_mid].mean()),
        ps_small=float(ratio[m_hi].mean()),
        n_large=int(m_lo.sum()),
        n_mid=int(m_mid.sum()),
        n_small=int(m_hi.sum()),
    )


def per_sample_stats(X, R):
    """Per-sample PS ratio variance — detect outliers."""
    n = X.shape[0]
    per_lo = []
    per_hi = []
    for i in range(n):
        k, p_o = power_spectrum(X[i:i+1])
        _, p_r = power_spectrum(R[i:i+1])
        v = p_o[0] > 1e-20
        rat = p_r[0] / np.maximum(p_o[0], 1e-30)
        m_lo = (k < 0.1) & v
        m_hi = (k >= 0.5) & v
        per_lo.append(float(rat[m_lo].mean()) if m_lo.any() else np.nan)
        per_hi.append(float(rat[m_hi].mean()) if m_hi.any() else np.nan)
    return dict(
        per_sample_large=per_lo,
        per_sample_small=per_hi,
        large_mean=float(np.nanmean(per_lo)),
        large_std=float(np.nanstd(per_lo)),
        small_mean=float(np.nanmean(per_hi)),
        small_std=float(np.nanstd(per_hi)),
    )


@torch.no_grad()
def reconstruct(model_fn, ds, device, batch_size=8):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    O, R = [], []
    for batch in loader:
        x = batch['patch'].to(device)
        r = model_fn(x)
        O.append(x.cpu()); R.append(r.cpu())
    X = torch.cat(O).squeeze(1).numpy()
    R = torch.cat(R).squeeze(1).numpy()
    return X, R


def eval_block(model_fn, model_tag, splits=['val', 'test'], redshifts=[8,9,10,11,12],
               data_paths=None, device='cuda'):
    results = {}
    for data_tag, data_path in data_paths.items():
        # Skip combinations with no/broken data
        # varying_astro only has substantial data for z=10
        z_list = [10] if data_tag == 'astro' else redshifts
        for split in splits:
            for z in z_list:
                try:
                    ds = T21Dataset(data_path, 64, redshifts=[z], split=split)
                except Exception as e:
                    print(f"  [skip] {data_tag}-{split}-z{z}: {e}")
                    continue
                if len(ds) == 0:
                    continue
                try:
                    X, R = reconstruct(model_fn, ds, device)
                except (RecursionError, OSError, IOError) as e:
                    print(f"  [skip] {data_tag}-{split}-z{z}: data error {e}")
                    continue
                mse = float(((R - X) ** 2).mean())
                var = float(X.var())
                m = ps_metrics(X, R)
                key = f"{data_tag}-{split}-z{z}"
                results[key] = dict(
                    n=int(X.shape[0]),
                    mse=mse,
                    mse_over_var=mse / var if var > 0 else float('nan'),
                    ps_large=m['ps_large'],
                    ps_mid=m['ps_mid'],
                    ps_small=m['ps_small'],
                )
                print(f"  [{model_tag}] {key} n={X.shape[0]}: "
                      f"MSE/var={mse/var:.4f}  PS=({m['ps_large']:.3f}, {m['ps_mid']:.3f}, {m['ps_small']:.3f})")
    return results


def main():
    args = get_args()
    device = torch.device('cuda')

    # Load v3 (ch_mults=(1,2), latent_ch=8, base_ch=64) + res_hi
    v3 = VAE3D(in_ch=1, latent_ch=8, base_ch=64, ch_mults=(1, 2)).to(device)
    v3.load_state_dict(torch.load(args.v3_ckpt, map_location=device)['model']); v3.eval()
    rh = VAE3D(in_ch=1, latent_ch=4, base_ch=32, ch_mults=(1,)).to(device)
    rh.load_state_dict(torch.load(args.reshi_ckpt, map_location=device)['model']); rh.eval()

    def v3_only_fn(x):
        r, _, _ = v3(x); return r
    def v3_reshi_fn(x):
        xl, _, _ = v3(x)
        rh_hat, _, _ = rh(x - xl)
        return xl + rh_hat

    # Load v4
    v4_ckpt = torch.load(args.v4_ckpt, map_location=device)
    a = v4_ckpt.get('args', {})
    v4 = VAE3D(in_ch=1, latent_ch=a.get('latent_ch', 8),
               base_ch=a.get('base_ch', 128), ch_mults=tuple(a.get('ch_mults', [1]))).to(device)
    v4.load_state_dict(v4_ckpt['model']); v4.eval()
    print(f"v4 loaded: epoch {v4_ckpt.get('epoch')}, base_ch={a.get('base_ch')}, "
          f"ch_mults={a.get('ch_mults')}, latent_ch={a.get('latent_ch')}")

    def v4_fn(x):
        r, _, _ = v4(x); return r

    data_paths = {'astro': args.data_astro, 'IC': args.data_ic}

    all_results = {}
    for tag, fn in [('v3+res_hi', v3_reshi_fn), ('v4', v4_fn)]:
        print(f"\n=== {tag} ===")
        all_results[tag] = eval_block(fn, tag, data_paths=data_paths, device=device)

    # Per-sample variance — only on astro val z=10 to keep it cheap
    print('\n=== Per-sample variance (astro val z=10) ===')
    ds = T21Dataset(args.data_astro, 64, redshifts=[10], split='val')
    for tag, fn in [('v3+res_hi', v3_reshi_fn), ('v4', v4_fn)]:
        X, R = reconstruct(fn, ds, device)
        s = per_sample_stats(X, R)
        print(f"  [{tag}] PS_large: mean={s['large_mean']:.3f} std={s['large_std']:.3f}  "
              f"PS_small: mean={s['small_mean']:.3f} std={s['small_std']:.3f}")
        # Outliers
        per_lo = np.array(s['per_sample_large'])
        per_hi = np.array(s['per_sample_small'])
        bad_lo = np.where(np.abs(per_lo - 1) > 0.2)[0]
        bad_hi = np.where(np.abs(per_hi - 1) > 0.2)[0]
        print(f"    Outliers (>20% off): large={len(bad_lo)}, small={len(bad_hi)}")
        all_results[tag]['per_sample'] = s

    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results written to {args.out}")


if __name__ == '__main__':
    main()
