"""Squeezed bispectrum B(k_L, k_S, k_S) at k_L≈0.05 cMpc^-1, for 3 sample cubes
selected at rel_MSE quantiles (2.5, 50, 97.5 percentile) per chain.

For each chain (2x/4x/8x/16x):
  - Compute rel_MSE per cube
  - Pick 3 cubes at 2.5%, 50%, 97.5% rel_MSE quantile
  - For each, compute squeezed bispec at k_L=0.05, varying k_S
  - Plot true vs gen, 3-panel per chain

Usage:
  python plot_squeezed_quantile.py --eval_root /path/to/eval_results --out plots/
"""
import os, argparse, glob, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LPIX = 3.0  # cMpc per pixel
N    = 256


def kgrid(N, device):
    k1  = torch.fft.fftfreq(N,  device=device) * N
    k1r = torch.fft.rfftfreq(N, device=device) * N
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    K = (KX**2 + KY**2 + KZ**2).sqrt()
    kbox = 2 * np.pi / (N * LPIX)
    return K * kbox


def bandpass(F, Kphys, lo, hi):
    """Return real-space field for modes in band [lo, hi]."""
    m = (Kphys >= lo) & (Kphys < hi)
    Fm = torch.zeros_like(F)
    Fm[m] = F[m]
    return torch.fft.irfftn(Fm, s=(N, N, N))


def squeezed_bispec(x_np, k_L=0.05, dk_L=0.01, kS_edges=None, device='cuda'):
    """Compute squeezed bispec B(k_L, k_S, k_S) = <delta_L * delta_S^2>.
    Returns (kS_centers, B_arr)."""
    x = torch.from_numpy(x_np).to(device).float()
    x = x - x.mean()
    Kphys = kgrid(N, device)
    F = torch.fft.rfftn(x)
    delta_L = bandpass(F, Kphys, k_L - dk_L/2, k_L + dk_L/2)
    if kS_edges is None:
        k_nyq = np.pi / LPIX
        kS_edges = np.logspace(np.log10(max(2*k_L, 0.1)), np.log10(k_nyq * 0.95), 9)
    kS_centers = np.sqrt(kS_edges[:-1] * kS_edges[1:])
    B = np.zeros(len(kS_centers))
    for i, (lo, hi) in enumerate(zip(kS_edges[:-1], kS_edges[1:])):
        delta_S = bandpass(F, Kphys, lo, hi)
        B[i] = float((delta_L * delta_S**2).mean())
    return kS_centers, B


def collect_chain(eval_root, tag):
    pattern = os.path.join(eval_root, f'ff_{tag}_test_z*', '*.npz')
    files = sorted(glob.glob(pattern))
    if not files: return None
    rel_mse = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        t = d['true_norm']; g = d['gen_norm']
        rel_mse.append(float(((t - g)**2).sum() / max((t**2).sum(), 1e-30)))
    return files, np.array(rel_mse)


def pick_quantile_idx(rel_mse, q):
    """Return index whose rel_MSE is at the q-th percentile."""
    target = np.percentile(rel_mse, q)
    return int(np.argmin(np.abs(rel_mse - target)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval_root', default='/home/ch2067/rds/hpc-work/21cm_gen/eval_results')
    p.add_argument('--out', default='plots/')
    p.add_argument('--tags', nargs='+', default=['2x', '4x', '8x', '16x'])
    p.add_argument('--k_L', type=float, default=0.05)
    p.add_argument('--dk_L', type=float, default=0.015)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    quantiles = [2.5, 50.0, 97.5]
    qlabels   = ['best (2.5%)', 'median (50%)', 'worst (97.5%)']

    results = {}
    for tag in args.tags:
        coll = collect_chain(args.eval_root, tag)
        if coll is None:
            print(f"WARN: no data for {tag}, skip"); continue
        files, rel_mse = coll
        results[tag] = dict(rel_mse=rel_mse.tolist(), curves=[])
        print(f"\n=== {tag}: {len(files)} cubes, rel_MSE quantiles ", end='')
        for q in quantiles:
            print(f"{q}:{np.percentile(rel_mse, q):.3f} ", end='')
        print()
        for q, qlab in zip(quantiles, qlabels):
            idx = pick_quantile_idx(rel_mse, q)
            d = np.load(files[idx], allow_pickle=True)
            kS, Bt = squeezed_bispec(d['true_norm'], k_L=args.k_L, dk_L=args.dk_L, device=device)
            kS, Bg = squeezed_bispec(d['gen_norm'],  k_L=args.k_L, dk_L=args.dk_L, device=device)
            results[tag]['curves'].append(dict(
                quantile=q, file=os.path.basename(files[idx]),
                rel_mse=float(rel_mse[idx]),
                kS=kS.tolist(), Bt=Bt.tolist(), Bg=Bg.tolist(),
            ))
            print(f"  {qlab}: {os.path.basename(files[idx])} rel_MSE={rel_mse[idx]:.3f}")

    # ---------- Plot ----------
    colors = dict(zip(args.tags, ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']))
    n_chains = len(results)
    fig, axes = plt.subplots(n_chains, 3, figsize=(15, 4 * n_chains), sharex=True)
    if n_chains == 1: axes = axes.reshape(1, -1)
    for r, (tag, dat) in enumerate(results.items()):
        for c, (curve, qlab) in enumerate(zip(dat['curves'], qlabels)):
            ax = axes[r, c]
            kS = np.array(curve['kS']); Bt = np.array(curve['Bt']); Bg = np.array(curve['Bg'])
            ax.plot(kS, np.abs(Bt), 'k.-', label='true')
            ax.plot(kS, np.abs(Bg), color=colors[tag], marker='s', label='gen')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_title(f'{tag}  |  {qlab}  |  rel_MSE={curve["rel_mse"]:.3f}', fontsize=10)
            if r == n_chains - 1: ax.set_xlabel(r'$k_S$ [cMpc$^{-1}$]')
            if c == 0: ax.set_ylabel(r'$|B(k_L, k_S, k_S)|$')
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.suptitle(f'Squeezed bispectrum at $k_L \\approx {args.k_L:.2f}$ cMpc$^{{-1}}$ per chain × rel_MSE quantile', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(args.out, 'squeezed_quantile.png'), dpi=150)
    plt.savefig(os.path.join(args.out, 'squeezed_quantile.pdf'))
    plt.close()
    print(f"\nsaved {args.out}/squeezed_quantile.png")

    with open(os.path.join(args.out, 'squeezed_quantile_data.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
