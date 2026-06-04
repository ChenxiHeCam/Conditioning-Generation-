# Fast GPU bispectrum on the 100 saved (true, gen) cubes.
# Equilateral B(k,k,k) via bandpass triple product, all on CUDA.
import os, sys, argparse, glob, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def equilateral_bispectrum_gpu(x, Lpix=3.0, kbins=8, device='cuda'):
    """x: (N,N,N) np.float32. Returns (k_centers, B(k))."""
    N = x.shape[-1]
    x_t = torch.from_numpy(x).to(device).float()
    x_t = x_t - x_t.mean()
    F = torch.fft.rfftn(x_t)
    k1  = torch.fft.fftfreq(N,  device=device) * N
    k1r = torch.fft.rfftfreq(N, device=device) * N
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = (KX**2 + KY**2 + KZ**2).sqrt()
    kbox = 2 * np.pi / (N * Lpix)
    K_phys = Kmag * kbox
    k_nyq = np.pi / Lpix
    edges = torch.logspace(np.log10(0.05), np.log10(k_nyq * 0.95),
                           kbins + 1, device=device)
    centers = (edges[:-1] * edges[1:]).sqrt()
    B = torch.zeros(kbins, device=device)
    for i in range(kbins):
        m = (K_phys >= edges[i]) & (K_phys < edges[i+1])
        if not m.any():
            continue
        F_masked = torch.zeros_like(F)
        F_masked[m] = F[m]
        delta_k = torch.fft.irfftn(F_masked, s=x_t.shape)
        B[i] = (delta_k ** 3).mean()
    return centers.cpu().numpy(), B.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',  required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--n_cubes', type=int, default=100)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.in_dir, '*.npz')))[:args.n_cubes]
    print(f"computing GPU bispectrum on {len(files)} cubes")

    bt_list, bg_list = [], []
    import time as tm
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        t0 = tm.time()
        kb, bt = equilateral_bispectrum_gpu(d['true_norm'])
        _,  bg = equilateral_bispectrum_gpu(d['gen_norm'])
        bt_list.append(bt); bg_list.append(bg)
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}]  per-cube {tm.time()-t0:.1f}s")

    bt = np.stack(bt_list); bg = np.stack(bg_list)
    np.savez(os.path.join(args.out_dir, 'bispec.npz'),
             kb=kb, bt=bt, bg=bg)

    bt_m, bt_s = np.nanmean(bt, 0), np.nanstd(bt, 0)
    bg_m, bg_s = np.nanmean(bg, 0), np.nanstd(bg, 0)
    rat = bg_m / np.where(np.abs(bt_m) > 1e-30, bt_m, np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    ax1.errorbar(kb, np.abs(bt_m), yerr=bt_s, fmt='ko-', lw=2,
                 label=f'target ({(bt_m<0).sum()} bins negative)')
    ax1.errorbar(kb, np.abs(bg_m), yerr=bg_s, fmt='C0o-', lw=2,
                 label=f'v5 + cond residual ({(bg_m<0).sum()} negative)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_ylabel(r'$|B(k,k,k)|$  (z-score units)')
    ax1.legend()
    ax1.set_title(f'Equilateral bispectrum, n={len(files)} test cubes')

    ax2.axhline(1.0, color='k', ls=':')
    ax2.axhspan(0.5, 1.5, color='g', alpha=0.10, label=r'$\pm 50\%$')
    ax2.semilogx(kb, rat, 'C0o-', lw=2)
    ax2.set_xlabel(r'$k$  [cMpc$^{-1}$]')
    ax2.set_ylabel(r'$B_{\rm gen}/B_{\rm true}$')
    ax2.set_ylim(-2, 4)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'I_LDM_bispectrum.png'),
                dpi=140, bbox_inches='tight')

    with open(os.path.join(args.out_dir, 'bispec_summary.json'), 'w') as f:
        json.dump(dict(k=kb.tolist(), bt_mean=bt_m.tolist(),
                       bg_mean=bg_m.tolist(), ratio=rat.tolist()), f, indent=2)
    print(f"done -> {args.out_dir}/I_LDM_bispectrum.png")


if __name__ == '__main__':
    main()
