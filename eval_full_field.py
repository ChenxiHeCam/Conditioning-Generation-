# Aggregate metrics + figures from generated full-field outputs (npz)
# Produces:
#   D_LDM_ps_curves.png        PS, target vs gen, mean ± std band + ratio panel
#   E_LDM_best_worst.png       Best/worst cube slice comparison (3 z-slices each)
#   F_LDM_astro_corner.png     Astro-param corner coloured by per-cube rel_MSE
#   G_LDM_metric_hists.png     Distributions of all metrics
#   H_LDM_slices.png           Multi-slice true vs gen vs diff for one example
#   I_LDM_bispectrum.png       Equilateral bispectrum |B(k,k,k)|
#   RESULTS_full.md            Text summary (median + IQR + outlier counts)
import os, sys, argparse, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scs


def spherical_ps(x, Lpix=3.0, kbins=30):
    N = x.shape[-1]
    xm = x - x.mean()
    F = np.fft.rfftn(xm)
    P = (F.real**2 + F.imag**2) / N**3
    k1  = np.fft.fftfreq(N)  * N
    k1r = np.fft.rfftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kbox = 2 * np.pi / (N * Lpix)
    K_phys = Kmag * kbox
    edges = np.linspace(0, K_phys.max() + 1e-6, kbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ps = np.zeros(kbins)
    for i in range(kbins):
        m = (K_phys >= edges[i]) & (K_phys < edges[i+1])
        if m.sum() > 0:
            ps[i] = P[m].mean()
    return centers, ps


def equilateral_bispectrum(x, Lpix=3.0, kbins=8):
    """Equilateral B(k,k,k). Bandpass-filter at each k, multiply 3 copies,
    take the real-space mean. Cheap because the three filters are identical."""
    N = x.shape[-1]
    xm = x - x.mean()
    F = np.fft.rfftn(xm)
    k1  = np.fft.fftfreq(N)  * N
    k1r = np.fft.rfftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kbox = 2 * np.pi / (N * Lpix)
    K_phys = Kmag * kbox
    # log-spaced bins from 0.05 to k_nyq
    k_nyq = np.pi / Lpix
    edges = np.logspace(np.log10(0.05), np.log10(k_nyq * 0.95), kbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    B = np.zeros(kbins)
    for i in range(kbins):
        m = (K_phys >= edges[i]) & (K_phys < edges[i+1])
        if m.sum() == 0:
            continue
        F_masked = np.zeros_like(F)
        F_masked[m] = F[m]
        delta_k = np.fft.irfftn(F_masked, s=x.shape)
        B[i] = (delta_k ** 3).mean()
    return centers, B


def load_all(npz_dir):
    files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
    cubes = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        cubes.append({
            'true': d['true_norm'],
            'gen':  d['gen_norm'],
            'params': d['params'],
            't21_mean': float(d['t21_mean']),
            't21_std':  float(d['t21_std']),
            'fname': str(d['fname']),
        })
    return cubes


def per_cube_metrics(c, compute_bispec=True):
    t = c['true']; g = c['gen']
    rel = float(((t-g)**2).sum() / (t**2).sum())
    mse = float(((t-g)**2).mean())
    k, pt = spherical_ps(t)
    _, pg = spherical_ps(g)
    rat = pg / np.clip(pt, 1e-30, None)
    valid = pt > 1e-20
    L = float(rat[(k<0.1)&valid].mean()) if ((k<0.1)&valid).any() else float('nan')
    M = float(rat[(k>=0.1)&(k<0.5)&valid].mean())
    S = float(rat[(k>=0.5)&valid].mean())
    pcorr = float(np.corrcoef(t.flatten(), g.flatten())[0,1])
    tp = t * c['t21_std'] + c['t21_mean']
    gp = g * c['t21_std'] + c['t21_mean']
    kt = float(scs.kurtosis(tp.flatten()))
    kg = float(scs.kurtosis(gp.flatten()))
    ks = float(scs.ks_2samp(tp.flatten()[::100], gp.flatten()[::100]).statistic)
    out = dict(rel_mse=rel, mse=mse, ps_large=L, ps_mid=M, ps_small=S,
               pix_corr=pcorr, kurt_true=kt, kurt_gen=kg, ks=ks,
               k=k.tolist(), ps_true=pt.tolist(), ps_gen=pg.tolist())
    if compute_bispec:
        kb, bt = equilateral_bispectrum(t)
        _,  bg = equilateral_bispectrum(g)
        out['kb'] = kb.tolist()
        out['bispec_true'] = bt.tolist()
        out['bispec_gen']  = bg.tolist()
    return out


def fig_ps_curves(cubes, metrics, out_path):
    k_arr = np.array(metrics[0]['k'])
    pt_arr = np.stack([m['ps_true'] for m in metrics])
    pg_arr = np.stack([m['ps_gen']  for m in metrics])
    pt_mean, pt_std = pt_arr.mean(0), pt_arr.std(0)
    pg_mean, pg_std = pg_arr.mean(0), pg_arr.std(0)
    ratio = pg_arr / np.clip(pt_arr, 1e-30, None)
    # clip per-cube outliers before averaging the ratio
    ratio = np.where((ratio > 0.1) & (ratio < 10), ratio, np.nan)
    rat_mean = np.nanmean(ratio, 0)
    rat_std  = np.nanstd(ratio, 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    pos = pt_mean > 0
    ax1.loglog(k_arr[pos], pt_mean[pos], 'k-', lw=2, label='target')
    ax1.fill_between(k_arr[pos], pt_mean[pos]-pt_std[pos], pt_mean[pos]+pt_std[pos],
                     color='k', alpha=0.15)
    ax1.loglog(k_arr[pos], pg_mean[pos], 'C0-', lw=2, label='v5 + cond residual')
    ax1.fill_between(k_arr[pos], pg_mean[pos]-pg_std[pos], pg_mean[pos]+pg_std[pos],
                     color='C0', alpha=0.25)
    ax1.set_ylabel(r'$P(k)$  (z-score units)')
    ax1.legend(loc='lower left')
    ax1.set_title(f'Full 256$^3$ field, n={len(metrics)} test cubes (z=10, astro)')

    ax2.semilogx(k_arr[pos], rat_mean[pos], 'C0-', lw=2)
    ax2.fill_between(k_arr[pos], rat_mean[pos]-rat_std[pos], rat_mean[pos]+rat_std[pos],
                     color='C0', alpha=0.25)
    ax2.axhline(1.0, color='k', ls=':')
    ax2.axhspan(0.95, 1.05, color='g', alpha=0.10, label=r'$\pm 5\%$')
    ax2.set_xlabel(r'$k$  [cMpc$^{-1}$]')
    ax2.set_ylabel(r'$P_{\rm gen}/P_{\rm true}$')
    ax2.set_ylim(0.6, 1.6)
    ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def fig_bispectrum(cubes, metrics, out_path):
    if 'bispec_true' not in metrics[0]:
        return
    kb_arr = np.array(metrics[0]['kb'])
    bt_arr = np.stack([m['bispec_true'] for m in metrics])
    bg_arr = np.stack([m['bispec_gen']  for m in metrics])
    bt_m = np.nanmean(bt_arr, 0); bt_s = np.nanstd(bt_arr, 0)
    bg_m = np.nanmean(bg_arr, 0); bg_s = np.nanstd(bg_arr, 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    sgn_t = np.sign(bt_m)
    sgn_g = np.sign(bg_m)
    ax1.errorbar(kb_arr, np.abs(bt_m), yerr=bt_s, fmt='ko-', lw=2,
                 label=f'target ({(sgn_t<0).sum()} bins < 0)')
    ax1.errorbar(kb_arr, np.abs(bg_m), yerr=bg_s, fmt='C0o-', lw=2,
                 label=f'v5 + cond res ({(sgn_g<0).sum()} bins < 0)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_ylabel(r'$|B(k,k,k)|$  (z-score units)')
    ax1.legend()
    ax1.set_title(f'Equilateral bispectrum, n={len(metrics)} test cubes')

    rat = bg_m / np.where(np.abs(bt_m) > 1e-30, bt_m, np.nan)
    ax2.axhline(1.0, color='k', ls=':')
    ax2.axhspan(0.5, 1.5, color='g', alpha=0.10, label=r'$\pm 50\%$')
    ax2.semilogx(kb_arr, rat, 'C0o-', lw=2)
    ax2.set_xlabel(r'$k$  [cMpc$^{-1}$]')
    ax2.set_ylabel(r'$B_{\rm gen}/B_{\rm true}$')
    ax2.set_ylim(-2, 4)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def _clip(img, p=1):
    """Per-image percentile clip for display."""
    lo, hi = np.percentile(img, [p, 100 - p])
    return lo, hi


def fig_best_worst(cubes, metrics, out_path, n_rows=3):
    idx_sorted = np.argsort([m['rel_mse'] for m in metrics])
    best  = idx_sorted[:n_rows].tolist()
    worst = idx_sorted[-n_rows:].tolist()

    fig, axes = plt.subplots(n_rows * 2, 3, figsize=(11, 3.4 * n_rows * 2))
    mid = cubes[0]['true'].shape[-1] // 2

    def show(ax_row, c, label, m):
        tp = c['true'] * c['t21_std'] + c['t21_mean']
        gp = c['gen']  * c['t21_std'] + c['t21_mean']
        vmin, vmax = _clip(tp[mid], p=1)
        ax_row[0].imshow(tp[mid], cmap='magma', vmin=vmin, vmax=vmax)
        ax_row[0].set_title(f'{label}  true (mK)  z-slice {mid}')
        ax_row[1].imshow(gp[mid], cmap='magma', vmin=vmin, vmax=vmax)
        ax_row[1].set_title(f'gen, rel_MSE={m["rel_mse"]:.3f}')
        diff = gp[mid] - tp[mid]
        dmax = max(abs(np.percentile(diff, 1)), abs(np.percentile(diff, 99)))
        ax_row[2].imshow(diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax)
        ax_row[2].set_title(f'(gen − true), PS_S={m["ps_small"]:.2f}')
        for a in ax_row: a.axis('off')

    for r, i in enumerate(best):
        show(axes[r], cubes[i], f'BEST #{r+1}', metrics[i])
    for r, i in enumerate(worst):
        show(axes[n_rows + r], cubes[i], f'WORST #{r+1}', metrics[i])
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def fig_slices(cubes, metrics, out_path):
    """For the cube with median rel_MSE, show 4 z-slices: true vs gen vs diff."""
    rel = np.array([m['rel_mse'] for m in metrics])
    med_idx = int(np.argmin(np.abs(rel - np.median(rel))))
    c = cubes[med_idx]; m = metrics[med_idx]
    tp = c['true'] * c['t21_std'] + c['t21_mean']
    gp = c['gen']  * c['t21_std'] + c['t21_mean']
    N = tp.shape[-1]
    slices = [N//8, N//4, N//2, 3*N//4]

    fig, axes = plt.subplots(3, len(slices), figsize=(3 * len(slices), 9))
    for col, sl in enumerate(slices):
        vmin, vmax = _clip(tp[sl], p=1)
        axes[0, col].imshow(tp[sl], cmap='magma', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'true   z-slice {sl}')
        axes[1, col].imshow(gp[sl], cmap='magma', vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f'gen')
        diff = gp[sl] - tp[sl]
        d = max(abs(np.percentile(diff, 1)), abs(np.percentile(diff, 99)))
        axes[2, col].imshow(diff, cmap='RdBu_r', vmin=-d, vmax=d)
        axes[2, col].set_title(f'gen − true')
        for ax in axes[:, col]: ax.axis('off')
    fig.suptitle(f'Median-quality test cube (rel_MSE={m["rel_mse"]:.3f}),  '
                 f'PS={m["ps_large"]:.2f}/{m["ps_mid"]:.2f}/{m["ps_small"]:.2f}',
                 y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def fig_astro_corner(cubes, metrics, out_path):
    pnames = ['MyStar_II', 'MyVc', 'MyFX', 'DelayParam']
    P = np.stack([c['params'][:4] for c in cubes])
    rel = np.array([m['rel_mse'] for m in metrics])

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    sc = None
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i == j:
                ax.hist(P[:, i], bins=15, color='C0', alpha=0.7)
                ax.set_xlabel(pnames[i])
                if i == 0:
                    ax.set_ylabel('count')
            elif j < i:
                sc = ax.scatter(P[:, j], P[:, i], c=rel, cmap='viridis',
                                s=30, vmin=rel.min(),
                                vmax=np.percentile(rel, 95))
                if i == 3:
                    ax.set_xlabel(pnames[j])
                if j == 0:
                    ax.set_ylabel(pnames[i])
            else:
                ax.axis('off')
    fig.suptitle(f'Astro corner coloured by rel_MSE  (n={len(cubes)})', y=0.995)
    if sc is not None:
        cax = fig.add_axes([0.78, 0.5, 0.02, 0.35])
        plt.colorbar(sc, cax=cax).set_label('rel_MSE')
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def fig_summary_card(metrics, out_path):
    rel = np.array([m['rel_mse'] for m in metrics])
    L = np.array([m['ps_large'] for m in metrics])
    M = np.array([m['ps_mid']   for m in metrics])
    S = np.array([m['ps_small'] for m in metrics])
    pc = np.array([m['pix_corr'] for m in metrics])
    ks = np.array([m['ks']       for m in metrics])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, vals, name in zip(axes.flatten(),
        [rel, L, M, S, pc, ks],
        ['rel_MSE', 'PS large (k<0.1)', 'PS mid (0.1<=k<0.5)',
         'PS small (k>=0.5)', 'pixel correlation', 'PDF KS distance']):
        # auto-clip to 1st-99th percentile for legibility
        vlo, vhi = np.percentile(vals, [1, 99])
        ax.hist(vals.clip(vlo, vhi), bins=15, color='C0', alpha=0.75)
        ax.axvline(np.median(vals), color='k', ls='--',
                   label=f'median={np.median(vals):.3f}')
        ax.set_title(name)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',  required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--skip_bispec', action='store_true')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cubes = load_all(args.in_dir)
    print(f"loaded {len(cubes)} cubes")
    metrics = [per_cube_metrics(c, compute_bispec=not args.skip_bispec)
               for c in cubes]

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    fig_ps_curves   (cubes, metrics, os.path.join(args.out_dir, 'D_LDM_ps_curves.png'))
    fig_best_worst  (cubes, metrics, os.path.join(args.out_dir, 'E_LDM_best_worst.png'))
    fig_astro_corner(cubes, metrics, os.path.join(args.out_dir, 'F_LDM_astro_corner.png'))
    fig_summary_card(        metrics, os.path.join(args.out_dir, 'G_LDM_metric_hists.png'))
    fig_slices      (cubes, metrics, os.path.join(args.out_dir, 'H_LDM_slices.png'))
    if not args.skip_bispec:
        fig_bispectrum(cubes, metrics, os.path.join(args.out_dir, 'I_LDM_bispectrum.png'))

    rel = np.array([m['rel_mse']  for m in metrics])
    L   = np.array([m['ps_large'] for m in metrics])
    M   = np.array([m['ps_mid']   for m in metrics])
    S   = np.array([m['ps_small'] for m in metrics])
    pc  = np.array([m['pix_corr'] for m in metrics])
    ks  = np.array([m['ks']       for m in metrics])
    pct = lambda v: np.percentile(v, [25, 75])
    n_out = int(((L > 2) | (L < 0.5)).sum())
    with open(os.path.join(args.out_dir, 'RESULTS_full.md'), 'w') as f:
        f.write(f"# Full 256^3 field test-set results\n\n")
        f.write(f"Generated by tiling 64^3 patches with 50% overlap and Hann blending.\n")
        f.write(f"n = {len(cubes)} test cubes, z = 10, varying_astro suite.\n\n")
        f.write(f"PS-large outliers (>2× or <0.5×): {n_out}/{len(cubes)}\n\n")
        f.write(f"| metric | median | mean | IQR | min | max |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for name, v in [('rel_MSE', rel), ('PS large', L), ('PS mid', M),
                        ('PS small', S), ('pixel corr', pc), ('PDF KS', ks)]:
            lo, hi = pct(v)
            f.write(f"| {name} | {np.median(v):.4f} | {v.mean():.4f} | "
                    f"[{lo:.4f}, {hi:.4f}] | {v.min():.4f} | {v.max():.4f} |\n")

    print(f"figures and metrics written to {args.out_dir}")


if __name__ == '__main__':
    main()
