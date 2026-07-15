# Publication-quality MULTI-MODEL comparison from infer_baselines.py outputs.
#
# Usage:
#   python eval_models_pub.py --models \
#       "8x latent:/root/autodl-tmp/ff_latent8x" \
#       "pixel:/root/autodl-tmp/ff_pixel" \
#       "VQGAN-AR:/root/autodl-tmp/ff_ar" \
#       "StyleGAN:/root/autodl-tmp/ff_gan" \
#     --out_dir /root/autodl-tmp/model_compare_figs
#
# Outputs (png+pdf): M1 Delta^2(k) + ratio | M2 equilateral bispec |
# M3 squeezed bispec | M4 rel_MSE hist | M5 slice gallery |
# M6 speed-vs-quality trade-off | MODEL_SUMMARY.md + model_metrics.json
import os, argparse, glob, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eval_stages_pub import kgrid, ps_physical, bispec, LPIX

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 10,
    'figure.dpi': 100, 'savefig.dpi': 250, 'axes.grid': True,
    'grid.alpha': 0.25, 'lines.linewidth': 1.8,
})
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True, help='label:npz_dir pairs')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--n_cubes', type=int, default=100)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = []
    for spec in args.models:
        label, d = spec.split(':', 1)
        files = sorted(glob.glob(os.path.join(d, '*.npz')))[:args.n_cubes]
        assert files, f'no npz in {d}'
        models.append((label, files))
    n = min(len(f) for _, f in models)
    print(f"{len(models)} models, {n} cubes each (device={dev})")

    d0 = np.load(models[0][1][0], allow_pickle=True)
    N = d0['true_norm'].shape[-1]
    Kphys = kgrid(N, dev)
    k_nyq = np.pi / LPIX
    k_box = 2 * np.pi / (N * LPIX)
    ps_edges = np.logspace(np.log10(2 * k_box), np.log10(k_nyq * 0.95), 25)
    bs_edges = np.logspace(np.log10(0.05),      np.log10(k_nyq * 0.95), 9)
    sq_band = (k_box, 0.04)

    # truth pass (from the first model's files — truths identical across dirs)
    ps_t, beq_t, bsq_t = [], [], []
    for f in models[0][1][:n]:
        d = np.load(f, allow_pickle=True)
        mk = d['true_norm'] * float(d['t21_std']) + float(d['t21_mean'])
        kc, d2 = ps_physical(mk, Kphys, ps_edges, dev)
        kbe, be = bispec(mk, Kphys, bs_edges, dev)
        _,   bs = bispec(mk, Kphys, bs_edges, dev, squeezed_band=sq_band)
        ps_t.append(d2); beq_t.append(be); bsq_t.append(bs)
    ps_t, beq_t, bsq_t = map(np.stack, (ps_t, beq_t, bsq_t))

    acc = {}
    for label, files in models:
        a = dict(ps=[], beq=[], bsq=[], rel=[], corr=[], secs=[])
        for f in files[:n]:
            d = np.load(f, allow_pickle=True)
            std, mean = float(d['t21_std']), float(d['t21_mean'])
            g = d['gen_norm']; t = d['true_norm']
            mk = g * std + mean
            _, d2 = ps_physical(mk, Kphys, ps_edges, dev)
            _, be = bispec(mk, Kphys, bs_edges, dev)
            _, bs = bispec(mk, Kphys, bs_edges, dev, squeezed_band=sq_band)
            a['ps'].append(d2); a['beq'].append(be); a['bsq'].append(bs)
            a['rel'].append(float(((t - g)**2).sum() / (t**2).sum()))
            a['corr'].append(float(np.corrcoef(t.ravel()[::50], g.ravel()[::50])[0, 1]))
            if 'gen_seconds' in d.files:
                a['secs'].append(float(d['gen_seconds']))
        acc[label] = {k: np.array(v) if k in ('rel', 'corr', 'secs') else np.stack(v)
                      for k, v in a.items()}
        print(f"  {label}: rel_MSE med {np.median(acc[label]['rel']):.4f}")

    def med_band(a):
        return (np.nanmedian(a, 0), np.nanpercentile(a, 16, 0),
                np.nanpercentile(a, 84, 0))

    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(args.out_dir, f'{name}.{ext}'),
                        bbox_inches='tight')
        plt.close(fig); print(f"  -> {name}")

    # ---- M1: PS ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 8.0), sharex=True,
                                   gridspec_kw={'height_ratios': [2.2, 1]})
    mt, lo, hi = med_band(ps_t)
    ax1.plot(kc, mt, 'k-', lw=2.4, label='Simulation (truth)')
    ax1.fill_between(kc, lo, hi, color='k', alpha=0.12)
    for c, (label, _) in zip(COLORS, models):
        mg, _, _ = med_band(acc[label]['ps'])
        ax1.plot(kc, mg, color=c, ls='--', label=label)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_ylabel(r'$\Delta^2_{21}(k)\ \ [\mathrm{mK}^2]$')
    ax1.set_title(f'21cm power spectrum — model comparison ($z=10$, $n={n}$)')
    ax1.legend(frameon=False)
    ax2.axhline(1, color='k', ls=':')
    ax2.axhspan(0.8, 1.2, color='g', alpha=0.08)
    for c, (label, _) in zip(COLORS, models):
        r = acc[label]['ps'] / np.where(ps_t > 0, ps_t, np.nan)
        mr, lr, hr = med_band(r)
        ax2.plot(kc, mr, color=c)
        ax2.fill_between(kc, lr, hr, color=c, alpha=0.12)
    ax2.set_xscale('log'); ax2.set_ylim(0.3, 2.5)
    ax2.set_xlabel(r'$k\ \ [\mathrm{cMpc}^{-1}]$')
    ax2.set_ylabel(r'$\Delta^2_{\rm gen}/\Delta^2_{\rm true}$')
    save(fig, 'M1_ps_models')

    # ---- M2/M3: bispectra ----
    for name, bt_all, key, title in [
        ('M2_bispec_eq_models', beq_t, 'beq', r'Equilateral bispectrum $B(k,k,k)$'),
        ('M3_bispec_sq_models', bsq_t, 'bsq',
         r'Squeezed bispectrum $B(k,k,k_L)$, $k_L\in[%.3f,%.2f]$' % sq_band)]:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 8.0), sharex=True,
                                       gridspec_kw={'height_ratios': [2.2, 1]})
        mt, lo, hi = med_band(bt_all)
        ax1.plot(kbe, np.abs(mt), 'k-', lw=2.4, label='Simulation (truth)')
        ax1.fill_between(kbe, np.abs(lo), np.abs(hi), color='k', alpha=0.12)
        for c, (label, _) in zip(COLORS, models):
            mg, _, _ = med_band(acc[label][key])
            ax1.plot(kbe, np.abs(mg), color=c, ls='--', label=label)
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_ylabel(r'$|B|\ \ [\mathrm{mK}^3]$')
        ax1.set_title(title + f'  ($z=10$, $n={n}$)')
        ax1.legend(frameon=False)
        ax2.axhline(1, color='k', ls=':')
        ax2.axhspan(0.5, 1.5, color='g', alpha=0.08)
        for c, (label, _) in zip(COLORS, models):
            r = acc[label][key] / np.where(np.abs(bt_all) > 1e-30, bt_all, np.nan)
            mr, lr, hr = med_band(r)
            ax2.plot(kbe, mr, color=c)
            ax2.fill_between(kbe, lr, hr, color=c, alpha=0.12)
        ax2.set_xscale('log'); ax2.set_ylim(-1, 4)
        ax2.set_xlabel(r'$k\ \ [\mathrm{cMpc}^{-1}]$')
        ax2.set_ylabel(r'$B_{\rm gen}/B_{\rm true}$')
        save(fig, name)

    # ---- M4: rel_MSE histogram ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.logspace(np.log10(5e-3), np.log10(2.0), 40)
    for c, (label, _) in zip(COLORS, models):
        ax.hist(acc[label]['rel'], bins=bins, histtype='step', lw=2, color=c,
                label=f"{label} (med {np.median(acc[label]['rel']):.3f})")
    ax.set_xscale('log')
    ax.set_xlabel('relative MSE per cube'); ax.set_ylabel('number of test cubes')
    ax.set_title(f'Per-cube relative MSE ($z=10$, $n={n}$)')
    ax.legend(frameon=False)
    save(fig, 'M4_relmse_hist')

    # ---- M5: slice gallery ----
    ncol = len(models) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.0 * ncol, 3.6),
                             constrained_layout=True)
    d = np.load(models[0][1][0], allow_pickle=True)
    std, mean = float(d['t21_std']), float(d['t21_mean'])
    zc = N // 2
    extent = [0, N * LPIX, 0, N * LPIX]
    img_t = d['true_norm'][:, :, zc] * std + mean
    vmin, vmax = np.percentile(img_t, [1, 99])
    im = axes[0].imshow(img_t.T, origin='lower', cmap='viridis',
                        vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title('Simulation (truth)', fontsize=10)
    axes[0].set_xlabel('x [cMpc]'); axes[0].set_ylabel('y [cMpc]')
    for ax, (label, files) in zip(axes[1:], models):
        dd = np.load(files[0], allow_pickle=True)
        img = dd['gen_norm'][:, :, zc] * std + mean
        ax.imshow(img.T, origin='lower', cmap='viridis',
                  vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(label, fontsize=10); ax.set_xlabel('x [cMpc]')
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label(r'$\delta T_b$ [mK]')
    save(fig, 'M5_slices_models')

    # ---- M6: speed vs quality ----
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for c, (label, _) in zip(COLORS, models):
        a = acc[label]
        if len(a['secs']):
            rat = a['ps'] / np.where(ps_t > 0, ps_t, np.nan)
            band = np.nanmedian(np.abs(np.nanmedian(rat, 0) - 1))
            ax.scatter(np.median(a['secs']), np.median(a['rel']),
                       s=90, color=c, label=label, zorder=3)
            ax.annotate(f"PS err {band*100:.0f}%",
                        (np.median(a['secs']), np.median(a['rel'])),
                        textcoords='offset points', xytext=(8, 6), fontsize=8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'inference wall-clock per $256^3$ cube [s]')
    ax.set_ylabel('median relative MSE')
    ax.set_title('Speed vs quality trade-off')
    ax.legend(frameon=False)
    save(fig, 'M6_speed_quality')

    # ---- summary ----
    summ = {}
    for label, _ in models:
        a = acc[label]
        rat = a['ps'] / np.where(ps_t > 0, ps_t, np.nan)
        m_rat = np.nanmedian(rat, 0)
        bands = dict(large=float(np.nanmean(m_rat[(kc > 0.05) & (kc < 0.15)])),
                     mid=float(np.nanmean(m_rat[(kc >= 0.15) & (kc < 0.40)])),
                     small=float(np.nanmean(m_rat[kc >= 0.40])))
        beq_r = np.nanmedian(a['beq'] / np.where(np.abs(beq_t) > 1e-30, beq_t, np.nan), 0)
        bsq_r = np.nanmedian(a['bsq'] / np.where(np.abs(bsq_t) > 1e-30, bsq_t, np.nan), 0)
        summ[label] = dict(
            rel_mse=float(np.median(a['rel'])),
            pix_corr=float(np.median(a['corr'])),
            ps_bands=bands,
            bispec_eq_mean_ratio=float(np.nanmean(beq_r)),
            bispec_sq_mean_ratio=float(np.nanmean(bsq_r)),
            sec_per_cube=float(np.median(a['secs'])) if len(a['secs']) else None)
    with open(os.path.join(args.out_dir, 'model_metrics.json'), 'w') as f:
        json.dump(dict(n_cubes=n, summary=summ), f, indent=2)
    with open(os.path.join(args.out_dir, 'MODEL_SUMMARY.md'), 'w') as f:
        f.write(f'# Model comparison, n={n} test cubes, z=10\n\n')
        f.write('| model | rel_MSE | pix corr | PS L/M/S | B_eq | B_sq | s/cube |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        for label, m in summ.items():
            b = m['ps_bands']
            f.write(f"| {label} | {m['rel_mse']:.4f} | {m['pix_corr']:.3f} | "
                    f"{b['large']:.2f}/{b['mid']:.2f}/{b['small']:.2f} | "
                    f"{m['bispec_eq_mean_ratio']:.2f} | {m['bispec_sq_mean_ratio']:.2f} | "
                    f"{m['sec_per_cube']} |\n")
    print('done ->', args.out_dir)


if __name__ == '__main__':
    main()
