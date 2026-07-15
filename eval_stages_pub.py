# Publication-quality stage-decomposition figures from infer_stages.py outputs.
#
# Every figure shows the ablation chain:  VAE only | VAE+LDM | VAE+LDM+Res
# (plus extra residual variants if present in the npz files).
# All quantities in physical units: k [cMpc^-1], Delta^2 [mK^2], distance [cMpc].
#
# Outputs (--out_dir):
#   S1_ps_stages.png/pdf          dimensionless PS Delta^2(k), median + 16-84% band, ratio panel
#   S2_bispec_eq_stages.png/pdf   equilateral bispectrum B(k,k,k) + ratio panel
#   S3_bispec_sq_stages.png/pdf   squeezed bispectrum B(k,k,k_L->0) + ratio panel
#   S4_relmse_hist.png/pdf        per-cube rel_MSE histogram per stage
#   S5_slices.png/pdf             slice gallery: truth + each stage + diff, mK colorbar
#   S6_metric_hists.png/pdf       PS-band ratio distributions per stage
#   stage_metrics.json            per-cube numbers
#   STAGE_SUMMARY.md              median/IQR table
import os, argparse, glob, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LPIX = 3.0  # cMpc per voxel

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 100, 'savefig.dpi': 250, 'axes.grid': True,
    'grid.alpha': 0.25, 'lines.linewidth': 1.8,
})

STAGE_LABEL = {
    'vae_only':           'VAE only (recon. of truth)',
    'ldm_vae':            'VAE + LDM',
    'ldm_vae_res_v4logk': 'VAE + LDM + Res (v4logk)',
    'ldm_vae_res_v3lowk': 'VAE + LDM + Res (v3lowk)',
    'ldm_vae_res_v2':     'VAE + LDM + Res (v2)',
}
STAGE_COLOR = {
    'vae_only':           'tab:green',
    'ldm_vae':            'tab:orange',
    'ldm_vae_res_v4logk': 'tab:blue',
    'ldm_vae_res_v3lowk': 'tab:purple',
    'ldm_vae_res_v2':     'tab:brown',
}


def kgrid(N, device):
    k1  = torch.fft.fftfreq(N,  device=device) * N
    k1r = torch.fft.rfftfreq(N, device=device) * N
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = (KX**2 + KY**2 + KZ**2).sqrt()
    return Kmag * (2 * np.pi / (N * LPIX))     # physical k [cMpc^-1]


def ps_physical(x_mk, Kphys, edges, device):
    """Delta^2(k) [mK^2] on log-spaced bins. x_mk: (N,N,N) np array in mK."""
    N = x_mk.shape[-1]
    L = N * LPIX
    xt = torch.from_numpy(x_mk).to(device).float()
    xt = xt - xt.mean()
    F = torch.fft.rfftn(xt)
    P = (F.real**2 + F.imag**2) * (L**3 / N**6)   # P(k) [mK^2 cMpc^3]
    nb = len(edges) - 1
    d2 = torch.full((nb,), float('nan'), device=device)
    kc = torch.zeros(nb, device=device)
    for i in range(nb):
        m = (Kphys >= edges[i]) & (Kphys < edges[i+1])
        if m.any():
            km = Kphys[m].mean()
            d2[i] = P[m].mean() * km**3 / (2 * np.pi**2)
            kc[i] = km
    return kc.cpu().numpy(), d2.cpu().numpy()


def bandpass(F, Kphys, lo, hi, shape):
    Fm = torch.zeros_like(F)
    m = (Kphys >= lo) & (Kphys < hi)
    Fm[m] = F[m]
    return torch.fft.irfftn(Fm, s=shape)


def bispec(x_mk, Kphys, edges, device, squeezed_band=None):
    """Equilateral B(k,k,k) or, if squeezed_band=(lo,hi), squeezed
    B(k,k,k_L) with the long leg fixed to that band. Units mK^3."""
    xt = torch.from_numpy(x_mk).to(device).float()
    xt = xt - xt.mean()
    F = torch.fft.rfftn(xt)
    nb = len(edges) - 1
    B = torch.zeros(nb, device=device)
    kc = np.sqrt(edges[:-1] * edges[1:])
    dL = None
    if squeezed_band is not None:
        dL = bandpass(F, Kphys, squeezed_band[0], squeezed_band[1], xt.shape)
    for i in range(nb):
        dk = bandpass(F, Kphys, edges[i], edges[i+1], xt.shape)
        B[i] = (dk * dk * (dL if dL is not None else dk)).mean()
    return kc, B.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--n_cubes', type=int, default=100)
    p.add_argument('--primary_res', default='ldm_vae_res_v4logk')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    files = sorted(glob.glob(os.path.join(args.in_dir, '*.npz')))[:args.n_cubes]
    print(f"{len(files)} cubes from {args.in_dir} (device={dev})")
    d0 = np.load(files[0], allow_pickle=True)
    stages = [k for k in d0.files
              if k.startswith(('vae_only', 'ldm_vae'))]
    stages = sorted(stages, key=lambda s: (s != 'vae_only', s != 'ldm_vae', s))
    print("stages:", stages)

    N = d0['true_norm'].shape[-1]
    Kphys = kgrid(N, dev)
    k_nyq = np.pi / LPIX
    k_box = 2 * np.pi / (N * LPIX)
    ps_edges = np.logspace(np.log10(2 * k_box), np.log10(k_nyq * 0.95), 25)
    bs_edges = np.logspace(np.log10(0.05),      np.log10(k_nyq * 0.95), 9)
    sq_band = (k_box, 0.04)   # long leg of the squeezed configuration

    # ---------- pass over cubes ----------
    acc = {s: dict(ps=[], beq=[], bsq=[], rel=[], psl=[], psm=[], pss=[])
           for s in stages}
    ps_true, beq_true, bsq_true = [], [], []
    slice_example = None

    for fi, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        std, mean = float(d['t21_std']), float(d['t21_mean'])
        true_mk = d['true_norm'] * std + mean
        kc, d2t = ps_physical(true_mk, Kphys, ps_edges, dev)
        kbe, bte = bispec(true_mk, Kphys, bs_edges, dev)
        _,   bts = bispec(true_mk, Kphys, bs_edges, dev, squeezed_band=sq_band)
        ps_true.append(d2t); beq_true.append(bte); bsq_true.append(bts)

        for s in stages:
            gen_mk = d[s] * std + mean
            _, d2g = ps_physical(gen_mk, Kphys, ps_edges, dev)
            _, bge = bispec(gen_mk, Kphys, bs_edges, dev)
            _, bgs = bispec(gen_mk, Kphys, bs_edges, dev, squeezed_band=sq_band)
            acc[s]['ps'].append(d2g)
            acc[s]['beq'].append(bge)
            acc[s]['bsq'].append(bgs)
            t, g = d['true_norm'], d[s]
            acc[s]['rel'].append(float(((t-g)**2).sum() / (t**2).sum()))
            rat = d2g / np.where(d2t > 0, d2t, np.nan)
            acc[s]['psl'].append(np.nanmean(rat[(kc > 0.05) & (kc < 0.15)]))
            acc[s]['psm'].append(np.nanmean(rat[(kc >= 0.15) & (kc < 0.40)]))
            acc[s]['pss'].append(np.nanmean(rat[kc >= 0.40]))

        if fi == 0:
            slice_example = d
        if (fi + 1) % 10 == 0:
            print(f"  [{fi+1}/{len(files)}]", flush=True)

    ps_true = np.stack(ps_true); beq_true = np.stack(beq_true); bsq_true = np.stack(bsq_true)
    for s in stages:
        for key in acc[s]:
            acc[s][key] = np.stack([np.atleast_1d(v) for v in acc[s][key]]) \
                if key in ('ps', 'beq', 'bsq') else np.array(acc[s][key])

    def med_band(a):  # (n_cubes, n_bins) -> median, 16%, 84%
        return (np.nanmedian(a, 0), np.nanpercentile(a, 16, 0),
                np.nanpercentile(a, 84, 0))

    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(args.out_dir, f'{name}.{ext}'),
                        bbox_inches='tight')
        plt.close(fig)
        print(f"  -> {name}")

    # ---------- S1: power spectrum ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 8.0), sharex=True,
                                   gridspec_kw={'height_ratios': [2.2, 1]})
    mt, lo, hi = med_band(ps_true)
    ax1.plot(kc, mt, 'k-', lw=2.4, label='Simulation (truth)')
    ax1.fill_between(kc, lo, hi, color='k', alpha=0.12)
    for s in stages:
        mg, lg, hg = med_band(acc[s]['ps'])
        ax1.plot(kc, mg, color=STAGE_COLOR.get(s, 'gray'), ls='--',
                 label=STAGE_LABEL.get(s, s))
        ax1.fill_between(kc, lg, hg, color=STAGE_COLOR.get(s, 'gray'), alpha=0.10)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_ylabel(r'$\Delta^2_{21}(k)\ \ [\mathrm{mK}^2]$')
    ax1.set_title(f'21cm power spectrum, $z=10$, $n={len(files)}$ test cubes '
                  f'(median, 16–84% band)')
    ax1.legend(frameon=False)
    ax2.axhline(1, color='k', ls=':')
    ax2.axhspan(0.8, 1.2, color='g', alpha=0.08)
    for s in stages:
        r = acc[s]['ps'] / np.where(ps_true > 0, ps_true, np.nan)
        mr, lr, hr = med_band(r)
        ax2.plot(kc, mr, color=STAGE_COLOR.get(s, 'gray'))
        ax2.fill_between(kc, lr, hr, color=STAGE_COLOR.get(s, 'gray'), alpha=0.12)
    ax2.set_xscale('log'); ax2.set_ylim(0.4, 2.0)
    ax2.set_xlabel(r'$k\ \ [\mathrm{cMpc}^{-1}]$')
    ax2.set_ylabel(r'$\Delta^2_{\rm gen}/\Delta^2_{\rm true}$')
    save(fig, 'S1_ps_stages')

    # ---------- S2 / S3: bispectra ----------
    for name, bt_all, key, title in [
        ('S2_bispec_eq_stages', beq_true, 'beq',
         r'Equilateral bispectrum $B(k,k,k)$'),
        ('S3_bispec_sq_stages', bsq_true, 'bsq',
         r'Squeezed bispectrum $B(k,k,k_L),\ k_L\in[%.3f,%.2f]\,\mathrm{cMpc}^{-1}$'
         % (sq_band[0], sq_band[1]))]:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 8.0), sharex=True,
                                       gridspec_kw={'height_ratios': [2.2, 1]})
        mt, lo, hi = med_band(bt_all)
        ax1.plot(kbe, np.abs(mt), 'k-', lw=2.4, label='Simulation (truth)')
        ax1.fill_between(kbe, np.abs(lo), np.abs(hi), color='k', alpha=0.12)
        for s in stages:
            mg, lg, hg = med_band(acc[s][key])
            ax1.plot(kbe, np.abs(mg), color=STAGE_COLOR.get(s, 'gray'), ls='--',
                     label=STAGE_LABEL.get(s, s))
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_ylabel(r'$|B|\ \ [\mathrm{mK}^3]$')
        ax1.set_title(title + f',  $z=10$, $n={len(files)}$')
        ax1.legend(frameon=False)
        ax2.axhline(1, color='k', ls=':')
        ax2.axhspan(0.5, 1.5, color='g', alpha=0.08, label=r'$\pm50\%$')
        for s in stages:
            r = acc[s][key] / np.where(np.abs(bt_all) > 1e-30, bt_all, np.nan)
            mr, lr, hr = med_band(r)
            ax2.plot(kbe, mr, color=STAGE_COLOR.get(s, 'gray'))
            ax2.fill_between(kbe, lr, hr, color=STAGE_COLOR.get(s, 'gray'), alpha=0.12)
        ax2.set_xscale('log'); ax2.set_ylim(-1, 4)
        ax2.set_xlabel(r'$k\ \ [\mathrm{cMpc}^{-1}]$')
        ax2.set_ylabel(r'$B_{\rm gen}/B_{\rm true}$')
        ax2.legend(frameon=False)
        save(fig, name)

    # ---------- S4: rel_MSE histograms ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.logspace(np.log10(5e-3), np.log10(1.0), 36)
    for s in stages:
        ax.hist(acc[s]['rel'], bins=bins, histtype='step', lw=2,
                color=STAGE_COLOR.get(s, 'gray'),
                label=f"{STAGE_LABEL.get(s, s)}  (med={np.median(acc[s]['rel']):.3f})")
    ax.set_xscale('log')
    ax.set_xlabel('relative MSE per cube')
    ax.set_ylabel('number of test cubes')
    ax.set_title(f'Per-cube relative MSE, $z=10$, $n={len(files)}$')
    ax.legend(frameon=False)
    save(fig, 'S4_relmse_hist')

    # ---------- S5: slice gallery ----------
    d = slice_example
    std, mean = float(d['t21_std']), float(d['t21_mean'])
    zc = N // 2
    panels = [('true_norm', 'Simulation (truth)')] + \
             [(s, STAGE_LABEL.get(s, s)) for s in stages]
    ncol = len(panels)
    extent = [0, N * LPIX, 0, N * LPIX]
    true_mk = d['true_norm'][:, :, zc] * std + mean
    vmin, vmax = np.percentile(true_mk, [1, 99])
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.4),
                             constrained_layout=True)
    for ci, (key, label) in enumerate(panels):
        img = d[key][:, :, zc] * std + mean
        im = axes[0, ci].imshow(img.T, origin='lower', cmap='viridis',
                                vmin=vmin, vmax=vmax, extent=extent)
        axes[0, ci].set_title(label, fontsize=10)
        axes[0, ci].set_xlabel('x [cMpc]')
        if ci == 0:
            axes[0, ci].set_ylabel('y [cMpc]')
        if key == 'true_norm':
            axes[1, ci].axis('off')
        else:
            diff = img - true_mk
            im2 = axes[1, ci].imshow(diff.T, origin='lower', cmap='RdBu_r',
                                     vmin=-0.5 * (vmax - vmin),
                                     vmax= 0.5 * (vmax - vmin), extent=extent)
            axes[1, ci].set_title('residual vs truth', fontsize=9)
            axes[1, ci].set_xlabel('x [cMpc]')
            if ci == 1:
                axes[1, ci].set_ylabel('y [cMpc]')
    cb = fig.colorbar(im, ax=axes[0, :], shrink=0.85, pad=0.01)
    cb.set_label(r'$\delta T_b$ [mK]')
    cb2 = fig.colorbar(im2, ax=axes[1, :], shrink=0.85, pad=0.01)
    cb2.set_label(r'$\Delta\,\delta T_b$ [mK]')
    fig.suptitle(f"central slice $z$-index {zc} ({str(d['fname'])}), $z=10$")
    save(fig, 'S5_slices')

    # ---------- S6: PS-band ratio distributions ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, key, title in zip(
            axes, ['psl', 'psm', 'pss'],
            [r'large scales $0.05<k<0.15$', r'mid $0.15<k<0.40$',
             r'small $k>0.40\ \mathrm{cMpc}^{-1}$']):
        bins = np.logspace(np.log10(0.3), np.log10(30), 30)
        for s in stages:
            ax.hist(acc[s][key], bins=bins, histtype='step', lw=2,
                    color=STAGE_COLOR.get(s, 'gray'),
                    label=STAGE_LABEL.get(s, s))
        ax.axvline(1, color='k', ls=':')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\Delta^2_{\rm gen}/\Delta^2_{\rm true}$')
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel('number of test cubes')
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, 'S6_metric_hists')

    # ---------- summary ----------
    summ = {}
    for s in stages:
        summ[s] = {k: dict(median=float(np.nanmedian(acc[s][k])),
                           mean=float(np.nanmean(acc[s][k])),
                           iqr=[float(np.nanpercentile(acc[s][k], 25)),
                                float(np.nanpercentile(acc[s][k], 75))])
                   for k in ['rel', 'psl', 'psm', 'pss']}
    with open(os.path.join(args.out_dir, 'stage_metrics.json'), 'w') as f:
        json.dump(dict(n_files=len(files), stages=stages, summary=summ), f, indent=2)

    with open(os.path.join(args.out_dir, 'STAGE_SUMMARY.md'), 'w') as f:
        f.write(f'# Stage decomposition, n={len(files)} test cubes, z=10\n\n')
        f.write('| stage | rel_MSE | PS large | PS mid | PS small |\n|---|---|---|---|---|\n')
        for s in stages:
            m = summ[s]
            f.write(f"| {STAGE_LABEL.get(s, s)} | "
                    + " | ".join(f"{m[k]['median']:.3f} [{m[k]['iqr'][0]:.2f},{m[k]['iqr'][1]:.2f}]"
                                 for k in ['rel', 'psl', 'psm', 'pss']) + " |\n")
    print("done ->", args.out_dir)


if __name__ == '__main__':
    main()
