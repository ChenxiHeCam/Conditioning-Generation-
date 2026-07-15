# Recompute PS metrics from saved (true, gen) .npz cubes with a k_min cut.
# Removes the DC-mode artifact (k<0.05 cMpc^-1) that inflates PS_large.
import os, sys, glob, argparse, json
import numpy as np


def radial_ps(x, Lpix=3.0, n_bins=20, k_min=0.0):
    N = x.shape[-1]
    x = x - x.mean()
    F = np.fft.rfftn(x)
    P = (F.real**2 + F.imag**2) / N**3
    k1  = np.fft.fftfreq(N)  * N
    k1r = np.fft.rfftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kbox = 2*np.pi / (N*Lpix)
    Kphys = Kmag * kbox
    k_nyq = np.pi/Lpix
    edges = np.logspace(np.log10(max(k_min, 0.01)), np.log10(k_nyq*0.95), n_bins+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    P_b = np.zeros(n_bins)
    for i in range(n_bins):
        m = (Kphys >= edges[i]) & (Kphys < edges[i+1])
        if m.any():
            P_b[i] = P[m].mean()
        else:
            P_b[i] = np.nan
    return centers, P_b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', required=True)
    p.add_argument('--out',    required=True)
    p.add_argument('--k_min',  type=float, default=0.05,
                   help='exclude k below this cMpc^-1 (DC-mode cut)')
    p.add_argument('--n_bins', type=int, default=20)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, '*.npz')))
    print(f"found {len(files)} cubes, k_min={args.k_min}")

    rows = []
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        t, g = d['true_norm'], d['gen_norm']
        kc, Pt = radial_ps(t, k_min=args.k_min, n_bins=args.n_bins)
        _ , Pg = radial_ps(g, k_min=args.k_min, n_bins=args.n_bins)
        rat = Pg / np.where(Pt>0, Pt, np.nan)
        # band-average ratios: large k<0.15, mid 0.15<=k<0.40, small k>=0.40
        m_l = (kc < 0.15) & (kc >= args.k_min)
        m_m = (kc >= 0.15) & (kc < 0.40)
        m_s = (kc >= 0.40)
        rel_mse = float(((t-g)**2).sum() / (t**2).sum())
        pix_corr = float(np.corrcoef(t.ravel(), g.ravel())[0,1])
        rows.append({
            'fname': str(d.get('fname','')),
            'rel_mse': rel_mse,
            'pix_corr': pix_corr,
            'ps_large': float(np.nanmean(rat[m_l])) if m_l.any() else np.nan,
            'ps_mid'  : float(np.nanmean(rat[m_m])) if m_m.any() else np.nan,
            'ps_small': float(np.nanmean(rat[m_s])) if m_s.any() else np.nan,
        })
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(files)}]")

    def summ(key):
        a = np.array([r[key] for r in rows])
        a = a[np.isfinite(a)]
        return dict(median=float(np.median(a)), mean=float(a.mean()),
                    iqr=[float(np.percentile(a,25)), float(np.percentile(a,75))],
                    min=float(a.min()), max=float(a.max()), n=int(len(a)))

    out = {'k_min': args.k_min, 'n_files': len(files), 'per_cube': rows,
           'summary': {k: summ(k) for k in ('rel_mse','pix_corr','ps_large','ps_mid','ps_small')}}
    with open(args.out, 'w') as fp:
        json.dump(out, fp, indent=2)

    print("\n=== summary (k>{:.3f}) ===".format(args.k_min))
    for k,v in out['summary'].items():
        print(f"  {k:10s} median={v['median']:.4f}  mean={v['mean']:.4f}  "
              f"IQR=[{v['iqr'][0]:.4f},{v['iqr'][1]:.4f}]  n={v['n']}")


if __name__ == '__main__':
    main()
