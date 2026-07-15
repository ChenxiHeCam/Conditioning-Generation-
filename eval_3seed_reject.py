# 3-seed self-consistency rejection (no ground truth used to pick seed).
# For each cube: compute PS for 3 seeds, pick the seed closest to the per-cube mean PS.
import os, glob, argparse, json
import numpy as np


def radial_ps(x, Lpix=3.0, n_bins=20, k_min=0.05):
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
        P_b[i] = P[m].mean() if m.any() else np.nan
    return centers, P_b


def metrics_for_cube(true_x, gen_x, k_min=0.05):
    kc, Pt = radial_ps(true_x, k_min=k_min)
    _ , Pg = radial_ps(gen_x,  k_min=k_min)
    rat = Pg / np.where(Pt > 0, Pt, np.nan)
    m_l = (kc < 0.15) & (kc >= k_min)
    m_m = (kc >= 0.15) & (kc < 0.40)
    m_s = (kc >= 0.40)
    return dict(
        rel_mse =float(((true_x - gen_x)**2).sum() / (true_x**2).sum()),
        pix_corr=float(np.corrcoef(true_x.ravel(), gen_x.ravel())[0,1]),
        ps_large=float(np.nanmean(rat[m_l])) if m_l.any() else np.nan,
        ps_mid  =float(np.nanmean(rat[m_m])) if m_m.any() else np.nan,
        ps_small=float(np.nanmean(rat[m_s])) if m_s.any() else np.nan,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+', required=True,
                   help='3 directories of (true,gen) npz from different seeds')
    p.add_argument('--out',  required=True)
    args = p.parse_args()
    assert len(args.dirs) == 3, "need exactly 3 seed dirs"

    # enumerate matching files by basename
    files_per_seed = [sorted(glob.glob(os.path.join(d, '*.npz'))) for d in args.dirs]
    assert all(len(fs) == len(files_per_seed[0]) for fs in files_per_seed), \
        "seed dirs must have the same number of cubes"
    N = len(files_per_seed[0])
    print(f"{N} cubes, 3 seeds")

    n_bins = 20
    P_all = np.zeros((N, 3, n_bins))  # PS for each (cube, seed)
    # also store per-seed metrics so we can report all three
    all_metrics = {s: [] for s in range(3)}

    for ci in range(N):
        for si in range(3):
            d = np.load(files_per_seed[si][ci], allow_pickle=True)
            t, g = d['true_norm'], d['gen_norm']
            kc, Pg = radial_ps(g, k_min=0.05, n_bins=n_bins)
            P_all[ci, si] = Pg
            all_metrics[si].append(metrics_for_cube(t, g))
        if (ci+1) % 20 == 0:
            print(f"  [{ci+1}/{N}]")

    # ===== self-consistency rejection =====
    # for each cube, the chosen seed minimises log-PS distance to the per-cube mean PS
    Lp = np.log(np.clip(P_all, 1e-20, None))     # (N, 3, K)
    Lp_mean = Lp.mean(axis=1, keepdims=True)     # (N, 1, K)
    dist = ((Lp - Lp_mean)**2).mean(axis=-1)     # (N, 3) — no GT used
    chosen = dist.argmin(axis=1)                  # (N,)

    # ===== build "selected" metric list using chosen seed per cube =====
    selected = [all_metrics[chosen[ci]][ci] for ci in range(N)]

    def summ(L, key):
        a = np.array([r[key] for r in L])
        a = a[np.isfinite(a)]
        return dict(median=float(np.median(a)), mean=float(a.mean()),
                    iqr=[float(np.percentile(a,25)), float(np.percentile(a,75))],
                    min=float(a.min()), max=float(a.max()), n=int(len(a)))

    out = {
        'n_cubes': N,
        'chosen_seed_hist': [int((chosen == s).sum()) for s in range(3)],
        'per_seed_summary': {s: {k: summ(all_metrics[s], k)
                                  for k in ('rel_mse','pix_corr','ps_large','ps_mid','ps_small')}
                             for s in range(3)},
        'self_consistency_summary': {k: summ(selected, k)
                                       for k in ('rel_mse','pix_corr','ps_large','ps_mid','ps_small')},
    }
    with open(args.out, 'w') as fp:
        json.dump(out, fp, indent=2)

    print("\n=== chosen seed histogram ===", out['chosen_seed_hist'])
    print("\n=== per-seed (median) ===")
    print(f"  {'seed':5s} {'rel_mse':>9s} {'ps_large':>9s} {'ps_mid':>9s} {'ps_small':>9s} {'pix_corr':>9s}")
    for s in range(3):
        ps = out['per_seed_summary'][s]
        print(f"  {s:5d} {ps['rel_mse']['median']:9.4f} {ps['ps_large']['median']:9.4f} "
              f"{ps['ps_mid']['median']:9.4f} {ps['ps_small']['median']:9.4f} {ps['pix_corr']['median']:9.4f}")
    print(f"\n=== self-consistency (median) ===")
    s = out['self_consistency_summary']
    print(f"  rel_mse={s['rel_mse']['median']:.4f} ps_L={s['ps_large']['median']:.4f} "
          f"ps_M={s['ps_mid']['median']:.4f} ps_S={s['ps_small']['median']:.4f} corr={s['pix_corr']['median']:.4f}")
    print(f"  mean ps_large={s['ps_large']['mean']:.4f} (tail catcher)")


if __name__ == '__main__':
    main()
