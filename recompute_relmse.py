"""Recompute rel_MSE with per-cube mean subtracted (fluctuation MSE).
Re-select best/median/worst per chain and copy npz to plots/slices2_<tag>/.
Emit updated JSON.
"""
import glob, os, json, shutil, numpy as np

out = {}
for tag in ['2x','4x','8x','16x']:
    files = sorted(glob.glob(f'eval_results/ff_{tag}_test_z*/[0-9]*.npz'))
    print(f'{tag}: {len(files)} files', flush=True)
    per = []
    for i,f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        t = d['true_norm']; g = d['gen_norm']
        tc = t - t.mean(); gc = g - g.mean()
        num = ((tc-gc)**2).sum()
        den = (tc**2).sum() + 1e-30
        rel_flu = float(num / den)
        rel_old = float(((t-g)**2).sum() / max((t**2).sum(), 1e-30))
        bias    = float(t.std() / (g.std()+1e-30) - 1.0)
        dc_off  = float(g.mean() - t.mean())
        z = int(os.path.basename(os.path.dirname(f)).split('_z')[-1].split('_')[0])
        per.append(dict(file=os.path.basename(f), path=f, z=z,
                        rel_mse_flu=rel_flu, rel_mse_old=rel_old,
                        bias=bias, dc_offset=dc_off,
                        t_mean=float(t.mean()), t_std=float(t.std()),
                        g_mean=float(g.mean()), g_std=float(g.std())))
        if (i+1)%25==0: print(f'   {i+1}/{len(files)}', flush=True)
    idx = sorted(range(len(per)), key=lambda i: per[i]['rel_mse_flu'])
    picks = {'best': idx[0], 'median': idx[len(idx)//2], 'worst': idx[-1]}
    outdir = f'plots/slices2_{tag}'
    os.makedirs(outdir, exist_ok=True)
    for lbl, ix in picks.items():
        src = per[ix]['path']
        dst = os.path.join(outdir, f'{lbl}_relflu{per[ix]["rel_mse_flu"]:.3f}_z{per[ix]["z"]}.npz')
        shutil.copy(src, dst)
        print(f'  pick {tag} {lbl}: {os.path.basename(src)} rel_flu={per[ix]["rel_mse_flu"]:.3f} rel_old={per[ix]["rel_mse_old"]:.3f}', flush=True)
    out[tag] = dict(per_cube=per, picks={k: per[v]['file'] for k,v in picks.items()},
                    median_rel_flu=float(np.median([p['rel_mse_flu'] for p in per])),
                    median_rel_old=float(np.median([p['rel_mse_old'] for p in per])),
                    median_bias   =float(np.median([p['bias'] for p in per])))

os.makedirs('plots', exist_ok=True)
with open('plots/relmse_fluct.json','w') as f:
    json.dump(out, f)
print('done. size=', os.path.getsize('plots/relmse_fluct.json'), flush=True)
for tag in ['2x','4x','8x','16x']:
    o = out[tag]
    print(f"  {tag}: median rel_MSE_fluct={o['median_rel_flu']:.3f}  (old={o['median_rel_old']:.3f})  bias={o['median_bias']:+.3f}")
