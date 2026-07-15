"""Extract PDF / equilateral bispec / cross-corr r(k) / params corner for 4 chains.

Also selects best/worst/median cubes per chain and copies their npz to
plots/slices_<tag>/ for local plotting.
"""
import glob, json, os, shutil, numpy as np, torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device, flush=True)

Lpix   = 3.0
KBINS  = 20
NBINS_PDF = 80

def kgrid(N):
    k1  = torch.fft.fftfreq (N, device=device)*N
    k1r = torch.fft.rfftfreq(N, device=device)*N
    KX,KY,KZ = torch.meshgrid(k1,k1,k1r, indexing='ij')
    K = (KX**2+KY**2+KZ**2).sqrt() * (2*np.pi/(N*Lpix))
    edges = torch.logspace(np.log10(0.05), np.log10(np.pi/Lpix*0.95), KBINS+1, device=device)
    centers = (edges[:-1]*edges[1:]).sqrt().cpu().numpy()
    return K, edges, centers

def all_stats(t_np, g_np):
    """Return: bispec_true, bispec_gen, r_of_k (cross-coherence), hist_true, hist_gen"""
    x_t = torch.from_numpy(t_np).to(device).float(); x_t -= x_t.mean()
    x_g = torch.from_numpy(g_np).to(device).float(); x_g -= x_g.mean()
    N = x_t.shape[-1]
    Ft = torch.fft.rfftn(x_t); Fg = torch.fft.rfftn(x_g)
    K, edges, centers = kgrid(N)
    Bt = np.full(KBINS, np.nan); Bg = np.full(KBINS, np.nan); rK = np.full(KBINS, np.nan)
    for i in range(KBINS):
        m = (K >= edges[i]) & (K < edges[i+1])
        if not m.any(): continue
        Ft_m = torch.zeros_like(Ft); Ft_m[m] = Ft[m]
        Fg_m = torch.zeros_like(Fg); Fg_m[m] = Fg[m]
        d_t = torch.fft.irfftn(Ft_m, s=x_t.shape)
        d_g = torch.fft.irfftn(Fg_m, s=x_g.shape)
        Bt[i] = (d_t**3).mean().item()
        Bg[i] = (d_g**3).mean().item()
        # cross-coherence per bin
        num = (Ft[m].conj()*Fg[m]).real.sum()
        den = (Ft[m].abs()**2).sum().sqrt() * (Fg[m].abs()**2).sum().sqrt()
        rK[i] = (num/den).item() if den > 0 else np.nan
    return Bt, Bg, rK

def hist_edges():
    return np.linspace(-5, 5, NBINS_PDF+1)  # z-score range

out = {}
for tag in ['2x','4x','8x','16x']:
    files = sorted(glob.glob(f'eval_results/ff_{tag}_test_z*/[0-9]*.npz'))
    print(f'{tag}: {len(files)} files', flush=True)
    if not files: continue
    Bt_all=[]; Bg_all=[]; rK_all=[]
    hist_t = np.zeros(NBINS_PDF); hist_g = np.zeros(NBINS_PDF)
    edges  = hist_edges()
    per_cube = []
    for i,f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        t = d['true_norm']; g = d['gen_norm']
        Bt, Bg, rK = all_stats(t, g)
        Bt_all.append(Bt); Bg_all.append(Bg); rK_all.append(rK)
        ht,_ = np.histogram(t.ravel(), bins=edges); hist_t += ht
        hg,_ = np.histogram(g.ravel(), bins=edges); hist_g += hg
        z  = int(os.path.basename(os.path.dirname(f)).split('_z')[-1].split('_')[0])
        rel = float(((t-g)**2).sum()/max((t**2).sum(),1e-30))
        bias= float(t.std()/(g.std()+1e-30)-1.0)
        par = d['params'] if 'params' in d.files else None
        per_cube.append(dict(file=os.path.basename(f), path=f, z=z, rel_mse=rel, bias=bias,
                             params=(par.tolist() if par is not None else None)))
        if (i+1)%25==0: print(f'   {i+1}/{len(files)}', flush=True)
    _, _, centers = kgrid(256)
    out[tag] = dict(
        k = centers.tolist(),
        n = len(files),
        Bt_mean = np.nanmean(Bt_all,0).tolist(),
        Bg_mean = np.nanmean(Bg_all,0).tolist(),
        Bt_std  = np.nanstd (Bt_all,0).tolist(),
        Bg_std  = np.nanstd (Bg_all,0).tolist(),
        rK_mean = np.nanmean(rK_all,0).tolist(),
        rK_std  = np.nanstd (rK_all,0).tolist(),
        pdf_edges = edges.tolist(),
        pdf_true  = hist_t.tolist(),
        pdf_gen   = hist_g.tolist(),
        per_cube  = per_cube,
    )
    # Pick 3 representative cubes: best/median/worst by rel_MSE
    idx_sorted = sorted(range(len(per_cube)), key=lambda i: per_cube[i]['rel_mse'])
    picks = {'best': idx_sorted[0], 'median': idx_sorted[len(idx_sorted)//2], 'worst': idx_sorted[-1]}
    outdir = f'plots/slices_{tag}'
    os.makedirs(outdir, exist_ok=True)
    for label, ix in picks.items():
        src = per_cube[ix]['path']
        dst = os.path.join(outdir, f'{label}_rel{per_cube[ix]["rel_mse"]:.3f}_z{per_cube[ix]["z"]}.npz')
        shutil.copy(src, dst)
        print(f'  picked {tag} {label}: {os.path.basename(src)} -> {dst}', flush=True)

os.makedirs('plots', exist_ok=True)
with open('plots/extras_raw.json','w') as f:
    json.dump(out, f)
print('wrote plots/extras_raw.json  size=', os.path.getsize('plots/extras_raw.json'), flush=True)
