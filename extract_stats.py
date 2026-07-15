import glob, json, os, numpy as np, torch, sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device, flush=True)

def ps_stats(t, g, Lpix=3.0, kbins=20):
    x_t = torch.from_numpy(t).to(device).float(); x_t -= x_t.mean()
    x_g = torch.from_numpy(g).to(device).float(); x_g -= x_g.mean()
    N = x_t.shape[-1]
    Ft = torch.fft.rfftn(x_t); Fg = torch.fft.rfftn(x_g)
    k1  = torch.fft.fftfreq (N, device=device)*N
    k1r = torch.fft.rfftfreq(N, device=device)*N
    KX,KY,KZ = torch.meshgrid(k1,k1,k1r, indexing='ij')
    K = (KX**2+KY**2+KZ**2).sqrt() * (2*np.pi/(N*Lpix))
    k_nyq = np.pi/Lpix
    edges = torch.logspace(np.log10(0.05), np.log10(k_nyq*0.95), kbins+1, device=device)
    centers = (edges[:-1]*edges[1:]).sqrt().cpu().numpy()
    Pt=np.full(kbins,np.nan); Pg=np.full(kbins,np.nan); Km=np.full(kbins,np.nan)
    for i in range(kbins):
        m=(K>=edges[i])&(K<edges[i+1])
        if not m.any(): continue
        pt=(Ft[m].abs()**2).mean().item()
        pg=(Fg[m].abs()**2).mean().item()
        Pt[i]=pt; Pg[i]=pg
        Km[i]=((Ft[m]-Fg[m]).abs()**2).mean().item()/max(pt,1e-30)
    return centers, Pt, Pg, Km

out = {}
for tag in ['2x','4x','8x','16x']:
    files = sorted(glob.glob(f'eval_results/ff_{tag}_test_z*/[0-9]*.npz'))
    print(f'{tag}: {len(files)} files', flush=True)
    if not files:
        print(f'  no files for {tag}, skipping'); continue
    per_cube=[]; Pt_all=[]; Pg_all=[]; Km_all=[]
    for i,f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        k, pt, pg, km = ps_stats(d['true_norm'], d['gen_norm'])
        t = d['true_norm']; g = d['gen_norm']
        rel = float(((t-g)**2).sum()/max((t**2).sum(),1e-30))
        bias = float(t.std()/(g.std()+1e-30) - 1.0)
        z = int(os.path.basename(os.path.dirname(f)).split('_z')[-1].split('_')[0])
        per_cube.append(dict(file=os.path.basename(f), z=z,
            rel_mse=rel, bias=bias, gen_seconds=float(d['gen_seconds']),
            Pt=pt.tolist(), Pg=pg.tolist(), kMSE=km.tolist()))
        Pt_all.append(pt); Pg_all.append(pg); Km_all.append(km)
        if (i+1)%25==0: print(f'   {i+1}/{len(files)}', flush=True)
    out[tag] = dict(k=k.tolist(), n=len(files),
        Pt_mean=np.nanmean(Pt_all,0).tolist(), Pg_mean=np.nanmean(Pg_all,0).tolist(),
        Pt_std =np.nanstd (Pt_all,0).tolist(), Pg_std =np.nanstd (Pg_all,0).tolist(),
        kMSE_mean=np.nanmean(Km_all,0).tolist(),
        kMSE_std =np.nanstd (Km_all,0).tolist(),
        per_cube=per_cube)

os.makedirs('plots', exist_ok=True)
with open('plots/ps_per_chain_raw.json','w') as f:
    json.dump(out, f)
print('wrote plots/ps_per_chain_raw.json  size=', os.path.getsize('plots/ps_per_chain_raw.json'), flush=True)
