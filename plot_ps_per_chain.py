"""Power spectrum + per-k relative MSE, all chains overlaid.

For each chain: averages P(k) across all test cubes, plots
  (a) |P(k)|  true vs gen
  (b) P_gen / P_true
  (c) per-k rel MSE  <|delta_gen-delta_true|^2>_k / <|delta_true|^2>_k
"""
import os, argparse, glob, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ps_and_kmse(true_np, gen_np, Lpix=3.0, kbins=20, device='cuda'):
    """Return (k_centers, P_true, P_gen, kMSE) all shape (kbins,)."""
    x_t = torch.from_numpy(true_np).to(device).float(); x_t -= x_t.mean()
    x_g = torch.from_numpy(gen_np ).to(device).float(); x_g -= x_g.mean()
    N = x_t.shape[-1]
    Ft = torch.fft.rfftn(x_t); Fg = torch.fft.rfftn(x_g)
    k1  = torch.fft.fftfreq (N, device=device) * N
    k1r = torch.fft.rfftfreq(N, device=device) * N
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = (KX**2 + KY**2 + KZ**2).sqrt()
    kbox = 2*np.pi / (N*Lpix); k_nyq = np.pi/Lpix
    edges   = torch.logspace(np.log10(0.05), np.log10(k_nyq*0.95), kbins+1, device=device)
    centers = (edges[:-1]*edges[1:]).sqrt().cpu().numpy()
    K = Kmag * kbox
    Pt = np.full(kbins, np.nan); Pg = np.full(kbins, np.nan); Km = np.full(kbins, np.nan)
    for i in range(kbins):
        m = (K >= edges[i]) & (K < edges[i+1])
        if not m.any(): continue
        pt = (Ft[m].abs()**2).mean().item()
        pg = (Fg[m].abs()**2).mean().item()
        km = ((Ft[m]-Fg[m]).abs()**2).mean().item() / max(pt, 1e-30)
        Pt[i], Pg[i], Km[i] = pt, pg, km
    return centers, Pt, Pg, Km


def collect(eval_root, tag, device):
    files = sorted(glob.glob(os.path.join(eval_root, f'ff_{tag}_test_z*_ep*', '[0-9]*.npz')))
    if not files: return None
    Pt_list, Pg_list, K_list = [], [], []
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        k, pt, pg, km = ps_and_kmse(d['true_norm'], d['gen_norm'], device=device)
        Pt_list.append(pt); Pg_list.append(pg); K_list.append(km)
        if (i+1) % 20 == 0: print(f'  {tag}: {i+1}/{len(files)}')
    return dict(tag=tag, k=k,
                Pt=np.nanmean(Pt_list, 0), Pg=np.nanmean(Pg_list, 0),
                kMSE=np.nanmean(K_list, 0), n=len(files))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval_root', default='/home/ch2067/rds/hpc-work/21cm_gen/eval_results')
    p.add_argument('--out', default='plots/')
    p.add_argument('--tags', nargs='+', default=['2x','4x','8x','16x'])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}')
    chains = [c for c in (collect(args.eval_root, t, device) for t in args.tags) if c is not None]
    if not chains: return
    colors = ['tab:blue','tab:orange','tab:green','tab:red']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes
    for i, c in enumerate(chains):
        ax1.loglog(c['k'], c['Pt'], color=colors[i], ls='--', alpha=0.5)
        ax1.loglog(c['k'], c['Pg'], color=colors[i], lw=2, label=f"{c['tag']} (n={c['n']})")
        ax2.semilogx(c['k'], c['Pg']/c['Pt'], color=colors[i], lw=2, label=c['tag'])
        ax3.loglog (c['k'], c['kMSE'],        color=colors[i], lw=2, label=c['tag'])
    ax1.set_xlabel('k [cMpc^-1]'); ax1.set_ylabel('P(k)  (z-score)')
    ax1.set_title('true (dashed) vs gen (solid)'); ax1.legend(); ax1.grid(alpha=.3)
    ax2.axhline(1, color='k', ls=':'); ax2.axhspan(0.9,1.1,color='g',alpha=.1)
    ax2.set_xlabel('k'); ax2.set_ylabel('P_gen / P_true'); ax2.set_ylim(0.3,1.7)
    ax2.set_title('PS ratio'); ax2.legend(); ax2.grid(alpha=.3)
    ax3.set_xlabel('k'); ax3.set_ylabel('rel MSE(k)')
    ax3.set_title('per-k relative MSE'); ax3.legend(); ax3.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'ps_per_chain.png'), dpi=140)
    plt.savefig(os.path.join(args.out, 'ps_per_chain.pdf'))
    print(f'saved {args.out}/ps_per_chain.png')

    summary = {c['tag']: dict(k=c['k'].tolist(), Pt=c['Pt'].tolist(),
                              Pg=c['Pg'].tolist(), kMSE=c['kMSE'].tolist(),
                              n=c['n']) for c in chains}
    with open(os.path.join(args.out,'ps_per_chain.json'),'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
