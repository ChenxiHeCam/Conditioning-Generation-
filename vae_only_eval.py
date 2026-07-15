"""
VAE-only test-set eval: tile-stitch encode->decode of full 256^3 test cubes,
compute PS / equilateral bispec / squeezed bispec / rel_MSE / pixel corr.

Usage:
  python vae_only_eval.py \
    --vae_ckpts 2x:/.../vae_2x_final.pt 4x:... 8x:... 16x:... \
    --data_root /root/autodl-tmp/ASR21cm/varying_astro \
    --out_dir /root/autodl-tmp/vae_only_eval \
    --n_cubes 20

Writes per-chain metrics JSON + compact summary npz + figures comparing the
four compression points.
"""
import os, sys, argparse, time, json
import numpy as np
import torch

from dataset import T21Dataset, load_mat
from models.vae import VAE3D


LPIX = 3.0


def hann3d(s):
    w = np.hanning(s)
    return (w[:, None, None] * w[None, :, None] * w[None, None, :]).astype(np.float32)


def kgrid(N, dev):
    k1  = torch.fft.fftfreq(N,  device=dev) * N
    k1r = torch.fft.rfftfreq(N, device=dev) * N
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    return (KX**2 + KY**2 + KZ**2).sqrt() * (2 * np.pi / (N * LPIX))


def ps_d2(x_mk, Kphys, edges, dev):
    N = x_mk.shape[-1]
    L = N * LPIX
    xt = torch.from_numpy(x_mk).to(dev).float()
    xt = xt - xt.mean()
    F = torch.fft.rfftn(xt)
    P = (F.real**2 + F.imag**2) * (L**3 / N**6)
    nb = len(edges) - 1
    d2 = torch.full((nb,), float('nan'), device=dev)
    kc = torch.zeros(nb, device=dev)
    for i in range(nb):
        m = (Kphys >= edges[i]) & (Kphys < edges[i+1])
        if m.any():
            km = Kphys[m].mean()
            d2[i] = P[m].mean() * km**3 / (2 * np.pi**2)
            kc[i] = km
    return kc.cpu().numpy(), d2.cpu().numpy()


def bispec(x_mk, Kphys, edges, dev, squeezed_band=None):
    xt = torch.from_numpy(x_mk).to(dev).float()
    xt = xt - xt.mean()
    F = torch.fft.rfftn(xt)
    nb = len(edges) - 1
    B = torch.zeros(nb, device=dev)
    kc = np.sqrt(edges[:-1] * edges[1:])
    dL = None
    if squeezed_band is not None:
        Fm = torch.zeros_like(F)
        m = (Kphys >= squeezed_band[0]) & (Kphys < squeezed_band[1])
        Fm[m] = F[m]
        dL = torch.fft.irfftn(Fm, s=xt.shape)
    for i in range(nb):
        Fm = torch.zeros_like(F)
        m = (Kphys >= edges[i]) & (Kphys < edges[i+1])
        Fm[m] = F[m]
        dk = torch.fft.irfftn(Fm, s=xt.shape)
        B[i] = (dk * dk * (dL if dL is not None else dk)).mean()
    return kc, B.cpu().numpy()


@torch.no_grad()
def vae_stitch(vae, true_full, patch=64, stride=32, batch=8, device='cuda'):
    """Encode->decode every patch and Hann-blend back to 256^3."""
    N = true_full.shape[-1]
    out = np.zeros((N, N, N), dtype=np.float32)
    wsum = np.zeros_like(out)
    win = hann3d(patch)
    coords = list(range(0, N - patch + 1, stride))
    if coords[-1] != N - patch:
        coords.append(N - patch)
    origins = [(i, j, k) for i in coords for j in coords for k in coords]
    true_t = torch.from_numpy(true_full.astype(np.float32))
    for s in range(0, len(origins), batch):
        chunk = origins[s:s+batch]
        B = len(chunk)
        x = torch.empty(B, 1, patch, patch, patch, device=device)
        for b, (i, j, k) in enumerate(chunk):
            x[b, 0] = true_t[i:i+patch, j:j+patch, k:k+patch]
        mu, _ = vae.encoder(x)
        rec = vae.decoder(mu).float().cpu().numpy()
        for b, (i, j, k) in enumerate(chunk):
            sl = np.s_[i:i+patch, j:j+patch, k:k+patch]
            out[sl] += rec[b, 0] * win
            wsum[sl] += win
    return out / np.clip(wsum, 1e-8, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpts', nargs='+', required=True,
                   help='tag:path pairs')
    p.add_argument('--data_root', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--redshift', type=int, default=10)
    p.add_argument('--n_cubes', type=int, default=20)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda'

    ds = T21Dataset(args.data_root, 64, redshifts=[args.redshift], split='test')
    files = ds.files[:args.n_cubes]
    print(f"test files: {len(ds.files)}; using {len(files)}", flush=True)

    N = 256
    Kphys = kgrid(N, dev)
    k_nyq = np.pi / LPIX
    k_box = 2 * np.pi / (N * LPIX)
    ps_edges = np.logspace(np.log10(2 * k_box), np.log10(k_nyq * 0.95), 25)
    bs_edges = np.logspace(np.log10(0.05),      np.log10(k_nyq * 0.95), 9)
    sq_band = (k_box, 0.04)

    chains = []
    for spec in args.vae_ckpts:
        tag, path = spec.split(':', 1)
        chains.append((tag, path))

    results = {}
    truth_ps, truth_beq, truth_bsq = [], [], []
    truth_collected = False

    for tag, path in chains:
        print(f"\n=== {tag} ({path}) ===", flush=True)
        ck = torch.load(path, map_location=dev)
        cfg = ck['model_config']
        vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'], base_ch=cfg['base_ch'],
                    ch_mults=tuple(cfg['ch_mults'])).to(dev).eval()
        vae.load_state_dict(ck['model'])

        acc = dict(rel=[], corr=[], ps=[], beq=[], bsq=[])
        for ci, fname in enumerate(files):
            t0 = time.time()
            true_full = load_mat(os.path.join(ds.t21_dir, fname))
            true_full = (true_full - ds.t21_mean) / ds.t21_std
            gen_full = vae_stitch(vae, true_full, device=dev)
            std, mean = float(ds.t21_std), float(ds.t21_mean)
            tm = true_full * std + mean
            gm = gen_full * std + mean

            rel = float(((true_full - gen_full)**2).sum() / (true_full**2).sum())
            corr = float(np.corrcoef(true_full.ravel()[::50], gen_full.ravel()[::50])[0, 1])
            _, d2g = ps_d2(gm, Kphys, ps_edges, dev)
            _, beg = bispec(gm, Kphys, bs_edges, dev)
            _, bsg = bispec(gm, Kphys, bs_edges, dev, squeezed_band=sq_band)
            acc['rel'].append(rel); acc['corr'].append(corr)
            acc['ps'].append(d2g); acc['beq'].append(beg); acc['bsq'].append(bsg)

            if not truth_collected:
                kc, d2t = ps_d2(tm, Kphys, ps_edges, dev)
                kbe, bet = bispec(tm, Kphys, bs_edges, dev)
                _,   bst = bispec(tm, Kphys, bs_edges, dev, squeezed_band=sq_band)
                truth_ps.append(d2t); truth_beq.append(bet); truth_bsq.append(bst)

            dt = time.time() - t0
            print(f"  [{ci+1}/{len(files)}] {fname}  rel={rel:.3f} corr={corr:.3f}  {dt:.1f}s",
                  flush=True)
        truth_collected = True
        results[tag] = {k: np.array(v) if k in ('rel', 'corr') else np.stack(v)
                        for k, v in acc.items()}
        del vae; torch.cuda.empty_cache()

    truth_ps = np.stack(truth_ps); truth_beq = np.stack(truth_beq); truth_bsq = np.stack(truth_bsq)
    np.savez(os.path.join(args.out_dir, 'vae_only_metrics.npz'),
             kc=kc, kbe=kbe, ps_true=truth_ps, beq_true=truth_beq, bsq_true=truth_bsq,
             **{f'{tag}_ps': r['ps']   for tag, r in results.items()},
             **{f'{tag}_beq': r['beq'] for tag, r in results.items()},
             **{f'{tag}_bsq': r['bsq'] for tag, r in results.items()},
             **{f'{tag}_rel': r['rel'] for tag, r in results.items()},
             **{f'{tag}_corr': r['corr'] for tag, r in results.items()},
             files=np.array(files))

    summary = {}
    for tag, r in results.items():
        rat_ps = r['ps'] / np.where(truth_ps > 0, truth_ps, np.nan)
        mps = np.nanmedian(rat_ps, 0)
        summary[tag] = dict(
            rel_mse=float(np.median(r['rel'])),
            pix_corr=float(np.median(r['corr'])),
            ps_large=float(np.nanmean(mps[(kc > 0.05) & (kc < 0.15)])),
            ps_mid=float(np.nanmean(mps[(kc >= 0.15) & (kc < 0.40)])),
            ps_small=float(np.nanmean(mps[kc >= 0.40])),
            bispec_eq=float(np.nanmean(np.nanmedian(
                r['beq'] / np.where(np.abs(truth_beq) > 1e-30, truth_beq, np.nan), 0))),
            bispec_sq=float(np.nanmean(np.nanmedian(
                r['bsq'] / np.where(np.abs(truth_bsq) > 1e-30, truth_bsq, np.nan), 0))),
        )
    with open(os.path.join(args.out_dir, 'vae_only_summary.json'), 'w') as f:
        json.dump(dict(n_cubes=len(files), summary=summary), f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
