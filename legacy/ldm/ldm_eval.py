"""
Comprehensive LDM eval — diagnose what was learned well and what wasn't.
Outputs metrics organized by failure mode, with concrete loss/training fixes.

Usage:
  python ldm_eval.py --ldm_ckpt /path/to/ldm_epochXXXX.pt
"""
import os, argparse, json, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.vae import VAE3D
from ldm_dataset import build_train_dataset
from ldm_unet   import LDMUNet3D
from train_ldm import edm_precond, denoise, heun_sample
from utils.power_spectrum import power_spectrum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pearson(a, b):
    a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
    return float((a*b).sum() / (np.sqrt((a**2).sum()*(b**2).sum()) + 1e-30))


def moments(x):
    m = float(x.mean()); s = float(x.std() + 1e-30)
    c = x - m
    return m, s, float((c**3).mean()/s**3), float((c**4).mean()/s**4 - 3)


def ps_curve(X, R):
    """Returns k, ratio_curve, ratio_large/mid/small (valid bins only)."""
    k, ps_o = power_spectrum(X)
    _, ps_r = power_spectrum(R)
    valid = ps_o.mean(0) > 1e-20
    ratio = ps_r.mean(0) / np.maximum(ps_o.mean(0), 1e-30)
    return k, ratio, valid


def cross_phase_corr(X, R):
    """Phase-aware cross-correlation in Fourier domain (per sample, averaged)."""
    out = []
    for i in range(X.shape[0]):
        x = X[i] - X[i].mean()
        r = R[i] - R[i].mean()
        Fx = np.fft.rfftn(x); Fr = np.fft.rfftn(r)
        # cross-power / sqrt(P_x * P_r) -- coherence
        num = (Fx * np.conj(Fr)).real.sum()
        denom = np.sqrt((np.abs(Fx)**2).sum() * (np.abs(Fr)**2).sum()) + 1e-30
        out.append(num / denom)
    return float(np.mean(out))


def pdf_ks_distance(X, R, n_bins=80):
    """KS-like distance between voxel-value histograms (max CDF difference)."""
    x_flat = X.flatten(); r_flat = R.flatten()
    lo = min(x_flat.min(), r_flat.min()); hi = max(x_flat.max(), r_flat.max())
    edges = np.linspace(lo, hi, n_bins+1)
    hx, _ = np.histogram(x_flat, bins=edges, density=False)
    hr, _ = np.histogram(r_flat, bins=edges, density=False)
    cx = np.cumsum(hx) / hx.sum()
    cr = np.cumsum(hr) / hr.sum()
    return float(np.max(np.abs(cx - cr)))


def ionization_fraction(X, threshold=-0.5):
    """Volume fraction below a threshold (proxy for ionized regions in normalized
    units). With patches normalized to mean ~0 and std varying, threshold -0.5
    isolates the strongly-absorbing or strongly-ionized tail.
    Returns mean fraction over samples."""
    return float((X < threshold).mean())


# ---------------------------------------------------------------------------
# Generation (one chunk at a time)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, vae, items, latent_mean, latent_std, sigma_data, device,
             chunk=8, num_steps=18, cfg_ic=1.0, cfg_params=1.0):
    n = len(items)
    ic_d  = torch.stack([it['ic_delta'] for it in items]).to(device)
    ic_v  = torch.stack([it['ic_vbv']   for it in items]).to(device)
    params = torch.stack([it['params'][:4] for it in items]).to(device)
    redshift = torch.stack([it['redshift']   for it in items]).to(device)
    x_true = torch.stack([it['patch'] for it in items]).to(device)

    recon_list = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        z = heun_sample(model, e-s, (8, 32, 32, 32),
                        ic_d[s:e], ic_v[s:e], params[s:e], redshift[s:e],
                        sigma_data=sigma_data, num_steps=num_steps,
                        cfg_ic=cfg_ic, cfg_params=cfg_params,
                        null_param=None, device=device)
        z = z * latent_std + latent_mean
        recon_list.append(vae.decoder(z).cpu())
    return (x_true.cpu().squeeze(1).numpy(),
            torch.cat(recon_list).squeeze(1).numpy(),
            ic_d.cpu().squeeze(1).numpy())


# ---------------------------------------------------------------------------
# Per-group eval
# ---------------------------------------------------------------------------

def eval_group(items, model, vae, latent_mean, latent_std, sigma_data, device,
               tag, two_runs_for_diversity=True):
    if not items:
        return None
    X, R, IC = generate(model, vae, items, latent_mean, latent_std, sigma_data, device)
    mse = float(((R - X) ** 2).mean())
    var_t = float(X.var())
    var_r = float(R.var())
    rel_mse = mse / max(var_t, 1e-30)

    k, ratio, valid = ps_curve(X, R)
    ps_large = float(ratio[(k<0.1)&valid].mean()) if ((k<0.1)&valid).any() else float('nan')
    ps_mid   = float(ratio[(k>=0.1)&(k<0.5)&valid].mean()) if ((k>=0.1)&(k<0.5)&valid).any() else float('nan')
    ps_small = float(ratio[(k>=0.5)&valid].mean()) if ((k>=0.5)&valid).any() else float('nan')

    # Per-sample stats
    pcorrs = [pearson(R[i], X[i]) for i in range(X.shape[0])]
    iccorr_r = [pearson(R[i], IC[i]) for i in range(X.shape[0])]
    iccorr_t = [pearson(X[i], IC[i]) for i in range(X.shape[0])]
    moms_t = [moments(X[i]) for i in range(X.shape[0])]
    moms_r = [moments(R[i]) for i in range(X.shape[0])]

    cross_coh = cross_phase_corr(X, R)
    ks = pdf_ks_distance(X, R)
    ion_t = ionization_fraction(X); ion_r = ionization_fraction(R)

    out = dict(
        n=int(X.shape[0]),
        mse=mse, rel_mse=rel_mse, var_true=var_t, var_recon=var_r,
        ps_large=ps_large, ps_mid=ps_mid, ps_small=ps_small,
        ps_k=k.tolist(), ps_ratio=ratio.tolist(), ps_valid=valid.tolist(),
        pixel_corr_mean=float(np.mean(pcorrs)),
        pixel_corr_std =float(np.std(pcorrs)),
        ic_corr_recon  =float(np.mean(iccorr_r)),
        ic_corr_true   =float(np.mean(iccorr_t)),
        std_true =float(np.mean([m[1] for m in moms_t])),
        std_recon=float(np.mean([m[1] for m in moms_r])),
        skew_true =float(np.mean([m[2] for m in moms_t])),
        skew_recon=float(np.mean([m[2] for m in moms_r])),
        kurt_true =float(np.mean([m[3] for m in moms_t])),
        kurt_recon=float(np.mean([m[3] for m in moms_r])),
        cross_coherence=cross_coh,
        pdf_ks=ks,
        ion_frac_true=ion_t, ion_frac_recon=ion_r,
    )

    # Sample diversity: re-generate same conditioning, compare two samples
    if two_runs_for_diversity:
        _, R2, _ = generate(model, vae, items, latent_mean, latent_std, sigma_data, device)
        diversity_mse = float(((R2 - R) ** 2).mean())
        diversity_corr = float(np.mean([pearson(R[i], R2[i]) for i in range(R.shape[0])]))
        out['sample_diversity_mse'] = diversity_mse
        out['sample_diversity_corr'] = diversity_corr
        # Diversity ratio: how much do two LDM samples differ vs how much do true field and one LDM sample differ
        out['sample_diversity_rel'] = diversity_mse / max(mse, 1e-30)
    return out


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def diagnose(all_results):
    """Compare across groups, return list of (issue, severity, fix)."""
    issues = []

    # Use 'all' if available, otherwise pick first
    if 'all' in all_results:
        ref = all_results['all']
    else:
        ref = next(iter(all_results.values()))
    if ref is None: return issues

    def deviation_from_one(v):
        return abs(v - 1.0) if v == v else 0  # nan-safe

    # 1. PS ratio overshoot or undershoot
    for band, v in [('large', ref['ps_large']), ('mid', ref['ps_mid']), ('small', ref['ps_small'])]:
        if v != v: continue
        dev = deviation_from_one(v)
        if dev > 0.20:
            severity = 'HIGH' if dev > 0.30 else 'MED'
            direction = 'overshoot' if v > 1 else 'undershoot'
            fix_map = {
                'large': 'add low-k weighting in EDM loss, e.g. weighted MSE by 1/sigma at low k',
                'mid':   'increase resolution-1 attention or reduce ic_stem capacity',
                'small': 'add explicit small-scale FFT loss or train longer (small-scale needs more iterations)',
            }
            issues.append((f'PS {band} {direction} {v:.3f}', severity, fix_map[band]))
        elif dev < 0.05:
            issues.append((f'PS {band} {v:.3f} GOOD', 'ok', ''))

    # 2. Kurtosis gap
    kr, kt = ref['kurt_recon'], ref['kurt_true']
    if abs(kr - kt) > 0.5:
        sev = 'HIGH' if abs(kr - kt) > 1.0 else 'MED'
        issues.append((f'Kurtosis underprediction: {kr:.2f} vs {kt:.2f}', sev,
                       'add explicit higher-moment loss or train longer; current loss is MSE-only which is mean-seeking'))

    # 3. Pixel correlation
    pc = ref['pixel_corr_mean']
    if pc < 0.7:
        issues.append((f'Pixel correlation {pc:.2f} < 0.7', 'HIGH',
                       'conditioning is too weak — increase CFG params scale at inference, or reduce conditioning dropout in training'))
    elif pc > 0.9:
        issues.append((f'Pixel correlation {pc:.2f} EXCELLENT', 'ok', ''))

    # 4. IC correlation: model should match the true IC-T21 sign and magnitude
    ic_r, ic_t = ref['ic_corr_recon'], ref['ic_corr_true']
    if abs(ic_r - ic_t) > 0.10:
        issues.append((f'IC alignment off: recon {ic_r:.3f} vs truth {ic_t:.3f}', 'MED',
                       'IC stem capacity may be too small or input concat is insufficient — try larger ic_stem_out'))

    # 5. PDF mismatch
    ks = ref['pdf_ks']
    if ks > 0.10:
        sev = 'HIGH' if ks > 0.20 else 'MED'
        issues.append((f'PDF KS distance {ks:.3f}', sev,
                       'add distribution-matching loss (Wasserstein, moment-matching), or train longer'))

    # 6. Ionization fraction
    ir, it = ref['ion_frac_recon'], ref['ion_frac_true']
    if abs(ir - it) / max(it, 1e-3) > 0.20:
        issues.append((f'Ionization fraction: {ir:.3f} vs {it:.3f}', 'MED',
                       'physics-specific issue — model not producing enough ionized voxels; consider physics-aware loss'))

    # 7. Sample diversity
    if 'sample_diversity_rel' in ref:
        div = ref['sample_diversity_rel']
        if div < 0.05:
            issues.append((f'Sample diversity {div:.3f} too LOW (mode collapse)', 'MED',
                           'reduce CFG strength at inference, or reduce conditioning dropout during training'))
        elif div > 2.0:
            issues.append((f'Sample diversity {div:.3f} too HIGH', 'MED',
                           'conditioning not strong enough; increase CFG scale or train longer'))
        else:
            issues.append((f'Sample diversity {div:.3f} REASONABLE', 'ok', ''))

    # Suite-level imbalance (astro vs IC)
    if 'astro' in all_results and 'IC' in all_results and all_results['astro'] and all_results['IC']:
        a = all_results['astro']; b = all_results['IC']
        pc_gap = a['pixel_corr_mean'] - b['pixel_corr_mean']
        if abs(pc_gap) > 0.10:
            worse = 'IC' if pc_gap > 0 else 'astro'
            issues.append((f'Suite gap: {worse} much worse (pixel_corr astro={a["pixel_corr_mean"]:.3f} vs IC={b["pixel_corr_mean"]:.3f})',
                           'HIGH', f'increase {worse} sampling weight or add per-suite finetune'))

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ldm_ckpt', required=True)
    p.add_argument('--vae_ckpt', default='/root/autodl-tmp/checkpoints/vae_v4/vae_v4_final.pt')
    p.add_argument('--out',      default='/root/autodl-tmp/ldm_eval.json')
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    args = p.parse_args()

    device = torch.device('cuda')
    # Load VAE
    vae_ck = torch.load(args.vae_ckpt, map_location=device)
    cfg = vae_ck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
                base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(device)
    vae.load_state_dict(vae_ck['model']); vae.eval()
    latent_mean = vae_ck['latent_mean'].view(1, -1, 1, 1, 1).to(device)
    latent_std  = vae_ck['latent_std' ].view(1, -1, 1, 1, 1).to(device)

    # Load LDM
    ldm_ck = torch.load(args.ldm_ckpt, map_location=device)
    sigma_data = ldm_ck.get('sigma_data', 1.0)
    model = LDMUNet3D().to(device)
    model.load_state_dict(ldm_ck['model']); model.eval()
    print(f"LDM ep {ldm_ck['epoch']}, sigma_data={sigma_data:.3f}")

    # Build full set (val or test)
    print(f"\nUsing split = {args.split}")
    val_ds, _ = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        patches_per_cube=1, augment=False, split=args.split,
    )
    val_items_all   = [val_ds[i] for i in range(len(val_ds))]
    val_items_astro = [it for it in val_items_all if it['suite'] == 'astro']
    val_items_IC    = [it for it in val_items_all if it['suite'] == 'IC']

    # Per-redshift IC groups
    val_items_per_z = {}
    for z_int in [8, 9, 10, 11, 12]:
        val_items_per_z[z_int] = [it for it in val_items_IC
                                  if int(round(float(it['redshift']))) == z_int]

    all_results = {}
    print(f"\n--- 'all' (n={len(val_items_all)}) ---")
    all_results['all']   = eval_group(val_items_all, model, vae, latent_mean, latent_std, sigma_data, device, tag='all')
    print(f"--- astro (n={len(val_items_astro)}) ---")
    all_results['astro'] = eval_group(val_items_astro, model, vae, latent_mean, latent_std, sigma_data, device, tag='astro')
    print(f"--- IC (n={len(val_items_IC)}) ---")
    all_results['IC']    = eval_group(val_items_IC, model, vae, latent_mean, latent_std, sigma_data, device, tag='IC')
    for z, items in val_items_per_z.items():
        if items:
            print(f"--- IC z={z} (n={len(items)}) ---")
            all_results[f'IC_z{z}'] = eval_group(items, model, vae, latent_mean, latent_std,
                                                 sigma_data, device, tag=f'IC_z{z}',
                                                 two_runs_for_diversity=False)

    # Pretty print
    print("\n" + "="*78)
    print(f"{'group':<14} {'n':>4} {'rel_MSE':>8} {'pix_corr':>9} {'PS(L/M/S)':>22} {'kurt(r/t)':>14} {'PDF_KS':>7}")
    print("-"*78)
    for k, r in all_results.items():
        if r is None: continue
        ps_str = f"{r['ps_large']:.2f}/{r['ps_mid']:.2f}/{r['ps_small']:.2f}"
        kurt_str = f"{r['kurt_recon']:.2f}/{r['kurt_true']:.2f}"
        print(f"{k:<14} {r['n']:>4d} {r['rel_mse']:>8.4f} {r['pixel_corr_mean']:>9.3f} "
              f"{ps_str:>22} {kurt_str:>14} {r['pdf_ks']:>7.3f}")

    # Diagnosis
    print("\n" + "="*78)
    print("DIAGNOSIS")
    print("="*78)
    issues = diagnose(all_results)
    for issue, sev, fix in issues:
        if sev == 'ok':
            print(f"  [OK]   {issue}")
        else:
            print(f"  [{sev:4s}] {issue}")
            if fix:
                print(f"         → fix: {fix}")

    # Save full results
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results written to {args.out}")


if __name__ == '__main__':
    main()
