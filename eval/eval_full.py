"""
Extended eval: comprehensive metrics + latent distribution + outliers + full PS curves.

Adds to eval_comprehensive.py:
  - Full per-bin PS ratio curves saved per dataset
  - Per-sample MSE + PS for ALL conditions (not just astro-val-z10)
  - Latent distribution stats: per-channel mean/std, KL to N(0,1), dead-channel detection,
    logvar percentiles, optional latent sampling test
  - Outlier identification: worst K patches per dataset, with filenames

Usage:
  python eval_full.py --v4_ckpt PATH --out PATH
"""
import os, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import T21Dataset
from models.vae import VAE3D
from utils.power_spectrum import power_spectrum


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--v3_ckpt',     default='/root/autodl-tmp/checkpoints/vae_v3/vae_v3_epoch0059.pt')
    p.add_argument('--reshi_ckpt',  default='/root/autodl-tmp/checkpoints/vae_residual/residual_epoch0019.pt')
    p.add_argument('--v4_ckpt',     required=True)
    p.add_argument('--out',         default='/root/autodl-tmp/eval_full.json')
    p.add_argument('--data_ic',     default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_astro',  default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--n_outliers',  type=int, default=5, help='top-K worst patches to log')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def per_sample_metrics(X, R):
    """Return arrays of per-sample (mse, ps_large, ps_mid, ps_small).
    Uses the valid-bin mask per sample."""
    n = X.shape[0]
    mse = np.zeros(n)
    pl = np.full(n, np.nan)
    pm = np.full(n, np.nan)
    ph = np.full(n, np.nan)
    for i in range(n):
        mse[i] = ((R[i] - X[i]) ** 2).mean()
        k, p_o = power_spectrum(X[i:i+1])
        _, p_r = power_spectrum(R[i:i+1])
        v = p_o[0] > 1e-20
        rat = p_r[0] / np.maximum(p_o[0], 1e-30)
        m_lo  = (k < 0.1)  & v
        m_mid = (k >= 0.1) & (k < 0.5) & v
        m_hi  = (k >= 0.5) & v
        if m_lo.any():  pl[i] = rat[m_lo].mean()
        if m_mid.any(): pm[i] = rat[m_mid].mean()
        if m_hi.any():  ph[i] = rat[m_hi].mean()
    return mse, pl, pm, ph


def aggregated_ps_curve(X, R):
    """Full per-bin PS ratio averaged over batch."""
    k, ps_o = power_spectrum(X)
    _, ps_r = power_spectrum(R)
    valid = ps_o.mean(0) > 1e-20
    ratio = ps_r.mean(0) / np.maximum(ps_o.mean(0), 1e-30)
    return k.tolist(), ratio.tolist(), valid.tolist()


def kl_to_n01(mean, logvar):
    """KL(N(mu, var) || N(0, 1)) per element."""
    return 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)


# ---------------------------------------------------------------------------
# Reconstruction / encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_with_latent(model, ds, device, batch_size=8):
    """Run encode->decode, return X, R, and latent stats (mean, logvar per sample).
    Works with raw VAE3D models; for composite (v3+res_hi) use reconstruct_composite.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    O, R, MU, LV = [], [], [], []
    for batch in loader:
        x = batch['patch'].to(device)
        mu, lv = model.encoder(x)
        z = mu  # use deterministic mu (no noise) for eval
        r = model.decoder(z)
        O.append(x.cpu()); R.append(r.cpu())
        MU.append(mu.cpu()); LV.append(lv.cpu())
    X  = torch.cat(O).squeeze(1).numpy()
    R  = torch.cat(R).squeeze(1).numpy()
    MU = torch.cat(MU)              # (N, C, D, H, W)
    LV = torch.cat(LV)
    return X, R, MU, LV


@torch.no_grad()
def reconstruct_composite(model_fn, ds, device, batch_size=8):
    """Composite model (no clean latent access). Returns X, R only."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    O, R = [], []
    for batch in loader:
        x = batch['patch'].to(device)
        r = model_fn(x)
        O.append(x.cpu()); R.append(r.cpu())
    X = torch.cat(O).squeeze(1).numpy()
    R = torch.cat(R).squeeze(1).numpy()
    return X, R


@torch.no_grad()
def sample_decode(model, ds_for_shape, device, n_samples=8):
    """Sample z ~ N(0,1), decode, compute PS — sanity for downstream LDM."""
    # Infer latent shape from one forward
    item = ds_for_shape[0]
    x = item['patch'].unsqueeze(0).to(device)
    mu, _ = model.encoder(x)
    latent_shape = (n_samples,) + tuple(mu.shape[1:])
    z = torch.randn(latent_shape, device=device)
    r = model.decoder(z).cpu().squeeze(1).numpy()
    # Compare PS of decoded samples vs typical data PS
    X_ref = []
    for i in range(min(n_samples, len(ds_for_shape))):
        X_ref.append(ds_for_shape[i]['patch'].numpy())
    X_ref = np.stack(X_ref).squeeze(1)
    k, ps_ref = power_spectrum(X_ref)
    _, ps_dec = power_spectrum(r)
    valid = ps_ref.mean(0) > 1e-20
    ratio = ps_dec.mean(0) / np.maximum(ps_ref.mean(0), 1e-30)
    return dict(
        k=k.tolist(),
        ratio=ratio.tolist(),
        valid=valid.tolist(),
        decoded_std=float(r.std()),
        ref_std=float(X_ref.std()),
        # bin averages
        ps_large=float(ratio[(k < 0.1)  & valid].mean()) if ((k < 0.1)  & valid).any() else None,
        ps_mid  =float(ratio[(k >= 0.1) & (k < 0.5) & valid].mean()) if ((k >= 0.1) & (k < 0.5) & valid).any() else None,
        ps_small=float(ratio[(k >= 0.5) & valid].mean()) if ((k >= 0.5) & valid).any() else None,
    )


def latent_stats(MU, LV):
    """
    Latent distribution analysis. Inputs are tensors (N, C, D, H, W).
    Returns per-channel stats and overall Gaussian-ness.
    """
    N, C = MU.shape[:2]
    mu_flat = MU.view(N, C, -1)           # (N, C, voxels)
    lv_flat = LV.view(N, C, -1)
    # Per-channel stats: mean over (N, voxels), std over (N, voxels)
    ch_mean = mu_flat.mean(dim=(0, 2)).numpy()    # (C,)
    ch_std  = mu_flat.std(dim=(0, 2)).numpy()
    # KL per channel (averaged over batch+spatial)
    kl_per_dim = kl_to_n01(MU, LV)        # (N, C, D, H, W)
    kl_per_ch  = kl_per_dim.mean(dim=(0, 2, 3, 4)).numpy()   # (C,)
    # logvar distribution
    lv_p = lv_flat.flatten().numpy()
    lv_pct = np.percentile(lv_p, [5, 25, 50, 75, 95]).tolist()
    # Dead channels: KL < 0.01 (effectively unused)
    dead = (kl_per_ch < 0.01).sum()
    return dict(
        n_channels=int(C),
        per_channel_mean=ch_mean.tolist(),
        per_channel_std=ch_std.tolist(),
        per_channel_kl=kl_per_ch.tolist(),
        kl_total=float(kl_per_ch.sum()),
        dead_channels=int(dead),
        logvar_pct=lv_pct,
        # Overall vs N(0,1)
        mean_abs_mean=float(np.abs(ch_mean).mean()),
        mean_std=float(ch_std.mean()),
    )


def find_outliers(per_sample_mse, per_sample_pl, per_sample_ph, file_list, n_top=5):
    """Top-K worst patches by MSE and by |PS_ratio - 1|."""
    n = len(per_sample_mse)
    out = {}
    # worst by MSE
    idx_mse = np.argsort(-per_sample_mse)[:n_top]
    out['worst_mse'] = [
        dict(idx=int(i), mse=float(per_sample_mse[i]),
             ps_large=float(per_sample_pl[i]) if not np.isnan(per_sample_pl[i]) else None,
             ps_small=float(per_sample_ph[i]) if not np.isnan(per_sample_ph[i]) else None,
             file=file_list[i] if i < len(file_list) else None)
        for i in idx_mse
    ]
    # worst by PS_large deviation
    dev_lo = np.abs(per_sample_pl - 1.0)
    dev_lo[np.isnan(dev_lo)] = 0
    idx_lo = np.argsort(-dev_lo)[:n_top]
    out['worst_ps_large'] = [
        dict(idx=int(i), ps_large=float(per_sample_pl[i]) if not np.isnan(per_sample_pl[i]) else None,
             mse=float(per_sample_mse[i]), file=file_list[i] if i < len(file_list) else None)
        for i in idx_lo
    ]
    # worst by PS_small deviation
    dev_hi = np.abs(per_sample_ph - 1.0)
    dev_hi[np.isnan(dev_hi)] = 0
    idx_hi = np.argsort(-dev_hi)[:n_top]
    out['worst_ps_small'] = [
        dict(idx=int(i), ps_small=float(per_sample_ph[i]) if not np.isnan(per_sample_ph[i]) else None,
             mse=float(per_sample_mse[i]), file=file_list[i] if i < len(file_list) else None)
        for i in idx_hi
    ]
    return out


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def eval_dataset(model_kind, model_or_fn, ds, device, args, dataset_key):
    """model_kind in ['raw', 'composite']. Returns one block of results."""
    if model_kind == 'raw':
        X, R, MU, LV = reconstruct_with_latent(model_or_fn, ds, device)
    else:
        X, R = reconstruct_composite(model_or_fn, ds, device)
        MU = LV = None

    out = {}
    out['n'] = int(X.shape[0])
    out['mse'] = float(((R - X) ** 2).mean())
    out['var'] = float(X.var())
    out['mse_over_var'] = out['mse'] / out['var']

    # Aggregated PS curve
    k, ratio, valid = aggregated_ps_curve(X, R)
    out['ps_curve'] = dict(k=k, ratio=ratio, valid=valid)
    # 3-bin aggregates
    karr = np.array(k); rarr = np.array(ratio); varr = np.array(valid)
    for name, mask in [('large', (karr<0.1)&varr),
                       ('mid',   (karr>=0.1)&(karr<0.5)&varr),
                       ('small', (karr>=0.5)&varr)]:
        out[f'ps_{name}'] = float(rarr[mask].mean()) if mask.any() else None

    # Per-sample metrics
    pm_mse, pl, pm, ph = per_sample_metrics(X, R)
    out['per_sample'] = dict(
        mse_mean=float(pm_mse.mean()), mse_std=float(pm_mse.std()),
        ps_large_mean=float(np.nanmean(pl)), ps_large_std=float(np.nanstd(pl)),
        ps_mid_mean  =float(np.nanmean(pm)), ps_mid_std  =float(np.nanstd(pm)),
        ps_small_mean=float(np.nanmean(ph)), ps_small_std=float(np.nanstd(ph)),
        # Counts of outliers (>20% off 1.0)
        n_outliers_large=int(np.sum(np.abs(pl - 1) > 0.2)),
        n_outliers_small=int(np.sum(np.abs(ph - 1) > 0.2)),
    )

    # Outlier identification (only if file list available)
    out['outliers'] = find_outliers(pm_mse, pl, ph, ds.files, n_top=args.n_outliers)

    # Latent stats (raw VAE only)
    if MU is not None:
        out['latent'] = latent_stats(MU, LV)

    return out


def main():
    args = get_args()
    device = torch.device('cuda')

    # Load v3 + res_hi (composite)
    v3 = VAE3D(in_ch=1, latent_ch=8, base_ch=64, ch_mults=(1, 2)).to(device)
    v3.load_state_dict(torch.load(args.v3_ckpt, map_location=device)['model']); v3.eval()
    rh = VAE3D(in_ch=1, latent_ch=4, base_ch=32, ch_mults=(1,)).to(device)
    rh.load_state_dict(torch.load(args.reshi_ckpt, map_location=device)['model']); rh.eval()
    def v3_reshi_fn(x):
        xl, _, _ = v3(x); rh_hat, _, _ = rh(x - xl); return xl + rh_hat

    # Load v4 (raw)
    v4_ckpt = torch.load(args.v4_ckpt, map_location=device)
    a = v4_ckpt.get('args', {})
    v4 = VAE3D(in_ch=1, latent_ch=a.get('latent_ch', 8),
               base_ch=a.get('base_ch', 128), ch_mults=tuple(a.get('ch_mults', [1]))).to(device)
    v4.load_state_dict(v4_ckpt['model']); v4.eval()
    print(f"v4 loaded: ep{v4_ckpt.get('epoch')}, base_ch={a.get('base_ch')}, "
          f"ch_mults={a.get('ch_mults')}, latent_ch={a.get('latent_ch')}")

    # Datasets to evaluate (skip combos with no data)
    eval_combos = []
    for split in ['val', 'test']:
        eval_combos.append(('astro', 10, split))
        for z in [8, 9, 10, 11, 12]:
            eval_combos.append(('IC', z, split))

    all_results = {'v3+res_hi': {}, 'v4': {}}

    for data_tag, z, split in eval_combos:
        data_path = args.data_astro if data_tag == 'astro' else args.data_ic
        try:
            ds = T21Dataset(data_path, 64, redshifts=[z], split=split)
        except Exception as e:
            print(f"  [skip] {data_tag}-{split}-z{z}: {e}")
            continue
        if len(ds) == 0:
            continue
        key = f"{data_tag}-{split}-z{z}"

        def fmt(v): return f"{v:.3f}" if v is not None else "na"

        # v3 + res_hi
        try:
            r = eval_dataset('composite', v3_reshi_fn, ds, device, args, key)
            all_results['v3+res_hi'][key] = r
            print(f"  [v3+res_hi] {key} n={r['n']}: MSE/var={r['mse_over_var']:.4f}  "
                  f"PS=({fmt(r['ps_large'])}, {fmt(r['ps_mid'])}, {fmt(r['ps_small'])})  "
                  f"outliers=({r['per_sample']['n_outliers_large']}, {r['per_sample']['n_outliers_small']})")
        except Exception as e:
            print(f"  [v3+res_hi] {key} failed: {e}")

        # v4 (raw — also returns latent)
        try:
            r = eval_dataset('raw', v4, ds, device, args, key)
            all_results['v4'][key] = r
            lat = r['latent']
            print(f"  [v4] {key} n={r['n']}: MSE/var={r['mse_over_var']:.4f}  "
                  f"PS=({fmt(r['ps_large'])}, {fmt(r['ps_mid'])}, {fmt(r['ps_small'])})  "
                  f"outliers=({r['per_sample']['n_outliers_large']}, {r['per_sample']['n_outliers_small']})  "
                  f"dead_ch={lat['dead_channels']}/{lat['n_channels']} "
                  f"|mu|_avg={lat['mean_abs_mean']:.3f} std_avg={lat['mean_std']:.3f}")
        except Exception as e:
            print(f"  [v4] {key} failed: {e}")

    # Latent sampling sanity (v4 only) — decode z~N(0,1)
    print('\n=== v4: sample-and-decode (z~N(0,1)) ===')
    ds_astro = T21Dataset(args.data_astro, 64, redshifts=[10], split='val')
    try:
        smp = sample_decode(v4, ds_astro, device, n_samples=16)
        print(f"  decoded std={smp['decoded_std']:.4f}  ref std={smp['ref_std']:.4f}")
        print(f"  PS ratio vs ref: large={smp['ps_large']:.3f} mid={smp['ps_mid']:.3f} small={smp['ps_small']:.3f}")
        all_results['v4_sample_decode'] = smp
    except Exception as e:
        print(f"  sample_decode failed: {e}")

    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results written to {args.out}")


if __name__ == '__main__':
    main()
