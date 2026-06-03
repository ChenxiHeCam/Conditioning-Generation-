"""
Finalize v4 for release:
  1. Load v4 ep74 ckpt
  2. Compute per-channel latent mean/std on TRAIN set (deterministic encoder.mean output)
  3. Save augmented ckpt with these stats
  4. Run final eval on val+test (sanity)
"""
import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from dataset import T21Dataset
from models.vae import VAE3D
from utils.power_spectrum import power_spectrum


CKPT_IN  = '/root/autodl-tmp/checkpoints/vae_v4/vae_epoch0074.pt'
CKPT_OUT = '/root/autodl-tmp/checkpoints/vae_v4/vae_v4_final.pt'
EVAL_OUT = '/root/autodl-tmp/checkpoints/vae_v4/vae_v4_final_eval.json'


def main():
    device = torch.device('cuda')

    ckpt = torch.load(CKPT_IN, map_location=device)
    a = ckpt['args']
    print(f"Loaded ep{ckpt['epoch']}  base_ch={a['base_ch']}  ch_mults={a['ch_mults']}  latent_ch={a['latent_ch']}")

    vae = VAE3D(in_ch=1, latent_ch=a['latent_ch'],
                base_ch=a['base_ch'], ch_mults=tuple(a['ch_mults'])).to(device)
    vae.load_state_dict(ckpt['model']); vae.eval()

    # --- 1) per-channel latent stats on TRAIN set ---
    print("\nComputing per-channel latent stats on train set ...")
    train_sets = []
    for root in ['/root/autodl-tmp/ASR21cm/varying_IC',
                 '/root/autodl-tmp/ASR21cm/varying_astro']:
        try:
            train_sets.append(T21Dataset(root, 64, redshifts=[8,9,10,11,12],
                                         split='train', load_ic=False))
        except Exception as e:
            print(f"  skip {root}: {e}")
    ds_train = ConcatDataset(train_sets)
    print(f"  train size: {len(ds_train)}")
    loader = DataLoader(ds_train, batch_size=8, shuffle=False, num_workers=0)

    # Accumulate sums for mean/std per channel
    C = a['latent_ch']
    sum_mu  = torch.zeros(C, device=device)
    sum_mu2 = torch.zeros(C, device=device)
    n_elem  = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch['patch'].to(device)
            mu, _ = vae.encoder(x)         # (B, C, D, H, W)
            mu_flat = mu.transpose(0, 1).reshape(C, -1)   # (C, N*spatial)
            sum_mu  += mu_flat.sum(dim=1)
            sum_mu2 += (mu_flat ** 2).sum(dim=1)
            n_elem  += mu_flat.shape[1]
            if (i + 1) % 50 == 0:
                print(f"  batch {i+1}/{len(loader)}")
    ch_mean = (sum_mu / n_elem).cpu()
    ch_var  = (sum_mu2 / n_elem - ch_mean.to(device)**2).clamp_min(0).cpu()
    ch_std  = ch_var.sqrt()
    print("\nPer-channel mean:", ch_mean.tolist())
    print("Per-channel std :", ch_std.tolist())
    print(f"Overall  mean: {ch_mean.mean():.4f}   std: {ch_std.mean():.4f}")

    # --- 2) Save augmented ckpt ---
    out_ckpt = dict(ckpt)
    out_ckpt['latent_mean'] = ch_mean
    out_ckpt['latent_std']  = ch_std
    out_ckpt['model_config'] = dict(
        in_ch=1,
        latent_ch=a['latent_ch'],
        base_ch=a['base_ch'],
        ch_mults=list(a['ch_mults']),
        downsamples=len(a['ch_mults']),
        latent_shape=[a['latent_ch'], 32, 32, 32],
    )
    torch.save(out_ckpt, CKPT_OUT)
    print(f"\nSaved augmented ckpt to {CKPT_OUT}  ({os.path.getsize(CKPT_OUT)/1e6:.1f} MB)")

    # --- 3) Final eval ---
    print("\nRunning final eval ...")
    eval_results = {}
    eval_combos = [('astro', 10, 'val'), ('astro', 10, 'test')]
    for z in [8, 9, 10, 11, 12]:
        eval_combos += [('IC', z, 'val'), ('IC', z, 'test')]

    for data_tag, z, split in eval_combos:
        data_path = ('/root/autodl-tmp/ASR21cm/varying_astro'
                     if data_tag == 'astro'
                     else '/root/autodl-tmp/ASR21cm/varying_IC')
        try:
            ds = T21Dataset(data_path, 64, redshifts=[z], split=split, load_ic=False)
        except Exception as e:
            continue
        if len(ds) == 0:
            continue
        ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        O, R = [], []
        with torch.no_grad():
            for batch in ld:
                x = batch['patch'].to(device)
                mu, _ = vae.encoder(x); r = vae.decoder(mu)
                O.append(x.cpu()); R.append(r.cpu())
        X = torch.cat(O).squeeze(1).numpy()
        R = torch.cat(R).squeeze(1).numpy()
        mse = float(((R - X) ** 2).mean())
        var = float(X.var())
        k, ps_o = power_spectrum(X)
        _, ps_r = power_spectrum(R)
        valid = ps_o.mean(0) > 1e-20
        ratio = ps_r.mean(0) / np.maximum(ps_o.mean(0), 1e-30)
        m_lo, m_mid, m_hi = (k<0.1)&valid, (k>=0.1)&(k<0.5)&valid, (k>=0.5)&valid
        key = f"{data_tag}-{split}-z{z}"
        eval_results[key] = dict(
            n=int(X.shape[0]),
            mse=mse, mse_over_var=mse/var,
            ps_large=float(ratio[m_lo].mean()) if m_lo.any() else None,
            ps_mid=float(ratio[m_mid].mean()) if m_mid.any() else None,
            ps_small=float(ratio[m_hi].mean()) if m_hi.any() else None,
        )
        print(f"  {key} n={X.shape[0]}: MSE/var={mse/var:.4f}  "
              f"PS=({eval_results[key]['ps_large']:.3f}, "
              f"{eval_results[key]['ps_mid']:.3f}, "
              f"{eval_results[key]['ps_small']:.3f})")

    # Save eval JSON
    summary = dict(
        ckpt='vae_v4_final.pt',
        epoch=int(ckpt['epoch']),
        model_config=out_ckpt['model_config'],
        latent_mean=ch_mean.tolist(),
        latent_std=ch_std.tolist(),
        per_dataset_eval=eval_results,
    )
    with open(EVAL_OUT, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nEval summary -> {EVAL_OUT}")


if __name__ == '__main__':
    main()
