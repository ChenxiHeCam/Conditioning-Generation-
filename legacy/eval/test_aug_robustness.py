"""Test whether v4 encoder/decoder generalize to flip + 90° rotation augmentations.
   v4 was trained without any augmentation, so we need to verify before relying on
   the 48x augmentation factor (3 flips x 24 rotations / 2 for double-counting)
   during Stage 2 LDM training.
"""
import os
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import T21Dataset
from models.vae import VAE3D
from utils.power_spectrum import power_spectrum


CKPT = '/root/autodl-tmp/checkpoints/vae_v4/vae_v4_final.pt'
N_SAMPLES = 50


def all_axis_flip_combos():
    """Returns all 8 sign-flip combos (no flip, x-flip, y-flip, z-flip, xy, xz, yz, xyz)."""
    return [tuple(c) for c in itertools.product([False, True], repeat=3)]


def axis_permutations():
    """24 unique rotations expressed as axis permutations (with signs).
    For simplicity, return the 6 permutations of (x,y,z) without sign flips,
    then combine with all 8 flips → 48 distinct orientations
    (which is the full symmetry group of the cube, ×2 for orientation).
    """
    return list(itertools.permutations([0, 1, 2]))


def aug_patch(x, flip_xyz, perm):
    """x: (B, 1, D, H, W) tensor. flip_xyz: 3-tuple of bool. perm: 3-permutation."""
    out = x
    # axis permutation (interpret perm as new ordering of spatial dims after (B,C))
    out = out.permute(0, 1, 2 + perm[0], 2 + perm[1], 2 + perm[2])
    # flips
    flip_dims = [2 + i for i, f in enumerate(flip_xyz) if f]
    if flip_dims:
        out = torch.flip(out, dims=flip_dims)
    return out


def metrics(X, R):
    mse = float(((R - X) ** 2).mean())
    var = float(X.var())
    k, ps_o = power_spectrum(X)
    _, ps_r = power_spectrum(R)
    valid = ps_o.mean(0) > 1e-20
    ratio = ps_r.mean(0) / np.maximum(ps_o.mean(0), 1e-30)
    return dict(
        mse_over_var=mse/var if var > 0 else float('nan'),
        ps_large=float(ratio[(k<0.1)&valid].mean()),
        ps_mid  =float(ratio[(k>=0.1)&(k<0.5)&valid].mean()),
        ps_small=float(ratio[(k>=0.5)&valid].mean()),
    )


@torch.no_grad()
def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device)
    cfg = ckpt['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'], base_ch=cfg['base_ch'],
                ch_mults=tuple(cfg['ch_mults'])).to(device)
    vae.load_state_dict(ckpt['model']); vae.eval()

    # Pull samples from astro val (largest, cleanest)
    ds = T21Dataset('/root/autodl-tmp/ASR21cm/varying_astro', 64,
                    redshifts=[10], split='val', load_ic=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    X = []
    for batch in loader:
        X.append(batch['patch'])
        if sum(b.shape[0] for b in X) >= N_SAMPLES: break
    X = torch.cat(X)[:N_SAMPLES].to(device)
    print(f"Using {X.shape[0]} astro val z=10 patches as the test set")

    # Baseline: no augmentation
    mu0, _ = vae.encoder(X)
    R0 = vae.decoder(mu0)
    base = metrics(X.squeeze(1).cpu().numpy(), R0.squeeze(1).cpu().numpy())
    print(f"\nbaseline                            MSE/var={base['mse_over_var']:.5f}  "
          f"PS=({base['ps_large']:.3f}, {base['ps_mid']:.3f}, {base['ps_small']:.3f})")

    # Test specific augmentations
    flips = all_axis_flip_combos()
    perms = axis_permutations()

    # Per-flip statistics (8 cases)
    print("\n--- 8 axis flips, identity rotation ---")
    flip_results = []
    for f in flips:
        Xa = aug_patch(X, f, (0,1,2))
        mu_a, _ = vae.encoder(Xa)
        Ra = vae.decoder(mu_a)
        m = metrics(Xa.squeeze(1).cpu().numpy(), Ra.squeeze(1).cpu().numpy())
        flip_results.append(m)
        tag = 'flip=' + ''.join('xyz'[i] if f[i] else '-' for i in range(3))
        print(f"  {tag:18s}  MSE/var={m['mse_over_var']:.5f}  "
              f"PS=({m['ps_large']:.3f}, {m['ps_mid']:.3f}, {m['ps_small']:.3f})")

    # Per-rotation statistics (6 unique permutations)
    print("\n--- 6 axis permutations, no flip ---")
    perm_results = []
    for p in perms:
        Xa = aug_patch(X, (False,False,False), p)
        mu_a, _ = vae.encoder(Xa)
        Ra = vae.decoder(mu_a)
        m = metrics(Xa.squeeze(1).cpu().numpy(), Ra.squeeze(1).cpu().numpy())
        perm_results.append(m)
        print(f"  perm{p}  MSE/var={m['mse_over_var']:.5f}  "
              f"PS=({m['ps_large']:.3f}, {m['ps_mid']:.3f}, {m['ps_small']:.3f})")

    # Random-sampled combos (representative of LDM training-time augmentation)
    print("\n--- 20 random (flip + permutation) combos ---")
    rng = np.random.default_rng(0)
    random_combos = [(rng.integers(0, 8), rng.integers(0, 6)) for _ in range(20)]
    mse_vals, ps_lg, ps_md, ps_sm = [], [], [], []
    for fi, pi in random_combos:
        Xa = aug_patch(X, flips[fi], perms[pi])
        mu_a, _ = vae.encoder(Xa)
        Ra = vae.decoder(mu_a)
        m = metrics(Xa.squeeze(1).cpu().numpy(), Ra.squeeze(1).cpu().numpy())
        mse_vals.append(m['mse_over_var'])
        ps_lg.append(m['ps_large']); ps_md.append(m['ps_mid']); ps_sm.append(m['ps_small'])

    print(f"\n  MSE/var:  mean={np.mean(mse_vals):.5f}  std={np.std(mse_vals):.5f}  "
          f"min={min(mse_vals):.5f}  max={max(mse_vals):.5f}")
    print(f"  PS large: mean={np.mean(ps_lg):.4f}  std={np.std(ps_lg):.4f}")
    print(f"  PS mid:   mean={np.mean(ps_md):.4f}  std={np.std(ps_md):.4f}")
    print(f"  PS small: mean={np.mean(ps_sm):.4f}  std={np.std(ps_sm):.4f}")

    # Decision
    base_mse = base['mse_over_var']
    max_aug_mse = max(mse_vals)
    base_ps = min(base['ps_large'], base['ps_mid'], base['ps_small'])
    min_aug_ps = min(min(ps_lg), min(ps_md), min(ps_sm))
    print(f"\n=== verdict ===")
    print(f"baseline MSE/var:    {base_mse:.5f}")
    print(f"worst-aug MSE/var:   {max_aug_mse:.5f}  ({max_aug_mse/base_mse:.1f}x baseline)")
    print(f"baseline worst PS:   {base_ps:.4f}")
    print(f"worst-aug worst PS:  {min_aug_ps:.4f}")

    if max_aug_mse < 2.0 * base_mse and min_aug_ps > 0.95:
        print("VERDICT: augmentation safe — use 48x aug at LDM training")
    elif max_aug_mse < 5.0 * base_mse and min_aug_ps > 0.90:
        print("VERDICT: augmentation acceptable — use 8x flip-only (skip rotations)")
    else:
        print("VERDICT: augmentation breaks the encoder — finetune VAE with aug for ~50 ep, "
              "or train LDM without augmentation")


if __name__ == '__main__':
    main()
