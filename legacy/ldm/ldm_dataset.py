"""
Dataset wrapper for Stage 2 LDM training.

Returns aligned (T21, IC_delta, IC_vbv, params, redshift) tuples with:
  - the SAME random crop applied to T21 and the IC channels (already done by T21Dataset)
  - random axis flip + permutation applied consistently across all spatial channels
  - per-suite normalization already handled by the underlying T21Dataset

Combines varying_IC + varying_astro into a single ConcatDataset; expose
suite-weighted sampling via `make_balanced_sampler` so each batch has roughly
50/50 mix instead of being dominated by the 4×-larger astro set.
"""
import re
import os
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler

from dataset import T21Dataset


_FLIPS = list(itertools.product([False, True], repeat=3))   # 8
_PERMS = list(itertools.permutations([0, 1, 2]))            # 6


def random_orient(rng):
    """Pick (flip3, perm3) uniformly from the 48 cube symmetries."""
    return _FLIPS[rng.integers(0, 8)], _PERMS[rng.integers(0, 6)]


def apply_orient(x, flip, perm):
    """x: (C, D, H, W) torch tensor. flip: 3-tuple bool. perm: 3-permutation."""
    # axis permutation: (C, D, H, W) -> permute spatial dims
    x = x.permute(0, 1 + perm[0], 1 + perm[1], 1 + perm[2])
    flip_dims = [1 + i for i, f in enumerate(flip) if f]
    if flip_dims:
        x = torch.flip(x, dims=flip_dims)
    return x


class LDMDataset(Dataset):
    """
    Wraps a T21Dataset (load_ic=True) and augments each sample.
    The base dataset must return:
        {'patch': (1,D,H,W), 'ic_delta': (1,D,H,W), 'ic_vbv': (1,D,H,W),
         'params': (5,)}
    We add 'redshift' (scalar tensor) and apply augmentation if requested.
    Also tags samples with their 'suite' name for weighted sampling.
    """
    def __init__(self, base: T21Dataset, suite_tag: str,
                 augment: bool = True, seed: int = 42):
        self.base = base
        self.suite_tag = suite_tag
        self.augment = augment
        self._rng = np.random.default_rng(seed)
        self._z_per_file = [base._redshift_of(f) for f in base.files]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        file_idx = idx // self.base.patches_per_cube
        z = self._z_per_file[file_idx] or 10

        if self.augment:
            flip, perm = random_orient(self._rng)
            item['patch']    = apply_orient(item['patch'],    flip, perm)
            item['ic_delta'] = apply_orient(item['ic_delta'], flip, perm)
            item['ic_vbv']   = apply_orient(item['ic_vbv'],   flip, perm)

        item['redshift'] = torch.tensor(float(z), dtype=torch.float32)
        item['suite']    = self.suite_tag
        return item


def build_train_dataset(
    data_root_ic='/root/autodl-tmp/ASR21cm/varying_IC',
    data_root_astro='/root/autodl-tmp/ASR21cm/varying_astro',
    patch_size=64,
    redshifts=(8, 9, 10, 11, 12),
    patches_per_cube=4,
    augment=True,
    split='train',
):
    """
    Returns (concat_ds, weights) where:
      concat_ds = ConcatDataset of [LDMDataset(IC), LDMDataset(astro)]
      weights   = per-sample probabilities for WeightedRandomSampler
                  (uniform within each suite, balanced across suites)
    """
    sets = []
    sizes = []
    suite_tags = []

    if data_root_ic and os.path.isdir(data_root_ic):
        ic_base = T21Dataset(data_root_ic, patch_size, redshifts=list(redshifts),
                             split=split, patches_per_cube=patches_per_cube,
                             load_ic=True)
        sets.append(LDMDataset(ic_base, 'IC', augment=augment))
        sizes.append(len(sets[-1]))
        suite_tags.append('IC')

    if data_root_astro and os.path.isdir(data_root_astro):
        astro_base = T21Dataset(data_root_astro, patch_size, redshifts=list(redshifts),
                                split=split, patches_per_cube=patches_per_cube,
                                load_ic=True)
        sets.append(LDMDataset(astro_base, 'astro', augment=augment))
        sizes.append(len(sets[-1]))
        suite_tags.append('astro')

    if not sets:
        raise RuntimeError('No valid data root found')

    concat = ConcatDataset(sets) if len(sets) > 1 else sets[0]

    # Build per-sample weights so that each suite is hit equally often
    n_suites = len(sets)
    weights = np.zeros(len(concat), dtype=np.float64)
    offset = 0
    for size, tag in zip(sizes, suite_tags):
        # weight = (1 / size) * (1 / n_suites) so each suite contributes equal mass
        weights[offset:offset + size] = 1.0 / (size * n_suites)
        offset += size
    weights = torch.from_numpy(weights / weights.sum())

    return concat, weights


def make_balanced_sampler(weights, num_samples=None):
    """num_samples per epoch; if None, uses len(weights)."""
    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples or len(weights),
        replacement=True,
    )


if __name__ == '__main__':
    # Smoke test
    ds, w = build_train_dataset(patches_per_cube=2)
    print(f"Concat size: {len(ds)}, weights sum: {w.sum():.4f}")
    item = ds[0]
    for k, v in item.items():
        if torch.is_tensor(v):
            print(f"  {k:10s} {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:10s} {v}")

    # Sample a small batch via the balanced sampler and check suite mix
    sampler = make_balanced_sampler(w, num_samples=256)
    suites = [ds[i]['suite'] for i in sampler]
    from collections import Counter
    print('256-sample suite distribution:', Counter(suites))
