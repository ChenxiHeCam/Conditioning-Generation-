# VAE3D v4 — 21cm Brightness Temperature Encoder

3D Variational Autoencoder for 21cm T21 brightness temperature patches.
Trained from scratch on `varying_IC` + `varying_astro` simulation suites
(combined 1028 training cubes, 4 patches per cube per epoch).

Checkpoint: `vae_v4_final.pt` (148 MB), stopped at epoch 74. MSE plateau
was reached at ep20; the PS-loss schedule was partially ramped (training
would have continued safely but improvement margin was diminishing).

---

## 1. Architecture

```python
from vae import VAE3D

model = VAE3D(
    in_ch=1,
    latent_ch=8,
    base_ch=128,
    ch_mults=(1,),   # single downsampling stage
)
```

- Input  shape: `(B, 1, 64, 64, 64)` — single-channel cubic patches
- Latent shape: `(B, 8, 32, 32, 32)` — 4× spatial compression, 8 channels
- 9.22 M parameters
- Encoder outputs `(mean, logvar)`. For downstream use prefer the
  deterministic `mean` over the stochastic `z = mean + std·ε`.

---

## 2. Three usage scenarios

### 2.1 Encode T21 → latent (compression / feature extraction)

```python
import torch
from vae import VAE3D

ckpt = torch.load('vae_v4_final.pt', map_location='cuda')
cfg  = ckpt['model_config']

vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
            base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).cuda()
vae.load_state_dict(ckpt['model'])
vae.eval()

# x: (B, 1, 64, 64, 64) on cuda, ALREADY normalized (see §3)
with torch.no_grad():
    mean, logvar = vae.encoder(x)        # latent: (B, 8, 32, 32, 32)
```

Compression: 262 144 input voxels → 32 768 latent dims (8×).

### 2.2 Encode + decode (reconstruction)

```python
with torch.no_grad():
    mean, _ = vae.encoder(x)
    recon = vae.decoder(mean)            # back to (B, 1, 64, 64, 64)
```

Reconstruction error MSE/var ≈ 1e-4 (≥ 99.86 % explained variance).

### 2.3 Use as the encoder for a latent diffusion model (LDM)

The latent is **not** N(0, 1) — see §4. Normalize before feeding the U-Net:

```python
mean_ch = ckpt['latent_mean'].view(1, -1, 1, 1, 1).cuda()
std_ch  = ckpt['latent_std' ].view(1, -1, 1, 1, 1).cuda()

with torch.no_grad():
    z, _ = vae.encoder(x)
z_norm = (z - mean_ch) / std_ch          # per-channel N(0, 1)

# ... train diffusion on z_norm ...

# After sampling from diffusion:
z = z_norm_sampled * std_ch + mean_ch
recon = vae.decoder(z)
```

This is the per-channel analogue of Stable Diffusion's `0.18215` scaling.

---

## 3. Input normalization (CRITICAL)

The two simulation suites have **very different brightness-temperature
distributions** and `dataset.T21Dataset` normalizes each one separately at
construction time:

| Suite          | t21_mean    | t21_std  | Source            |
|----------------|-------------|----------|-------------------|
| varying_astro  | -112.638    | 79.827   | training split     |
| varying_IC     |   10.168    |  4.825   | training split     |

For ANY new cube from these suites:

```python
patch = (raw_cube_patch - t21_mean) / t21_std   # before vae.encoder(...)
```

After decoding, invert if the original T21 values are needed:

```python
recon_raw = recon * t21_std + t21_mean
```

The simplest way for the collaborator to get this right is to **use the
shipped `dataset.py`** — it auto-computes these stats from the first 50
files of the requested split and applies them in `__getitem__`. The
provided values above match the **train split** that v4 was actually
trained on.

If the data source is unknown (mixed), normalize with whichever stats
match the file naming convention, or simply use `dataset.T21Dataset` with
`load_ic=False` and read `ds.t21_mean`, `ds.t21_std`.

---

## 4. Latent statistics (computed on train set, stored in ckpt)

Per-channel mean is near zero, but std is **not unit** (KL weight 3e-5
was deliberately small to preserve reconstruction):

| Channel | mean   | std   |
|---------|--------|-------|
| 0       | +0.119 | 2.612 |
| 1       | +0.073 | 3.191 |
| 2       | -0.004 | 1.633 |
| 3       | +0.114 | 4.039 |
| 4       | -0.015 | 1.613 |
| 5       | +0.041 | 1.805 |
| 6       | -0.091 | 2.967 |
| 7       | -0.083 | 1.626 |

Aggregate — mean 0.019, std 2.436. Available in `ckpt['latent_mean']`,
`ckpt['latent_std']` (torch tensors of shape (8,)).

---

## 5. Evaluation results (val + test, deterministic mean inference)

PS ratios use only non-empty k bins (10 of 50 low-k bins have no Fourier
modes for a 64³ box — must be masked, see caveats §7.1).

| Dataset            | n   | MSE/var | PS large | PS mid | PS small |
|--------------------|-----|---------|----------|--------|----------|
| astro val z=10     | 100 | 8.4e-5  | 0.994    | 0.996  | 0.991    |
| astro test z=10    | 100 | 9.2e-5  | 0.993    | 0.995  | 0.992    |
| IC val z=8         | 5   | 1.2e-4  | 0.985    | 0.991  | 0.992    |
| IC val z=9         | 5   | 1.2e-4  | 0.990    | 0.994  | 0.991    |
| IC val z=10        | 8   | 1.4e-4  | 0.989    | 0.995  | 0.990    |
| IC val z=11        | 5   | 4.0e-4  | 0.979    | 0.982  | 0.983    |
| IC val z=12        | 5   | 1.1e-3  | 0.965    | 0.966  | 0.968    |

All PS ratios are within `[0.965, 1.00]` across all redshifts and both
data sources. MSE/var ≤ 1.4e-3 worst case (IC test z=12).
`vae_v4_final_eval.json` contains the full per-dataset numbers and full
PS curves.

---

## 6. Training recipe (for reproducibility)

- Optimizer: AdamW, lr 1e-4, weight_decay 1e-5, cosine LR schedule
- AMP fp16 with GradScaler
- EMA decay 0.999 (`ckpt['ema']` available)
- KL weight 3e-5 with linear anneal over 100 epochs
- Staged auxiliary losses:
  - PS loss (`log1p(P_k)` MSE in 30 spherical bins) — ramp from ep 50, weight 0.1
  - Spectral FFT magnitude loss — ramp from ep 80, weight 0.05 (not fully ramped at ep 74)
- batch size 8, patches_per_cube 4, num_workers 0 (h5py thread safety)
- 5 redshifts: 8, 9, 10, 11, 12

---

## 7. Common pitfalls — please read

| # | Pitfall | Fix |
|---|---|---|
| 1 | **Feeding raw (un-normalized) T21 cubes to the encoder** | Apply `(cube - t21_mean) / t21_std` per the suite in §3 |
| 2 | Using stochastic `z` instead of deterministic `mean` for inference | Call `vae.encoder(x)` and use `mean`; do NOT use `vae.encode(x)` (it adds noise) |
| 3 | Augmenting (flip / rotation) at LDM-stage but VAE wasn't trained with augmentation | v4 saw NO axis flips or rotations during training. If you augment, either (a) skip aug, or (b) briefly finetune v4 with aug for ~50 epochs |
| 4 | Forgetting to **un-normalize the latent** after diffusion sampling | `z = z_norm * std_ch + mean_ch` before `vae.decoder(z)` |
| 5 | Computing band-averaged PS ratios without masking empty bins | `valid = ps.mean(axis=0) > 1e-20`, then mean over `ratio[valid & mask_band]` |
| 6 | Trying to encode varying_IC patches with varying_astro's t21_mean/std | Use the right per-suite normalization — they differ by an order of magnitude |

---

## 8. Files in this bundle

| File | Description |
|---|---|
| `vae_v4_final.pt`        | Checkpoint: `model`, `ema`, `latent_mean`, `latent_std`, `model_config`, `args`, `epoch` |
| `vae.py`                 | `VAE3D` model class (in_ch, latent_ch, base_ch, ch_mults configurable) |
| `dataset.py`             | `T21Dataset` loader with the same normalization v4 was trained with |
| `vae_v4_final_eval.json` | Per-dataset eval numbers (machine-readable) |
| `README_v4.md`           | This file |
| `REPORT_v4.md` / `.pdf`  | Detailed status report with per-redshift breakdown and PS curve plot |
| `ps_ratio_v4.png`        | PS ratio (recon/target) curves for all 6 (suite, z) val configurations |

Q: does the model need IC information? **No.** The VAE only encodes and decodes T21 alone. IC and astrophysical parameters are conditioning for the downstream LDM, not for the VAE.
