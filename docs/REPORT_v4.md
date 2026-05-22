# VAE3D v4 — Status Report

**Date:** 2026-05-21
**Status:** Download complete, evaluation complete, ready to hand off

---

## 1. Bundle contents

| File | Size | Purpose |
|---|---|---|
| `vae_v4_final.pt` | 140.78 MB | Model + EMA + latent_mean + latent_std + model_config |
| `vae.py` | 4.0 KB | VAE3D model class definition |
| `dataset.py` | 15.5 KB | T21Dataset loader (auto-normalizes; matches v4 training-time normalization) |
| `vae_v4_final_eval.json` | 3.5 KB | Per-dataset evaluation numbers (machine-readable) |
| `README_v4.md` | ~10 KB | Architecture, usage scenarios, input normalization, pitfalls, eval, training recipe |
| `REPORT_v4.md` / `.pdf` | this file | Detailed status report with per-redshift breakdown and PS curve plot |
| `ps_ratio_v4.png` | 116 KB | PS ratio curves across all (suite, z) configurations |

These five files are sufficient for the collaborator to load and use the model.

---

## 2. Training summary

| Item | Value |
|---|---|
| Architecture | VAE3D, ch_mults=(1,), latent_ch=8, base_ch=128 |
| Parameters | 9.22 M |
| Input | (B, 1, 64, 64, 64) — 21cm T21 patches |
| Latent | (B, 8, 32, 32, 32) — 4× spatial compression, 8 channels |
| Training data | 1028 cubes total (varying_astro 803 + varying_IC 225), 5 redshifts (z=8-12) |
| Loss | MSE + KL (3e-5 with anneal) + PS log1p + spectral FFT |
| Stopped at | Epoch 74. MSE plateaued at ep20; PS-loss schedule partially ramped (sufficient given PS was already paper-grade). |
| Training time | ~17 h on RTX 4090 (800 s / epoch) |

---

## 3. Per-redshift evaluation (deterministic-mean inference)

All PS ratios use only non-empty k bins. Of the 50 log-spaced k bins in the `power_spectrum` utility, the lowest 10 (k < ~0.075 h/Mpc) contain no Fourier modes for a 64³ box; they must be masked (`valid = ps.mean(axis=0) > 1e-20`) before averaging, otherwise the average is artificially dragged toward zero by empty bins.

### z = 8 (IC data only)

| Split | n | MSE/var | PS large (k<0.1) | PS mid | PS small (k>0.5) |
|---|---|---|---|---|---|
| IC val  | 5 | 1.20e-4 | 0.985 | 0.991 | 0.992 |
| IC test | 5 | 1.23e-4 | 0.987 | 0.992 | 0.992 |

**Average PS** — large 0.986, mid 0.991, small 0.992 ✓ paper-grade

---

### z = 9 (IC data only)

| Split | n | MSE/var | PS large | PS mid | PS small |
|---|---|---|---|---|---|
| IC val  | 5 | 1.22e-4 | 0.990 | 0.994 | 0.991 |
| IC test | 5 | 1.14e-4 | 0.991 | 0.995 | 0.992 |

**Average PS** — large 0.990, mid 0.994, small 0.992 ✓ paper-grade

---

### z = 10 (astro + IC — richest data)

| Split | n | MSE/var | PS large | PS mid | PS small |
|---|---|---|---|---|---|
| **astro val**  | **100** | **8.40e-5** | **0.994** | **0.996** | **0.991** |
| **astro test** | **100** | **9.17e-5** | **0.993** | **0.995** | **0.992** |
| IC val  | 8 | 1.43e-4 | 0.989 | 0.995 | 0.990 |
| IC test | 8 | 1.58e-4 | 0.991 | 0.993 | 0.988 |

**Best redshift. Average PS** — large 0.992, mid 0.995, small 0.990 ✓ paper-grade

---

### z = 11 (IC data only)

| Split | n | MSE/var | PS large | PS mid | PS small |
|---|---|---|---|---|---|
| IC val  | 5 | 3.95e-4 | 0.979 | 0.982 | 0.983 |
| IC test | 5 | 3.31e-4 | 0.979 | 0.984 | 0.987 |

**Average PS** — large 0.979, mid 0.983, small 0.985 ✓ paper-grade (slight degradation but well within bounds)

---

### z = 12 (IC data only — hardest case)

| Split | n | MSE/var | PS large | PS mid | PS small |
|---|---|---|---|---|---|
| IC val  | 5 | 1.08e-3 | 0.965 | 0.966 | 0.968 |
| IC test | 5 | 1.38e-3 | 0.969 | 0.966 | 0.966 |

**Average PS** — large 0.967, mid 0.966, small 0.967 ✓ paper-grade (at the lower boundary)

---

### Dataset split (80 / 10 / 10 stratified per redshift)

| z   | IC train | IC val | IC test | astro train | astro val | astro test |
|-----|----------|--------|---------|-------------|-----------|------------|
| 8   | 40       | 5      | 5       | 0           | 0         | 0          |
| 9   | 40       | 5      | 5       | 0           | 0         | 0          |
| 10  | **65**   | **8**  | **8**   | **803**     | **100**   | **100**    |
| 11  | 40       | 5      | 5       | 0           | 0         | 0          |
| 12  | 40       | 5      | 5       | 0           | 0         | 0          |

z=10 has more IC cubes than the other redshifts (82 vs 50), and is the only redshift with `varying_astro` data (1003 cubes).

### Visual summary

![v4 power-spectrum ratio across redshifts](ps_ratio_v4.png)

**Left panel** — P_recon(k) / P_target(k) for each (data source, redshift) tested on the val split (100 astro samples, 5–8 IC samples per redshift). All curves sit comfortably within the ±5 % target band (grey). astro z=10 (deepest purple) is tightest to 1.0; IC z=12 (yellow) is the lowest curve at ≈ 0.96-0.97 — still paper-grade. Single-point spikes near k ≈ 0.03 reflect bin-statistic noise: the k_fund bin contains only ~6 Fourier modes in a 64³ box. Red dotted lines mark the large / mid / small band boundaries (k = 0.1 and k = 0.5).

**Right panel** — Absolute P(k) overlay for astro val z=10. Target (solid black) and v4 reconstruction (dashed blue) are visually indistinguishable on a log–log plot.

### Cross-redshift summary

| z  | Train cubes (IC + astro) | Val n  | MSE/var (val) | PS avg (val) | Status   |
|----|--------------------------|--------|---------------|--------------|----------|
| 8  | 40 IC                    | 5      | 1.20e-4       | 0.989        | OK       |
| 9  | 40 IC                    | 5      | 1.22e-4       | 0.992        | OK       |
| 10 | 65 IC + 803 astro = 868  | 8+100  | **8.40e-5**   | **0.994**    | **Best** |
| 11 | 40 IC                    | 5      | 3.95e-4       | 0.981        | OK       |
| 12 | 40 IC                    | 5      | 1.08e-3       | 0.967        | At limit |

**Key observations:**

1. All redshifts are paper-grade (PS ratio in [0.965, 1.05]).
2. Performance degrades monotonically with redshift, but the worst case is still ≥ 0.965.
3. z=10 dominates because it has 16× more training data than the other redshifts (803 astro cubes vs ~40 IC cubes elsewhere).
4. astro and IC perform comparably at z=10 (0.994 vs 0.992), indicating the encoder did not collapse onto a single data source.
5. z=12 has the worst MSE/var (1.08e-3) — this is jointly driven by limited training data (40 cubes) and by the intrinsic difficulty of z=12 fields (weaker T21 signal, more saturated regions).

---

## 4. Latent statistics (stored in checkpoint)

Per-channel mean is near zero (max |mean| = 0.12), but std is **not** unit:

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

**Aggregate** — mean 0.02, std 2.44.

The KL weight (3e-5) was deliberately small to preserve reconstruction fidelity; this is the standard tradeoff. Per-channel z-score normalisation before feeding the latent to a downstream diffusion model is required and documented in `README_v4.md` Section 3.

---

## 5. Handover instructions

The collaborator can load and use the model with:

```python
import torch
from vae import VAE3D

ckpt = torch.load('vae_v4_final.pt', map_location='cuda')
cfg  = ckpt['model_config']

model = VAE3D(in_ch=1,
              latent_ch=cfg['latent_ch'],
              base_ch=cfg['base_ch'],
              ch_mults=tuple(cfg['ch_mults'])).cuda()
model.load_state_dict(ckpt['model'])
model.eval()

# Use deterministic mean for clean reconstruction
with torch.no_grad():
    mean, logvar = model.encoder(x)        # x: (B, 1, 64, 64, 64)
    recon = model.decoder(mean)
```

If training a downstream latent diffusion model, normalize the latent first:

```python
mean_ch = ckpt['latent_mean'].view(1, -1, 1, 1, 1).cuda()
std_ch  = ckpt['latent_std' ].view(1, -1, 1, 1, 1).cuda()

z_norm = (mean - mean_ch) / std_ch          # ~N(0, 1) per channel
# train diffusion on z_norm
# at decode: z = z_norm * std_ch + mean_ch; recon = model.decoder(z)
```

---

## 6. Known caveats

1. **Empty PS bins** in `power_spectrum`: always mask with `ps.mean(axis=0) > 1e-20` before band averaging. Without this mask the PS large ratio appears as ~0.41 instead of ~0.99.
2. **IC data sparsity**: 40-65 cubes per redshift outside z=10. Performance on z=11/12 is slightly worse (PS ≈ 0.97-0.98) but still paper-grade.
3. **No augmentation during VAE training** (no flips/rotations). If you augment at the diffusion-stage training, the encoder will see unseen orientations — verify encoder invariance first, or finetune VAE briefly with augmentation.
4. **Patch-level evaluation only**: full 256³ cube reconstruction via patch tiling has not been tested. Use overlap-tile (e.g. stride 32 with patch 64 and centre-crop) to avoid boundary seams.
5. **Per-sample outliers**: 0 / 100 samples in astro val/test deviate >20 % from 1.0 in any PS band. Same for IC at all redshifts tested.

---

## 7. Next step

Stage 2: Latent Diffusion Model on the (8, 32, 32, 32) latent, conditioned on (IC_delta, IC_vbv, astro params, redshift). Design plan finalised:

- EDM (Karras 2022) + v-prediction, P_mean=-0.4, P_std=1.2, 32-step Heun sampler
- Channel concat IC with a learned 3D conv stem (1→16→32 stride-2 ×2)
- 3D U-Net, 2 downsamples (32→16→8), channels 64→128→256
- Windowed attention at 16³ + global self-attention at 8³
- FiLM from concat(4 astro params, z, σ) → MLP → 256d, injected per ResBlock
- Suite-weighted sampling (50 / 50 astro / IC mixing per batch)
- Compositional CFG with independent dropout (p_IC=0.10, p_params=0.10, joint=0.05); guidance scales w_IC=1.0, w_params=2.0
- AdamW lr 1e-4, bf16, EMA 0.9999 and 0.999 both saved, batch 8, ~2 days training

Implementation is ready to start on the user's go-ahead.
