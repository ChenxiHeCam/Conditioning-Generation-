# Conditioning-Generation: 21cm Brightness-Temperature Emulator

3D latent-diffusion emulator for the 21cm brightness-temperature field, conditioned on
cosmological initial conditions (IC) and astrophysical parameters.

The pipeline has two stages:

1. **Stage 1 — VAE3D** encodes 64³ T21 patches into an (8, 32, 32, 32) latent.
   Trained on the `varying_astro` + `varying_IC` simulation suites (1028 training cubes).
2. **Stage 2 — Latent Diffusion Model (LDM)** is a 3D U-Net denoiser operating on the
   normalised latent, conditioned on IC density + velocity, four astrophysical parameters
   (`MyStar_II`, `MyVc`, `MyFX`, `DelayParam`), redshift, and EDM noise level σ.

---

## Headline results

| Stage | Group | n | Metric | Value |
|-------|-------|---|--------|-------|
| **VAE v4** | astro test z=10 | 100 | rel_MSE | **9.2e-5** |
|            | astro test z=10 | 100 | PS (large / mid / small) | **0.993 / 0.995 / 0.992** |
|            | IC test z=8–12  | 5–8 per z | rel_MSE | 1.2e-4 to 1.4e-3 |
|            | IC test z=8–12  | 5–8 per z | PS (all bands) | 0.965 to 0.995 |
| **LDM v1** (ep179) | astro test z=10 | 100 | rel_MSE | **0.054** |
|                    | astro test z=10 | 100 | PS (large / mid / small) | 0.83 / 0.83 / 0.80 |
|                    | astro test z=10 | 100 | pixel correlation | 0.91 |
|                    | astro test z=10 | 100 | kurt (recon / true) | 1.33 / 2.05 |

- VAE Stage 1 is **paper-grade** across all redshifts and suites (PS in [0.965, 1.00]).
- LDM Stage 2 is **strong on astro z=10 but underestimates PS by ~17 %**. IC suite is
  weaker because the training data has no joint (IC × params) variation.
- Per-sample LDM error is worst at high X-ray flux and low circular velocity (the
  hardest physical regimes); see `results/F_LDM_astro_corner.png`.

Full per-(suite, redshift, split) numbers and figure index live in
[`results/RESULTS.md`](results/RESULTS.md).

---

## Repository layout

```
vae/                       Stage 1 — VAE
├── vae.py                 VAE3D model definition (parametric ch_mults, latent_ch)
├── dataset.py             T21Dataset loader (auto-detects varying_astro / varying_IC)
├── train_vae.py           Training script with KL anneal + PS + spectral losses
├── train_vae_v3_finetune.py   v3 finetune attempt (k-weighted PS + free-bits KL)
├── train_residual.py      Residual high-k VAE (post-hoc small-scale fix)
├── train_residual_lo.py   Residual low-k VAE (failed experiment, kept for reference)
├── evaluate_residual.py   Evaluation for v3 + residual VAEs
├── evaluate_moe.py        Three-stage MoE evaluation (v3 + res_hi + res_lo)
├── finalize_v4.py         Compute latent stats and produce final ckpt
├── vae_visual_eval.py     Best/worst test reconstructions + PDF histograms
└── README_v4.md           VAE v4 model card (recipe + eval results)

ldm/                       Stage 2 — Latent Diffusion Model
├── ldm_unet.py            3D U-Net denoiser (FiLM, windowed + global attn, learned IC stem)
├── ldm_dataset.py         Aligned (T21, IC, params, z) dataset with augmentation
├── train_ldm.py           EDM training loop with CFG dropout, EMA, sigma_data calibration
├── train_ldm_v2_finetune.py   v2 with moment-matching + latent spectral losses
├── ldm_eval.py            Per-(suite, redshift) eval: PS, PDF KS, kurt, IC corr, etc.
└── LDM_LOSS_UPGRADES.md   Documented loss upgrades for next training round

eval/                      Standalone diagnostics
├── eval_comprehensive.py  VAE + residual comparison across all splits/groups
├── eval_full.py           Extended LDM eval (latent stats + outliers + full PS curves)
├── plot_ps.py             PS ratio plot for v4
└── test_aug_robustness.py Test whether v4 encoder is flip/rotation invariant

docs/
└── REPORT_v4.md           Full VAE v4 status report (per-redshift, per-split)

results/                   Eval artefacts (figures + machine-readable numbers)
├── RESULTS.md                  Test-set headline numbers + figure index (read this first)
├── vae_v4_final_eval.json      Per-dataset machine-readable VAE eval
├── vae_v4_best_worst.png       VAE best/worst test reconstructions (slices)
├── vae_v4_pdfs.png             VAE per-group voxel-value PDF overlays
├── ps_ratio_v4.png             VAE P(k) ratio across 6 (suite, z) groups
├── A_IC_z12_real_units.png     VAE IC z=12 best/worst — slices + PDF + PS in real mK
├── B_astro_corner.png          VAE astro parameter space, coloured by rel_MSE
├── C_LDM_trajectory.png        LDM key metrics (rel_MSE, PS, kurt) vs epoch
├── D_LDM_ps_curves.png         LDM ep179 PS ratio curves per (suite, z)
├── E_LDM_best_worst.png        LDM ep179 best/worst astro test samples
├── F_LDM_astro_corner.png      LDM ep179 astro parameter space, per-sample rel_MSE
└── F_LDM_astro_corner.json     Per-sample (params, rel_MSE) machine-readable
```

---

## Quick start

### Use the trained VAE

```python
import torch
from vae.vae import VAE3D

ckpt = torch.load('vae_v4_final.pt', map_location='cuda')   # not in repo; see bundle
cfg  = ckpt['model_config']
model = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
              base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).cuda()
model.load_state_dict(ckpt['model']); model.eval()

# Encode: (B, 1, 64, 64, 64) -> latent (B, 8, 32, 32, 32)
with torch.no_grad():
    mean, _ = model.encoder(x)
    recon   = model.decoder(mean)
```

Per-channel latent normalisation (required for downstream LDM training) is stored in
`ckpt['latent_mean']` and `ckpt['latent_std']`. See `vae/README_v4.md` §3 for usage.

### Re-train VAE (Stage 1)

```bash
python vae/train_vae.py \
  --data_root_ic /path/to/varying_IC \
  --data_root_astro /path/to/varying_astro \
  --redshifts 8 9 10 11 12 \
  --epochs 350 --batch_size 8 \
  --latent_ch 8 --base_ch 128 --ch_mults 1 \
  --kl_weight 3e-5 --kl_anneal 100 \
  --ps_weight 0.1 --ps_start 50 \
  --spec_weight 0.05 --spec_start 80 \
  --out_dir checkpoints/vae
```

### Re-train LDM (Stage 2)

```bash
python ldm/train_ldm.py \
  --vae_ckpt vae_v4_final.pt \
  --epochs 200 --batch_size 8 --lr 1e-4 \
  --patches_per_cube 4 \
  --save_every 10 --ps_check_every 10 \
  --out_dir checkpoints/ldm_v1
```

### Evaluate

```bash
python ldm/ldm_eval.py \
  --ldm_ckpt checkpoints/ldm_v1/ldm_epoch0179.pt \
  --vae_ckpt vae_v4_final.pt \
  --out ldm_eval.json \
  --split test
```

---

## Training set summary

| Suite | Train cubes | Redshifts | What varies | What's fixed |
|-------|-------------|-----------|-------------|--------------|
| `varying_astro` | 803 | z=10 only | 4 astro params (1003 unique vectors) | IC realisation (shared `delta1000.mat` / `vbv1000.mat`) |
| `varying_IC`    | 225 | z=8, 9, 10, 11, 12 | 81 IC seeds | astro params (fixed at fstarII=0.05, Vc=4.2, fX=1.0, delay=0.75) |

The two suites never sample the joint `(IC, astro_params)` plane — they cover only the
two axes that cross at one point. This is the structural reason the LDM struggles to
generalise to arbitrary `(IC, astro_params, z)` combinations and why the IC test set is
much weaker than the astro test set.

---

## Model checkpoints

Trained `*.pt` files are **not** committed (each is ~150 MB, beyond GitHub's hard limit).
The handover bundle `vae_v4_bundle.zip` (148 MB) is delivered separately and contains:

- `vae_v4_final.pt` — VAE checkpoint with `model`, `ema`, `latent_mean`, `latent_std`,
  `model_config`, full training args, epoch
- `vae.py`, `dataset.py` — model + data loader
- `vae_v4_final_eval.json` — machine-readable eval numbers
- `README_v4.md`, `REPORT_v4.md`, `REPORT_v4.pdf` — model card + status report
- `ps_ratio_v4.png` — PS ratio curves

LDM checkpoints (ep119, ep139, ep159, ep179) are kept on the GPU host; let me know
which epoch you want and they can be packaged the same way.

---

## What's next

1. **Drop varying_IC** for the next LDM training run. The two suites are not
   simulation-compatible; only `varying_astro` is the target.
2. **Generate joint (IC × params) simulations** so the LDM can learn the cross-term.
   Suggested scale: ~3000 new sims with random `(IC seed, astro_params)` pairs at
   z=6–30 (≈ 75 000 new cubes; ~10 hours on COSMA at 75-way parallelism).
3. **Finetune from ep179** with the loss additions documented in
   `ldm/LDM_LOSS_UPGRADES.md` (moment matching + latent spectral) once the new data
   arrives. Expected to push astro PS into the paper-grade [0.95, 1.05] band and lift
   kurt to within 10 % of the true value.
