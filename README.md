# Conditioning-Generation: 21cm Brightness-Temperature Emulator

3D latent diffusion emulator for the 21cm brightness-temperature field, conditioned on
cosmological initial conditions (IC) and astrophysical parameters.

The pipeline has two stages:

1. **Stage 1 — VAE3D**: encodes 64³ T21 patches to an (8, 32, 32, 32) latent.
   Trained on `varying_astro` + `varying_IC` simulation suites.
2. **Stage 2 — Latent Diffusion Model (LDM)**: 3D U-Net denoiser operating on the
   normalised latent, conditioned on IC density + velocity, four astrophysical parameters,
   redshift, and EDM noise level σ.

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
└── test_aug_robustness.py Test if v4 encoder is flip/rotation invariant

docs/                      Reports and walkthroughs
├── REPORT_v4.md           Full VAE v4 status report (per-redshift, per-split)
├── CODE_WALKTHROUGH.md    Top-down code map
└── HANDOFF.md             What's done / what to do next

results/                   Eval artefacts
├── vae_v4_final_eval.json     Per-dataset machine-readable eval
├── vae_v4_best_worst.png      Best/worst test reconstructions (slices)
├── vae_v4_pdfs.png            Per-group voxel-value PDF overlays
├── ps_ratio_v4.png            P_recon(k) / P_true(k) across 6 (suite, z) groups
├── A_IC_z12_real_units.png    varying_IC z=12 best/worst — slices+PDF+PS in real mK
├── B_astro_corner.png         varying_astro parameter space, coloured by rel_MSE
├── C_LDM_trajectory.png       LDM key metrics (rel_MSE, PS, kurt) vs epoch
├── D_LDM_ps_curves.png        LDM ep179 PS ratio curves per (suite, z)
└── E_LDM_best_worst.png       LDM ep179 best/worst astro test samples
```

## Quick start — using the trained VAE

```python
import torch
from vae.vae import VAE3D

ckpt = torch.load('vae_v4_final.pt', map_location='cuda')   # not in repo, see bundle
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
`ckpt['latent_mean']` and `ckpt['latent_std']`. See `vae/README_v4.md` §3.

## Headline results

- **VAE v4**: MSE/var 8.4e-5 ~ 1.4e-3 across 12 (suite, split, redshift) groups; all
  PS ratios in [0.965, 1.00] across z=8–12 (paper-grade).
- **LDM v1** (in progress, ~ep179): astro test rel_MSE 0.054, PS ratio 0.83-0.91
  (slightly underestimating, not yet paper-grade), kurt 1.33 vs true 1.78. IC suite
  weaker due to insufficient (IC × params) joint coverage in the training data.

Detailed metrics in `docs/REPORT_v4.md` and `vae/README_v4.md`.

## Model checkpoints

Trained `*.pt` files are **not** committed (~150 MB each, exceeds GitHub's hard limit).
The handover bundle `vae_v4_bundle.zip` (148 MB) is delivered separately.

