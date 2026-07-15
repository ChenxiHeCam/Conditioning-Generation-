# Stage-decomposition eval — runbook for the next GPU instance

Goal: publication-quality figures decomposing the pipeline into
**VAE only → VAE+LDM → VAE+LDM+Res**, on the 100 astro test cubes (z=10),
including equilateral **and squeezed** bispectrum, rel_MSE histograms,
and all residual-variant ablations (v2 / v3lowk / v4logk).

## 0. Prerequisites on the new instance

1. Rent RTX 4090 on AutoDL, PyTorch ≥ 2.0 image.
2. Restore the dataset to `/root/autodl-tmp/ASR21cm/`
   (`varying_astro` + `varying_IC` — from AutoDL netdisk or re-upload).
3. Upload code: the whole local `D:\Astro\repo\` → `/root/21cm_gen/`
   (needs `dataset.py`, `models/vae.py`, `ldm_unet.py`, `train_ldm.py`,
   `train_cond_residual.py`, `infer_stages.py`, `eval_stages_pub.py`).
4. Upload checkpoints from `D:\Astro\repo\ckpts_final\`:
   - `vae_v5_final.pt`
   - `ldm_v5_epoch0249.pt`
   - `cond_residual_v4logk_ep0029.pt`
   - `cond_residual_v3lowk_ep0019.pt`
   - `cond_residual_v2_ep0029.pt`

## 1. Generate stage-decomposed fields (~1.5–2 h for 100 cubes)

```bash
cd /root/21cm_gen
CK=/root/autodl-tmp/checkpoints
python infer_stages.py \
  --vae_ckpt $CK/vae_v5_final.pt \
  --ldm_ckpt $CK/ldm_v5_epoch0249.pt \
  --res_ckpts v4logk:$CK/cond_residual_v4logk_ep0029.pt \
              v3lowk:$CK/cond_residual_v3lowk_ep0019.pt \
              v2:$CK/cond_residual_v2_ep0029.pt \
  --out_dir /root/autodl-tmp/stages_100 \
  --n_cubes 100 --steps 30 --seed 0
```

Each npz holds 5 aligned 256³ fields (true, vae_only, ldm_vae, +3 res
variants). ~320 MB/cube → **~32 GB**; check disk first (`df -h`).
All residual variants share the same LDM latent draw → exact ablation.

## 2. Figures (~20 min, GPU FFTs)

```bash
python eval_stages_pub.py \
  --in_dir /root/autodl-tmp/stages_100 \
  --out_dir /root/autodl-tmp/stages_100_figs \
  --n_cubes 100
```

Outputs (png+pdf): S1 PS stages, S2 equilateral bispec, S3 **squeezed**
bispec, S4 rel_MSE hist, S5 slice gallery, S6 PS-band hists,
`stage_metrics.json`, `STAGE_SUMMARY.md`.

## 3. Validation-set version (rel_MSE histogram on val, as requested)

```bash
python infer_stages.py ... --split val --out_dir /root/autodl-tmp/stages_val \
  --n_cubes 100
python eval_stages_pub.py --in_dir /root/autodl-tmp/stages_val \
  --out_dir /root/autodl-tmp/stages_val_figs
```

## 4. Download back to local

`stages_100_figs/` + `stages_val_figs/` + the two `stage_metrics.json`.
Raw npz fields are large — only download 2–3 example cubes for slice
figures if needed later.

## Notes

- `eval_stages_pub.py` auto-discovers stages present in the npz; figure
  colors/labels are in `STAGE_LABEL`/`STAGE_COLOR` at the top.
- Squeezed bispectrum: long leg fixed at k_L ∈ [k_box, 0.04] cMpc⁻¹,
  hard legs log-spaced 0.05 → 0.95·k_Nyquist.
- Dry-run validated locally on synthetic cubes 2026-06-12.
