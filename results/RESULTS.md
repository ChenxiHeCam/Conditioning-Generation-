# Test-set results

All metrics below are evaluated on the **held-out test split** (stratified 80/10/10
split per (suite, redshift), seed 42). The test split is disjoint from training and from
the validation split. PS ratios use only k bins with a non-empty Fourier-mode count
(`ps.mean(axis=0) > 1e-20`); empty bins are masked out before band averaging.

---

## VAE v4 — test set

Stage-1 reconstruction quality (encode → decode through the frozen VAE). All redshifts
and both suites are included. Metric definitions:

- `rel_MSE = MSE(recon, true) / Var(true)` (lower is better; 0 = perfect)
- `PS ratio = P_recon(k) / P_true(k)`, averaged in three k bands
  - `PS large`: k < 0.1 h/Mpc
  - `PS mid`:   0.1 ≤ k < 0.5
  - `PS small`: k ≥ 0.5
- Paper-grade target: `PS ratio` ∈ [0.95, 1.05]

| Dataset             | n   | rel_MSE   | PS large | PS mid | PS small |
|---------------------|-----|-----------|----------|--------|----------|
| **astro test z=10** | 100 | **9.2e-5**| **0.993**| 0.995  | 0.992    |
| IC test z=8         | 5   | 1.2e-4    | 0.987    | 0.992  | 0.992    |
| IC test z=9         | 5   | 1.1e-4    | 0.991    | 0.995  | 0.992    |
| IC test z=10        | 8   | 1.6e-4    | 0.991    | 0.993  | 0.988    |
| IC test z=11        | 5   | 3.3e-4    | 0.979    | 0.984  | 0.987    |
| IC test z=12        | 5   | 1.4e-3    | 0.969    | 0.966  | 0.966    |

**Headline numbers (test set):**
- Best:  astro z=10 — rel_MSE 9 × 10⁻⁵ (≈ 99.99 % explained variance)
- Worst: IC z=12   — rel_MSE 1.4 × 10⁻³ (≈ 99.86 % explained variance)
- All PS ratios within [0.965, 1.0]; all three k bands paper-grade on every (suite, z).
- Per-sample outliers (|PS – 1| > 20 %): 0 / 100 on astro, 0 / 5 on every IC redshift.

Full machine-readable numbers in `vae_v4_final_eval.json`.

---

## LDM v1 (ep 179) — test set

Stage-2 latent diffusion: generates T21 patches from (IC delta, IC vbv, 4 astro params,
redshift), no access to the true T21. Heun 18-step EDM sampling, CFG scales w_IC = w_p = 1.

| Group               | n   | rel_MSE | PS large | PS mid | PS small | pixel_corr | kurt recon / true |
|---------------------|-----|---------|----------|--------|----------|------------|-------------------|
| **astro test z=10** | 100 | **0.054** | **0.83** | 0.83   | 0.80     | **0.905**  | **1.33 / 2.05**   |
| IC test z=8         | 5   | 0.96    | 1.65     | 1.88   | 1.94     | 0.87       | 0.65 / -0.21      |
| IC test z=9         | 5   | 0.79    | 1.55     | 1.74   | 1.85     | 0.87       | 1.36 / 0.78       |
| IC test z=10        | 8   | 0.66    | 2.29     | 1.92   | 1.90     | 0.93       | 2.13 / 1.43       |
| IC test z=11        | 5   | 2.10    | 1.85     | 2.12   | 2.32     | 0.89       | 1.20 / 0.22       |
| IC test z=12        | 5   | 3.10    | 1.64     | 1.95   | 1.70     | 0.93       | 0.32 / 0.55       |

Additional astro metrics (test, n=100):

| Metric                   | Value           | Comment                                |
|--------------------------|-----------------|----------------------------------------|
| `cross_coherence`        | 0.899           | Fourier-phase alignment ≈ 90 %         |
| `IC_corr` (recon / true) | -0.70 / -0.71   | 99 % match — model learnt the IC→T21 sign |
| `skew` (recon / true)    | 0.21 / 0.18     | Matches                                |
| `kurt` (recon / true)    | 1.33 / 2.05     | Recon distribution somewhat Gaussianised |
| `pdf_ks`                 | 0.047           | Pixel-value PDF nearly overlaps        |
| `ion_frac` (recon / true)| 0.29 / 0.34     | Recon under-predicts ionised fraction by ~15 % |
| `sample_diversity_rel`   | 0.57            | No mode collapse                       |

**Headline takeaways (test set):**

1. **astro z=10** is the strong group: rel_MSE 0.054 ≈ 5 % of the field's variance left
   as residual. pixel_corr 0.905 means per-voxel Pearson is high. Per-sample IC
   alignment (-0.70 vs -0.71) shows the model captured the IC→T21 physical sign.
2. **PS underestimated by 17 %** across the three k bands (0.80-0.83 vs target 0.95-1.05).
   This is the main gap to paper-grade quality for the astro suite.
3. **Kurtosis ≈ 65 % of true** (1.33 vs 2.05) — generated samples are slightly more
   Gaussian than real T21; ionisation-bubble edges are under-amplified.
4. **Per-sample LDM rel_MSE distribution on astro test** (`F_LDM_astro_corner.png`):
   median 0.28, mean 0.29, max 2.14. The worst-performing samples cluster at
   **high X-ray flux (log₁₀ f_X ≳ 0)** and **low circular velocity (V_c ≲ 15 km/s)** —
   the strong-heating and small-galaxy regimes are the hardest for the LDM to
   generate accurately.
5. **IC suite (z=8-12) clearly worse**: rel_MSE ≥ 0.66, PS ratios 1.6-2.3 (over-estimate).
   The current training data structure (1 IC × 1999 params for astro vs 81 IC × 1 params
   for IC) means the joint (IC × params) cross-term was never observed by the model.
   Per the collaborator's direction, future training will drop varying_IC and focus on
   varying_astro alone, so IC numbers here are mostly informational.

---

## VAE vs LDM — task differences

| Task                  | Input → output                                | Best test rel_MSE (astro) |
|-----------------------|-----------------------------------------------|---------------------------|
| VAE reconstruction    | T21 → latent → T21                            | 9.2e-5                    |
| LDM generation        | (IC, params, z) → latent → T21 (no T21 input) | 0.054                     |

These two are not directly comparable: the VAE has access to the input T21 and only
needs to compress it through the latent bottleneck, while the LDM must hallucinate the
T21 field given only the conditioning (IC + params + z). The LDM residual variance is
therefore bounded below by the irreducible variance of `p(T21 | IC, params, z)` for the
training simulator settings.

---

## Files in this folder

| File                       | What it shows |
|----------------------------|----------------|
| `vae_v4_final_eval.json`   | Machine-readable per-dataset VAE eval (val + test) |
| `ps_ratio_v4.png`          | VAE P(k) ratio across 6 (suite, z) groups |
| `vae_v4_best_worst.png`    | VAE best/worst test reconstructions (slices) |
| `vae_v4_pdfs.png`          | VAE per-group voxel-value PDF overlays |
| `A_IC_z12_real_units.png`  | VAE IC z=12 best/worst — slices + PDF + 1D PS in mK |
| `B_astro_corner.png`       | VAE astro parameter-space corner, coloured by rel_MSE |
| `C_LDM_trajectory.png`     | LDM key metrics vs epoch (val + test, both suites) |
| `D_LDM_ps_curves.png`      | LDM ep179 PS ratio curves per (suite, z) test set |
| `E_LDM_best_worst.png`     | LDM ep179 astro test best/worst — slices + PDF + 1D PS |
| `F_LDM_astro_corner.png`   | LDM ep179 per-sample rel_MSE across astro parameter space |
| `F_LDM_astro_corner.json`  | Machine-readable per-sample (params, rel_MSE) |
