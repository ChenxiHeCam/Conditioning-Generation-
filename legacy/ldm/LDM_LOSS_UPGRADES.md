# LDM Loss Upgrades — Ready to Apply After Jiten's New Data

**Context (as of 2026-05-22):** Current LDM at ep~99 has two structural failures that pure EDM-MSE
loss cannot fix:

1. **PS deviation** — astro 0.91-0.93 (7-10% off), IC 1.8-2.0 (80-100% off). Target [0.95, 1.05].
2. **Kurtosis underprediction** — recon 0.96-1.07 vs target 1.76-2.07. MSE is mean-seeking,
   intrinsically cannot reproduce heavy tails of T21 (ionization bubble edges).

**Plan**: when Jiten delivers the 1600 new (IC × params) cubes, train v2 LDM with these loss
additions. The functions below are drop-in additions to `train_ldm.py`.

---

## 1. Moment-matching loss (PRIORITY 1 — addresses kurt directly)

Cheapest possible. Per-batch tensor stats. ~10 μs overhead.

```python
def moment_match_loss(x_pred, x_target, eps=1e-6):
    """Match (std, skew, kurt) per channel between predicted and target latents.
    x: (B, C, D, H, W). Moments computed over spatial dims."""
    def moments(x):
        m = x.mean(dim=(2,3,4), keepdim=True)
        c = x - m
        s = c.pow(2).mean(dim=(2,3,4)).clamp_min(eps).sqrt()
        sk = c.pow(3).mean(dim=(2,3,4)) / s.pow(3)
        kt = c.pow(4).mean(dim=(2,3,4)) / s.pow(4) - 3
        return s, sk, kt
    sp, skp, ktp = moments(x_pred)
    st, skt, ktt = moments(x_target)
    return F.mse_loss(sp, st) + F.mse_loss(skp, skt) + F.mse_loss(ktp, ktt)
```

**Hook in edm_loss:**
```python
loss = base_edm_loss + 0.1 * moment_match_loss(x_hat, x_clean)
```

**Expected:** kurt 0.96 → 1.5+; risk near 0.

---

## 2. Latent spectral loss (PRIORITY 1 — addresses PS in latent space)

Cheap (~5 ms/batch). Directly penalizes Fourier spectral mismatch in normalized latent space.

```python
def latent_spectral_loss(x_pred, x_target):
    """Log-magnitude MSE in 3D Fourier space at the latent level."""
    fr = torch.fft.rfftn(x_pred.float(), dim=(-3,-2,-1), norm='ortho')
    ft = torch.fft.rfftn(x_target.float(), dim=(-3,-2,-1), norm='ortho')
    return F.mse_loss(fr.abs().log1p(), ft.abs().log1p())
```

**Hook:**
```python
loss = base_edm_loss \
       + 0.1  * moment_match_loss(x_hat, x_clean) \
       + 0.05 * latent_spectral_loss(x_hat, x_clean)
```

**Expected:** PS 1.20 → 1.08; risk near 0.

---

## 3. T21-space PS loss (PRIORITY 2 — most direct PS fix, expensive)

Conditional on small sigma (denoising end-stage) to keep cost manageable. Decodes through frozen
VAE then matches spherically-averaged PS in T21 space — the actual paper metric.

```python
def t21_ps_loss(x_pred_latent, x_target_latent, vae,
                latent_mean, latent_std, n_bins=20):
    """PS MSE in T21 space. Uses frozen VAE.decoder."""
    with torch.no_grad():
        x_target = vae.decoder(x_target_latent * latent_std + latent_mean)
    x_pred = vae.decoder(x_pred_latent * latent_std + latent_mean)
    # reuse ps_loss helper from train_vae.py (log1p MSE per spherical bin)
    return ps_loss(x_pred, x_target, n_bins=n_bins)
```

**Hook (apply only at low sigma):**
```python
loss = base_edm_loss + 0.1 * moment_match_loss(x_hat, x_clean) \
                     + 0.05 * latent_spectral_loss(x_hat, x_clean)
if sigma.mean() < 0.5:
    loss = loss + 0.1 * t21_ps_loss(x_hat, x_clean, vae, latent_mean, latent_std)
```

**Cost:** +50-80 ms/batch when active (≈10% of batches). Total epoch ~+5-8%.
**Expected:** PS 1.20 → 1.02 — close to paper-grade.

---

## 4. Patch discriminator + adversarial loss (PRIORITY 3 — heavy weapon for kurt)

Use only if moment_match isn't enough. 3D patch-level discriminator with hinge loss.

```python
class PatchDisc3D(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv3d(base, base*2, 4, 2, 1), nn.GroupNorm(8, base*2), nn.LeakyReLU(0.2),
            nn.Conv3d(base*2, base*4, 4, 2, 1), nn.GroupNorm(8, base*4), nn.LeakyReLU(0.2),
            nn.Conv3d(base*4, 1, 4, 1, 0),
        )
    def forward(self, x): return self.net(x)

# In training loop:
def gan_loss_g(disc, fake): return -disc(fake).mean()
def gan_loss_d(disc, real, fake):
    return F.relu(1 - disc(real)).mean() + F.relu(1 + disc(fake)).mean()

# Generator step (current model):
loss_g = base + 0.05 * gan_loss_g(disc, x_hat_decoded_T21)

# Discriminator step (separate optimizer):
opt_d.zero_grad()
loss_d = gan_loss_d(disc, x_clean_decoded.detach(), x_hat_decoded.detach())
loss_d.backward(); opt_d.step()
```

**Cost:** +~5M params, +200 ms/batch.
**Expected:** kurt 0.96 → 1.7-1.8.
**Risk:** GAN training instability — only apply after moment+spectral loss verified.

---

## 5. Curriculum / two-stage schedule

Train phase 1 with pure EDM MSE for structure learning (e.g. ep 0-100), then add moment+spectral
losses for distribution polishing (ep 100+). Implemented by passing epoch to edm_loss:

```python
def adaptive_loss_weights(epoch):
    if epoch < 100:
        return dict(moment=0.0, spec=0.0, t21_ps=0.0, gan=0.0)
    elif epoch < 150:
        return dict(moment=0.1, spec=0.05, t21_ps=0.0, gan=0.0)
    else:
        return dict(moment=0.2, spec=0.1, t21_ps=0.1, gan=0.0)
```

---

## Recommended combos for v2 LDM run

### Conservative (safe, ~30% improvement expected)

```
loss = base_edm
       + 0.1  * moment_match_loss
       + 0.05 * latent_spectral_loss
```
Risk: 0. Expected: kurt 0.96 → 1.4; PS 1.20 → 1.10.

### Aggressive (paper-grade aim, higher risk)

```
loss = base_edm
       + 0.1  * moment_match_loss
       + 0.05 * latent_spectral_loss
       + (if sigma<0.5) 0.1 * t21_ps_loss
       + 0.05 * gan_loss_g  (with separate disc training)
```
Risk: GAN instability. Expected: kurt 0.96 → 1.7+; PS 1.20 → 1.02.

---

## When applying (after new data arrives)

1. `cp train_ldm.py train_ldm_v2.py`
2. Add the chosen loss functions above
3. Modify `edm_loss(...)` to call them
4. Start from current best ckpt (ep~199 if training completed)
5. Finetune ~40 epochs with **new data** + new losses
6. Re-run `ldm_eval.py` to confirm improvement
