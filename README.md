# 21cm latent diffusion (v5, 8× compression)

Conditional 3D generative model for the 21cm brightness-temperature field at
high z. Conditions on the initial-condition field (density + relative
velocity), four astrophysical parameters and redshift. Output is a `1×64³`
patch in mK.

The earlier 1× pipeline (VAE v4 + LDM v1) is kept in `legacy/` for reference;
this folder is the current code.

## Pipeline

LDM diffuses an `(8, 16, 16, 16)` latent. The frozen VAE decodes it to a
`1×64³` field. A small conditional residual U-Net adds a per-voxel correction
that uses the IC field and the astrophysical conditioning. The whole thing
runs in a single forward of each model.

```
z ~ LDM(IC, params, redshift)
x_low = VAE.decoder(z * latent_std + latent_mean)
r_hat = cond_residual(x_low, IC, params, redshift)
T21   = x_low + r_hat
```

## Checkpoints

| File | What |
|---|---|
| `vae_v5_final.pt` | VAE encoder + decoder (42.6 M), with per-channel latent stats and `model_config` so the LDM can load it directly |
| `ldm_epoch0249.pt` | LDM denoiser, 5.7 M parameters, 250 epochs |
| `cond_residual_v2_ep0029.pt` | Residual U-Net, 6.1 M parameters, finetuned for 30 epochs from the stage-1 ckpt with mixed VAE + LDM training pairs |

## Numbers (test split, n=64 patches, z=10)

| | rel MSE | PS large | PS mid | PS small |
|---|---|---|---|---|
| LDM alone | 0.21 | 1.32 | 1.15 | 0.46 |
| LDM + cond residual | 0.012 | 0.72 | 1.04 | 1.07 |

The PS large value of 0.72 includes the DC mode in the bin (the `power_spectrum`
helper does not mean-subtract before FFT). With mean subtraction or restricting
to `k > 0.05 cMpc⁻¹` the same number is ~0.95.

## Files

```
dataset.py                  T21Dataset, including train/val/test/holdout splitting
ldm_unet.py                 LDM 3D U-Net; ICStem and the U-Net take an extra
                            downsample for the 16³ latent
train_vae.py                VAE training
train_ldm.py                LDM training
finalize_vae.py             writes latent_mean / latent_std / model_config into a
                            VAE checkpoint so train_ldm.py can load it
sample_ldm_pairs.py         offline LDM sampling; produces (x_low, real_x, IC,
                            params) tuples for the residual stage-2 training
train_cond_residual.py      conditional residual model + stage-1 trainer (VAE
                            encode-decode pairs only)
train_cond_residual_v2.py   stage-2 trainer (mixed VAE + LDM pairs)
evaluate_vae.py             VAE-only evaluation (PS bands + slices)
ldm_eval.py                 LDM evaluation (sampling, PS, kurtosis, PDF KS,
                            per-suite breakdown)
```

## Reproducing

```bash
# 1) VAE
python train_vae.py \
  --data_root_ic   /path/to/varying_IC \
  --data_root_astro /path/to/varying_astro \
  --redshifts 8 9 10 11 12 \
  --max_per_z 50 --primary_z 10 --holdout_frac 0.25 \
  --out_dir checkpoints/vae_v5 \
  --ch_mults 1 2 --latent_ch 8 --base_ch 128 \
  --batch_size 8 --lr 1e-4 --epochs 350 \
  --kl_weight 3e-5 --kl_anneal 100 \
  --ps_weight 0.3 --spec_weight 0.05 \
  --ps_start 25 --spec_start 25 --ramp_epochs 25 \
  --ps_k_alpha 2.0 --save_every 25 --patches_per_cube 4 --num_workers 4

# 2) finalize
python finalize_vae.py \
  --ckpt_in  checkpoints/vae_v5/vae_epoch0149.pt \
  --ckpt_out checkpoints/vae_v5/vae_v5_final.pt \
  --latent_ch 8 --base_ch 128 --ch_mults 1 2 --latent_spatial 16

# 3) LDM
python train_ldm.py \
  --vae_ckpt checkpoints/vae_v5/vae_v5_final.pt \
  --out_dir  checkpoints/ldm_v5 \
  --epochs 250 --batch_size 16 --lr 1e-4 \
  --patches_per_cube 4 --save_every 10 --num_workers 4 \
  --ps_check_every 9999

# 4) offline LDM pairs (~10 min)
python sample_ldm_pairs.py \
  --ldm_ckpt checkpoints/ldm_v5/ldm_epoch0249.pt \
  --vae_ckpt checkpoints/vae_v5/vae_v5_final.pt \
  --out_path ldm_pairs_K2.pt \
  --K 2 --num_steps 18 --val_frac 0.67 \
  --redshifts 8 9 10 11 12 \
  --max_per_z 50 --primary_z 10 --holdout_frac 0.25

# 5) residual, two stages
python train_cond_residual.py \
  --vae_ckpt checkpoints/vae_v5/vae_v5_final.pt \
  --out_dir  checkpoints/cond_residual \
  --epochs 80 --lr 1e-4 \
  --kl_weight 1e-5 --ps_weight 0.5 --high_k_alpha 3.0 \
  --redshifts 8 9 10 11 12 \
  --max_per_z 50 --primary_z 10 --holdout_frac 0.25

python train_cond_residual_v2.py \
  --vae_ckpt  checkpoints/vae_v5/vae_v5_final.pt \
  --init_ckpt checkpoints/cond_residual/cond_residual_ep0079.pt \
  --ldm_pairs_file ldm_pairs_K2.pt \
  --out_dir  checkpoints/cond_residual_v2 \
  --epochs 30 --lr 5e-5 --ldm_frac 0.4 \
  --ps_weight 0.5 --high_k_alpha 3.0
```

## Inference

```python
import torch
from models.vae import VAE3D
from ldm_unet import LDMUNet3D
from train_ldm import heun_sample
from train_cond_residual import CondResidualUNet

device = 'cuda'
vae_ck = torch.load('checkpoints/vae_v5_final.pt', map_location=device)
cfg = vae_ck['model_config']
vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
            base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(device).eval()
vae.load_state_dict(vae_ck['model'])
lm = vae_ck['latent_mean'].view(1,-1,1,1,1).to(device)
ls = vae_ck['latent_std' ].view(1,-1,1,1,1).to(device)

ldm = LDMUNet3D(latent_ch=cfg['latent_ch'],
                ch_mults=(1,2), attn_levels=(0,1),
                ic_stem_downsamples=2).to(device).eval()
ldm.load_state_dict(torch.load('checkpoints/ldm_epoch0249.pt', map_location=device)['model'])

res = CondResidualUNet(base_ch=32).to(device).eval()
res.load_state_dict(torch.load('checkpoints/cond_residual_v2_ep0029.pt',
                               map_location=device)['model'])

@torch.no_grad()
def generate(ic_delta, ic_vbv, params, redshift):
    B = ic_delta.shape[0]
    z = heun_sample(ldm, B, tuple(cfg['latent_shape']),
                    ic_delta, ic_vbv, params, redshift,
                    sigma_data=1.121, num_steps=18,
                    cfg_ic=1.0, cfg_params=1.0,
                    null_param=None, device=device)
    x_low = vae.decoder(z * ls + lm)
    return x_low + res(x_low, ic_delta, ic_vbv, params, redshift)
```
