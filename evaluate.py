"""
Evaluation script: generate 21cm fields from parameters and compare to ground truth.

Metrics (matching Simon's approach):
  1. Power spectrum Δ²(k)  — most important
  2. Pixel RMSE
  3. PDF of pixel values
  4. Visual slice comparison

Usage:
  python evaluate.py \
    --vae_ckpt  checkpoints/vae/vae_epoch0299.pt \
    --flow_ckpt checkpoints/flow/flow_epoch0499.pt \
    [--n_samples 50]  [--cfg_scale 3.0]
"""
import os, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import T21Dataset
from models.vae import VAE3D
from models.flow_matching import FlowUNet3D, ConditionalFlowMatcher
from utils.power_spectrum import power_spectrum


# -------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  default='/home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro')
    p.add_argument('--out_dir',    default='/home/ch2067/rds/hpc-work/21cm_gen/eval')
    p.add_argument('--vae_ckpt',   required=True)
    p.add_argument('--flow_ckpt',  required=True)
    p.add_argument('--redshifts',  nargs='+', type=int, default=[10])
    p.add_argument('--patch_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--n_samples',  type=int, default=50)
    p.add_argument('--cfg_scale',  type=float, default=3.0)
    p.add_argument('--num_steps',  type=int, default=100)
    p.add_argument('--latent_ch',  type=int, default=4)
    p.add_argument('--vae_base_ch',type=int, default=64)
    p.add_argument('--flow_base_ch',type=int, default=128)
    p.add_argument('--param_dim',  type=int, default=8)
    return p.parse_args()


# -------------------------------------------------------------------------
def rmse(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())


def ps_rmse(gen, real, Lpix=3.0):
    """RMSE of log10 Δ²(k) — same as Simon's rmse_dsq metric."""
    _, ps_gen  = power_spectrum(gen,  Lpix=Lpix)
    _, ps_real = power_spectrum(real, Lpix=Lpix)
    log_gen  = np.log10(ps_gen  + 1e-30)
    log_real = np.log10(ps_real + 1e-30)
    return float(np.sqrt(np.mean((log_gen.mean(0) - log_real.mean(0)) ** 2)))


# -------------------------------------------------------------------------
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # Models
    vae = VAE3D(1, args.latent_ch, args.vae_base_ch, (1,2)).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)['model'])
    vae.eval()

    unet = FlowUNet3D(args.latent_ch, args.flow_base_ch, (1,2),
                      param_dim=args.param_dim).to(device)
    flow = ConditionalFlowMatcher(unet).to(device)
    flow_ckpt = torch.load(args.flow_ckpt, map_location=device)
    # 优先加载 EMA 权重（向后兼容旧 checkpoint）
    flow.load_state_dict(flow_ckpt.get('ema', flow_ckpt['model']))
    flow.eval()

    # Data
    val_ds = T21Dataset(args.data_root, args.patch_size,
                        redshifts=args.redshifts, split='val')
    loader = DataLoader(val_ds, args.batch_size, shuffle=False)
    print(f"Val samples: {len(val_ds)}")

    all_rmse, all_ps_rmse = [], []
    real_patches, gen_patches = [], []

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if n_done >= args.n_samples:
                break
            x      = batch['patch'].to(device)          # (B,1,64,64,64)
            params = batch['params'].to(device)

            # Generate
            lat_shape = (x.shape[0], args.latent_ch, 16, 16, 16)
            z_gen  = flow.sample(params, lat_shape,
                                 num_steps=args.num_steps,
                                 cfg_scale=args.cfg_scale, device=device)
            x_gen  = vae.decode(z_gen)                  # (B,1,64,64,64)

            all_rmse.append(rmse(x_gen, x))
            all_ps_rmse.append(ps_rmse(x_gen.cpu(), x.cpu()))

            real_patches.append(x.cpu())
            gen_patches.append(x_gen.cpu())
            n_done += x.shape[0]
            print(f"  {n_done}/{args.n_samples} done")

    real = torch.cat(real_patches)[:args.n_samples]
    gen  = torch.cat(gen_patches) [:args.n_samples]

    mean_rmse    = float(np.mean(all_rmse))
    mean_ps_rmse = float(np.mean(all_ps_rmse))
    print(f"\nPixel RMSE:        {mean_rmse:.4f}")
    print(f"Power-spec RMSE:   {mean_ps_rmse:.4f}")

    # ---------- Plots ----------
    # 1. Power spectrum comparison
    k, ps_real = power_spectrum(real)
    _, ps_gen  = power_spectrum(gen)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.fill_between(k, ps_real.mean(0) - ps_real.std(0),
                       ps_real.mean(0) + ps_real.std(0),
                    alpha=0.3, color='C0', label='Real ±1σ')
    ax.fill_between(k, ps_gen.mean(0) - ps_gen.std(0),
                       ps_gen.mean(0) + ps_gen.std(0),
                    alpha=0.3, color='C1', label='Generated ±1σ')
    ax.loglog(k, ps_real.mean(0), 'C0-', label='Real mean')
    ax.loglog(k, ps_gen.mean(0),  'C1--', label='Gen mean')
    ax.set_xlabel('k [1/Mpc]'); ax.set_ylabel('Δ²(k)')
    ax.set_title('Power Spectrum'); ax.legend()

    # 2. Pixel PDF
    ax = axes[1]
    r_vals = real.numpy().ravel()
    g_vals = gen.numpy().ravel()
    bins = np.linspace(min(r_vals.min(), g_vals.min()),
                       max(r_vals.max(), g_vals.max()), 80)
    ax.hist(r_vals, bins=bins, density=True, alpha=0.6, label='Real',      color='C0')
    ax.hist(g_vals, bins=bins, density=True, alpha=0.6, label='Generated', color='C1')
    ax.set_xlabel('T21 (normalised)'); ax.set_ylabel('Density')
    ax.set_title('Pixel PDF'); ax.legend()

    # 3. Central slice comparison
    ax = axes[2]
    mid = args.patch_size // 2
    r_slice = real[0, 0, mid].numpy()
    g_slice = gen[0, 0, mid].numpy()
    vmin = min(r_slice.min(), g_slice.min())
    vmax = max(r_slice.max(), g_slice.max())
    combined = np.concatenate([r_slice, g_slice], axis=1)
    im = ax.imshow(combined, cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title('Real (left) | Generated (right)\nCentral slice')
    ax.axvline(args.patch_size - 0.5, color='white', lw=1)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, 'eval_summary.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {fig_path}")

    # Save metrics
    with open(os.path.join(args.out_dir, 'metrics.txt'), 'w') as f:
        f.write(f"n_samples:      {n_done}\n")
        f.write(f"pixel_rmse:     {mean_rmse:.6f}\n")
        f.write(f"ps_rmse:        {mean_ps_rmse:.6f}\n")
        f.write(f"cfg_scale:      {args.cfg_scale}\n")
        f.write(f"num_steps:      {args.num_steps}\n")

    print("Evaluation complete.")


if __name__ == '__main__':
    main()
