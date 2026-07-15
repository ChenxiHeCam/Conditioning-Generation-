"""
Evaluate trained Flow Matching model.
Generates samples, compares power spectra with ground truth, saves plots + stats.

Usage:
    python evaluate_flow.py \
        --vae_ckpt   checkpoints/vae/vae_epoch0299.pt \
        --flow_ckpt  checkpoints/flow/flow_epoch0499.pt \
        --out_dir    eval_results/run1
"""
import os, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import T21Dataset
from models.vae import VAE3D
from models.flow_matching import FlowUNet3D, ICEncoder, ConditionalFlowMatcher
from utils.power_spectrum import power_spectrum


# -------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  default='/home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro')
    p.add_argument('--vae_ckpt',   required=True)
    p.add_argument('--flow_ckpt',  required=True)
    p.add_argument('--out_dir',    default='eval_results')
    p.add_argument('--redshifts',  nargs='+', type=int, default=[10])
    p.add_argument('--patch_size', type=int, default=64)
    p.add_argument('--n_samples',  type=int, default=16,  help='Number of cubes to generate')
    p.add_argument('--num_steps',  type=int, default=100, help='ODE steps (use 20 after reflow)')
    p.add_argument('--cfg_scale',  type=float, default=3.0)
    p.add_argument('--latent_ch',  type=int, default=4)
    p.add_argument('--base_ch',    type=int, default=128)
    p.add_argument('--vae_base_ch',type=int, default=64)
    p.add_argument('--param_dim',  type=int, default=5)
    p.add_argument('--use_dit_mid',action='store_true')
    p.add_argument('--Lpix',       type=float, default=3.0, help='Mpc per voxel for PS')
    p.add_argument('--seed',       type=int, default=0)
    return p.parse_args()


# -------------------------------------------------------------------------
def load_models(args, device):
    # VAE
    vae = VAE3D(in_ch=1, latent_ch=args.latent_ch,
                base_ch=args.vae_base_ch, ch_mults=(1, 2)).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(ckpt['model'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Flow
    ic_encoder = ICEncoder(in_ch=2, out_ch=4).to(device)
    unet = FlowUNet3D(
        latent_ch=args.latent_ch, ic_ch=4, base_ch=args.base_ch, ch_mults=(1, 2),
        param_dim=args.param_dim, param_embed_dim=256,
        use_dit_mid=args.use_dit_mid,
    ).to(device)
    flow = ConditionalFlowMatcher(unet, ic_encoder).to(device)
    ckpt = torch.load(args.flow_ckpt, map_location=device)
    # 优先加载 EMA 权重（向后兼容旧 checkpoint）
    flow.load_state_dict(ckpt.get('ema', ckpt['model']))
    flow.eval()

    print(f"Loaded VAE from  {args.vae_ckpt}")
    ema_tag = " [EMA]" if 'ema' in ckpt else ""
    print(f"Loaded Flow from {args.flow_ckpt}  (epoch {ckpt.get('epoch', '?')}){ema_tag}")
    return vae, flow


# -------------------------------------------------------------------------
@torch.no_grad()
def generate_samples(flow, vae, batch, device, args):
    """Generate samples for a batch of conditions, return (generated, real) denormalised."""
    params   = batch['params'].to(device)    # (B, param_dim)
    ic_delta = batch['ic_delta'].to(device)
    ic_vbv   = batch['ic_vbv'].to(device)
    real     = batch['patch'].to(device)     # (B, 1, 64, 64, 64)

    B = params.shape[0]
    latent_shape = (B, args.latent_ch, 16, 16, 16)

    z_gen = flow.sample(
        params, ic_delta, ic_vbv,
        latent_shape=latent_shape,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        device=device,
    )
    x_gen = vae.decode(z_gen)    # (B, 1, 64, 64, 64)
    return x_gen.cpu(), real.cpu()


# -------------------------------------------------------------------------
def plot_power_spectra(ps_gen_all, ps_real_all, k, out_dir):
    """Plot mean ± std power spectra: generated vs real."""
    gen_mean  = ps_gen_all.mean(0)
    gen_std   = ps_gen_all.std(0)
    real_mean = ps_real_all.mean(0)
    real_std  = ps_real_all.std(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Δ²(k) comparison
    ax = axes[0]
    ax.fill_between(k, gen_mean - gen_std,  gen_mean + gen_std,  alpha=0.3, color='C0')
    ax.fill_between(k, real_mean - real_std, real_mean + real_std, alpha=0.3, color='C1')
    ax.loglog(k, gen_mean,  color='C0', label='Generated')
    ax.loglog(k, real_mean, color='C1', label='Real', linestyle='--')
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('Δ²(k)')
    ax.set_title('Power Spectrum Δ²(k)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    # Ratio: generated / real
    ax = axes[1]
    ratio      = gen_mean / (real_mean + 1e-30)
    ratio_std  = ratio * np.sqrt((gen_std / (gen_mean + 1e-30))**2 +
                                  (real_std / (real_mean + 1e-30))**2)
    ax.semilogx(k, ratio, color='C0', label='Gen / Real')
    ax.fill_between(k, ratio - ratio_std, ratio + ratio_std, alpha=0.3, color='C0')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.axhspan(0.9, 1.1, color='gray', alpha=0.1, label='±10%')
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('PS ratio')
    ax.set_title('Generated / Real ratio')
    ax.set_ylim(0.5, 1.5)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'power_spectrum.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_slices(gen_batch, real_batch, out_dir, n=4):
    """2-D mid-slice comparison: generated vs real."""
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    for i in range(n):
        mid = gen_batch.shape[-1] // 2
        axes[0, i].imshow(real_batch[i, 0, :, :, mid].numpy(), cmap='RdBu_r')
        axes[0, i].set_title(f'Real #{i}')
        axes[0, i].axis('off')
        axes[1, i].imshow(gen_batch[i, 0, :, :, mid].numpy(), cmap='RdBu_r')
        axes[1, i].set_title(f'Gen #{i}')
        axes[1, i].axis('off')
    plt.tight_layout()
    path = os.path.join(out_dir, 'slices.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_loss_curves(out_dir):
    """Read flow log.csv from the checkpoint dir and plot training curves."""
    # Try to find log.csv next to the checkpoint
    log_path = os.path.join(os.path.dirname(out_dir), 'log.csv')
    # fallback: look in common checkpoint dirs
    for candidate in [log_path,
                      '/home/ch2067/rds/hpc-work/21cm_gen/checkpoints/flow/log.csv']:
        if os.path.exists(candidate):
            data = np.loadtxt(candidate, delimiter=',', skiprows=1)
            if data.ndim == 1:
                data = data[None]
            epochs, tr, vl = data[:, 0], data[:, 1], data[:, 2]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.semilogy(epochs, tr, label='Train')
            ax.semilogy(epochs, vl, label='Val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Flow Matching Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = os.path.join(out_dir, 'loss_curves.png')
            plt.savefig(p, dpi=150)
            plt.close()
            print(f"  Saved {p}")
            return
    print("  [skip] log.csv not found for loss curves")


# -------------------------------------------------------------------------
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    vae, flow = load_models(args, device)

    ds = T21Dataset(args.data_root, args.patch_size,
                    redshifts=args.redshifts, split='val')
    print(f"Val set: {len(ds)} samples")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=min(8, args.n_samples),
                        shuffle=False, num_workers=2)

    all_gen, all_real = [], []
    total = 0
    for batch in loader:
        if total >= args.n_samples:
            break
        gen, real = generate_samples(flow, vae, batch, device, args)
        all_gen.append(gen)
        all_real.append(real)
        total += gen.shape[0]
        print(f"  Generated {total}/{args.n_samples}")

    gen_all  = torch.cat(all_gen,  0)[:args.n_samples]
    real_all = torch.cat(all_real, 0)[:args.n_samples]

    # Denormalise using dataset stats
    gen_all  = gen_all  * ds.t21_std + ds.t21_mean
    real_all = real_all * ds.t21_std + ds.t21_mean

    # Power spectra
    print("Computing power spectra...")
    k, ps_gen  = power_spectrum(gen_all,  Lpix=args.Lpix, kbins=30)
    k, ps_real = power_spectrum(real_all, Lpix=args.Lpix, kbins=30)

    plot_power_spectra(ps_gen, ps_real, k, args.out_dir)
    plot_slices(gen_all, real_all, args.out_dir, n=min(4, args.n_samples))
    plot_loss_curves(args.out_dir)

    # Print summary stats
    ratio = ps_gen.mean(0) / (ps_real.mean(0) + 1e-30)
    print(f"\nPS ratio summary (gen/real):")
    print(f"  Large scale (k<0.1):  {ratio[k < 0.1].mean():.3f}")
    print(f"  Mid scale:            {ratio[(k >= 0.1) & (k < 0.5)].mean():.3f}")
    print(f"  Small scale (k>0.5):  {ratio[k >= 0.5].mean():.3f}")
    print(f"  Overall mean ratio:   {ratio.mean():.3f}")

    # Save raw PS arrays for further analysis
    np.savez(os.path.join(args.out_dir, 'power_spectra.npz'),
             k=k, ps_gen=ps_gen, ps_real=ps_real)
    print(f"  Saved power_spectra.npz")

    print("\nEvaluation complete.")


if __name__ == '__main__':
    main()
