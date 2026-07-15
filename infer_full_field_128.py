# Inference at 128^3 patches with new LDM-128 + old VAE (works fully-conv).
# Uses v4logk CondRes (trained at 64^3 but fully-conv so runs on 128^3).
import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, '/root/21cm_gen')
from dataset import T21Dataset, load_mat
from models.vae import VAE3D
from ldm_unet import LDMUNet3D
from train_ldm import heun_sample
from train_cond_residual import CondResidualUNet


def hann3d(size):
    w = np.hanning(size)
    return (w[:, None, None] * w[None, :, None] * w[None, None, :]).astype(np.float32)


@torch.no_grad()
def generate_full_field(vae, ldm, res, cfg, lm, ls,
                        ic_delta_full, ic_vbv_full,
                        params, redshift,
                        patch=128, stride=64, batch=2,
                        steps=30, cfg_ic=1.0, cfg_params=1.0,
                        null_param=None, device='cuda'):
    N = ic_delta_full.shape[-1]
    out  = np.zeros((N, N, N), dtype=np.float32)
    wsum = np.zeros((N, N, N), dtype=np.float32)
    win  = hann3d(patch)

    origins = []
    coords = list(range(0, N - patch + 1, stride))
    if coords[-1] != N - patch:
        coords.append(N - patch)
    for i in coords:
        for j in coords:
            for k in coords:
                origins.append((i, j, k))

    for s in range(0, len(origins), batch):
        chunk = origins[s:s + batch]
        B = len(chunk)
        ic_d = torch.empty(B, 1, patch, patch, patch, device=device)
        ic_v = torch.empty(B, 1, patch, patch, patch, device=device)
        for b, (i, j, k) in enumerate(chunk):
            ic_d[b, 0] = ic_delta_full[i:i+patch, j:j+patch, k:k+patch].to(device)
            ic_v[b, 0] = ic_vbv_full  [i:i+patch, j:j+patch, k:k+patch].to(device)
        par4 = params[None, :4].expand(B, -1).to(device)
        zred = torch.full((B,), float(redshift), device=device)

        # latent shape: from cfg (vae_v5_for_128 has [8, 32, 32, 32])
        z = heun_sample(ldm, B, tuple(cfg['latent_shape']),
                        ic_d, ic_v, par4, zred,
                        sigma_data=1.121, num_steps=steps,
                        cfg_ic=cfg_ic, cfg_params=cfg_params,
                        null_param=null_param, device=device)
        x_low = vae.decoder(z * ls + lm)
        if res is not None:
            r_hat = res(x_low, ic_d, ic_v, par4, zred)
            final = (x_low + r_hat).cpu().numpy()
        else:
            final = x_low.cpu().numpy()

        for b, (i, j, k) in enumerate(chunk):
            out [i:i+patch, j:j+patch, k:k+patch] += final[b, 0] * win
            wsum[i:i+patch, j:j+patch, k:k+patch] += win
    return out / np.clip(wsum, 1e-8, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt', required=True)
    p.add_argument('--ldm_ckpt', required=True)
    p.add_argument('--res_ckpt', default='', help='empty = no residual')
    p.add_argument('--data_root', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--out_dir',   required=True)
    p.add_argument('--split',     default='test')
    p.add_argument('--redshift',  type=int, default=10)
    p.add_argument('--n_cubes',   type=int, default=10)
    p.add_argument('--batch',     type=int, default=2)
    p.add_argument('--patch',     type=int, default=128)
    p.add_argument('--stride',    type=int, default=64)
    p.add_argument('--steps',     type=int, default=30)
    p.add_argument('--seed',      type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda'
    torch.manual_seed(args.seed)

    vae_ck = torch.load(args.vae_ckpt, map_location=dev)
    cfg = vae_ck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
                base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(dev).eval()
    vae.load_state_dict(vae_ck['model'])
    lm = vae_ck['latent_mean'].view(1, -1, 1, 1, 1).to(dev)
    ls = vae_ck['latent_std' ].view(1, -1, 1, 1, 1).to(dev)
    print(f"VAE latent_shape: {cfg['latent_shape']}")

    ldm = LDMUNet3D(latent_ch=cfg['latent_ch'],
                    ch_mults=(1, 2), attn_levels=(0, 1),
                    ic_stem_downsamples=2).to(dev).eval()
    ldm.load_state_dict(torch.load(args.ldm_ckpt, map_location=dev)['model'])

    res = None
    if args.res_ckpt:
        res = CondResidualUNet(base_ch=32).to(dev).eval()
        res.load_state_dict(torch.load(args.res_ckpt, map_location=dev)['model'])
        print(f"using residual: {args.res_ckpt}")

    ds = T21Dataset(args.data_root, args.patch, redshifts=[args.redshift], split=args.split)
    print(f"{args.split} files: {len(ds.files)}; using {args.n_cubes}")

    ic_d_path = os.path.join(ds.ic_dir, [f for f in os.listdir(ds.ic_dir) if 'delta' in f.lower()][0])
    ic_v_path = os.path.join(ds.ic_dir, [f for f in os.listdir(ds.ic_dir) if 'vbv'   in f.lower()][0])
    ic_d_arr = load_mat(ic_d_path); ic_v_arr = load_mat(ic_v_path)
    ic_d_arr = (ic_d_arr - ic_d_arr.mean()) / (ic_d_arr.std() + 1e-8)
    ic_v_arr = (ic_v_arr - ic_v_arr.mean()) / (ic_v_arr.std() + 1e-8)
    ic_d_t = torch.from_numpy(ic_d_arr.astype(np.float32))
    ic_v_t = torch.from_numpy(ic_v_arr.astype(np.float32))

    t_total = time.time()
    for ci, fname in enumerate(ds.files[:args.n_cubes]):
        t0 = time.time()
        true_full = load_mat(os.path.join(ds.t21_dir, fname))
        true_full = (true_full - ds.t21_mean) / ds.t21_std
        par_vec = ds.param_vector(fname).to(dev)
        params = par_vec[:4]; redshift = par_vec[4].item()

        gen_full = generate_full_field(
            vae, ldm, res, cfg, lm, ls,
            ic_d_t, ic_v_t, params, redshift,
            patch=args.patch, stride=args.stride, batch=args.batch,
            steps=args.steps, device=dev)

        dt = time.time() - t0
        rel = float(((true_full - gen_full)**2).sum() / (true_full**2).sum())
        print(f"  [{ci+1}/{args.n_cubes}] {fname} | {dt:.1f}s | rel_MSE={rel:.4f}", flush=True)

        np.savez(os.path.join(args.out_dir, f'{ci:03d}_{fname.replace(".mat","")}.npz'),
                 true_norm=true_full.astype(np.float32),
                 gen_norm =gen_full.astype(np.float32),
                 params   =par_vec.cpu().numpy(),
                 t21_mean =np.float32(ds.t21_mean),
                 t21_std  =np.float32(ds.t21_std),
                 fname    =fname)

    print(f"\nTotal: {time.time()-t_total:.1f}s for {args.n_cubes} cubes")


if __name__ == '__main__':
    main()
