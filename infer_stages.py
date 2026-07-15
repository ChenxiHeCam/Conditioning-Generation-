# Stage-decomposed full-field generation for ablation figures.
# For each test cube, saves FOUR aligned 256^3 fields in one npz:
#   true_norm            ground truth (z-score units)
#   vae_only             VAE encode->decode of the TRUE field (compression floor)
#   ldm_vae              LDM sample -> VAE decode, NO residual
#   ldm_vae_res_<tag>    x_low + r_hat for every --res_ckpt tag:path given
# The LDM latent draw is shared across all residual variants so the
# residual ablation is exact (same x_low input).
import os, sys, argparse, time
import numpy as np
import torch

from dataset import T21Dataset, load_mat
from models.vae import VAE3D
from ldm_unet import LDMUNet3D
from train_ldm import heun_sample
from train_cond_residual import CondResidualUNet


def hann3d(size):
    w = np.hanning(size)
    return (w[:, None, None] * w[None, :, None] * w[None, None, :]).astype(np.float32)


@torch.no_grad()
def generate_stages(vae, ldm, res_models, cfg, lm, ls,
                    true_full, ic_delta_full, ic_vbv_full,
                    params, redshift,
                    patch=64, stride=32, batch=16,
                    steps=30, device='cuda'):
    """Returns dict of stage name -> stitched (N,N,N) float32 field."""
    N = ic_delta_full.shape[-1]
    win = hann3d(patch)
    stages = ['vae_only', 'ldm_vae'] + [f'ldm_vae_res_{t}' for t in res_models]
    out  = {s: np.zeros((N, N, N), dtype=np.float32) for s in stages}
    wsum = np.zeros((N, N, N), dtype=np.float32)

    coords = list(range(0, N - patch + 1, stride))
    if coords[-1] != N - patch:
        coords.append(N - patch)
    origins = [(i, j, k) for i in coords for j in coords for k in coords]

    for s0 in range(0, len(origins), batch):
        chunk = origins[s0:s0 + batch]
        B = len(chunk)
        ic_d = torch.empty(B, 1, patch, patch, patch, device=device)
        ic_v = torch.empty(B, 1, patch, patch, patch, device=device)
        x_tr = torch.empty(B, 1, patch, patch, patch, device=device)
        for b, (i, j, k) in enumerate(chunk):
            ic_d[b, 0] = ic_delta_full[i:i+patch, j:j+patch, k:k+patch].to(device)
            ic_v[b, 0] = ic_vbv_full  [i:i+patch, j:j+patch, k:k+patch].to(device)
            x_tr[b, 0] = true_full    [i:i+patch, j:j+patch, k:k+patch].to(device)
        par4 = params[None, :4].expand(B, -1).to(device)
        zred = torch.full((B,), float(redshift), device=device)

        # stage 1: VAE encode->decode of the true patch (no latent norm)
        mu, _ = vae.encoder(x_tr)
        rec = vae.decoder(mu).cpu().numpy()

        # stage 2: LDM sample -> decode (single shared draw)
        z = heun_sample(ldm, B, tuple(cfg['latent_shape']),
                        ic_d, ic_v, par4, zred,
                        sigma_data=1.121, num_steps=steps, device=device)
        x_low = vae.decoder(z * ls + lm)
        ldm_np = x_low.cpu().numpy()

        # stage 3: + each residual variant, reusing the same x_low
        res_np = {}
        for tag, res in res_models.items():
            r_hat = res(x_low, ic_d, ic_v, par4, zred)
            res_np[tag] = (x_low + r_hat).cpu().numpy()

        for b, (i, j, k) in enumerate(chunk):
            sl = np.s_[i:i+patch, j:j+patch, k:k+patch]
            out['vae_only'][sl] += rec   [b, 0] * win
            out['ldm_vae'][sl]  += ldm_np[b, 0] * win
            for tag in res_models:
                out[f'ldm_vae_res_{tag}'][sl] += res_np[tag][b, 0] * win
            wsum[sl] += win

    wsum = np.clip(wsum, 1e-8, None)
    return {s: v / wsum for s, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt', required=True)
    p.add_argument('--ldm_ckpt', required=True)
    p.add_argument('--res_ckpts', nargs='+', required=True,
                   help='tag:path pairs, e.g. v4logk:/path/a.pt v2:/path/b.pt')
    p.add_argument('--data_root', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--out_dir',   required=True)
    p.add_argument('--split',     default='test')
    p.add_argument('--redshift',  type=int, default=10)
    p.add_argument('--n_cubes',   type=int, default=100)
    p.add_argument('--batch',     type=int, default=16)
    p.add_argument('--stride',    type=int, default=32)
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

    ldm = LDMUNet3D(latent_ch=cfg['latent_ch'],
                    ch_mults=(1, 2), attn_levels=(0, 1),
                    ic_stem_downsamples=2).to(dev).eval()
    ldm.load_state_dict(torch.load(args.ldm_ckpt, map_location=dev)['model'])

    res_models = {}
    for spec in args.res_ckpts:
        tag, path = spec.split(':', 1)
        m = CondResidualUNet(base_ch=32).to(dev).eval()
        m.load_state_dict(torch.load(path, map_location=dev)['model'])
        res_models[tag] = m
        print(f"residual [{tag}] <- {path}")

    ds = T21Dataset(args.data_root, 64, redshifts=[args.redshift], split=args.split)
    file_list = ds.files[:args.n_cubes]
    print(f"{args.split} files: {len(ds.files)}; using {len(file_list)}")

    ic_d_path = os.path.join(ds.ic_dir, [f for f in os.listdir(ds.ic_dir) if 'delta' in f.lower()][0])
    ic_v_path = os.path.join(ds.ic_dir, [f for f in os.listdir(ds.ic_dir) if 'vbv'   in f.lower()][0])
    ic_d_arr = load_mat(ic_d_path); ic_v_arr = load_mat(ic_v_path)
    ic_d_arr = (ic_d_arr - ic_d_arr.mean()) / (ic_d_arr.std() + 1e-8)
    ic_v_arr = (ic_v_arr - ic_v_arr.mean()) / (ic_v_arr.std() + 1e-8)
    ic_d_t = torch.from_numpy(ic_d_arr.astype(np.float32))
    ic_v_t = torch.from_numpy(ic_v_arr.astype(np.float32))

    t_total = time.time()
    for ci, fname in enumerate(file_list):
        t0 = time.time()
        true_full = load_mat(os.path.join(ds.t21_dir, fname))
        true_full = (true_full - ds.t21_mean) / ds.t21_std
        true_t = torch.from_numpy(true_full.astype(np.float32))
        par_vec = ds.param_vector(fname).to(dev)

        fields = generate_stages(
            vae, ldm, res_models, cfg, lm, ls,
            true_t, ic_d_t, ic_v_t, par_vec[:4], par_vec[4].item(),
            patch=64, stride=args.stride, batch=args.batch,
            steps=args.steps, device=dev)

        dt = time.time() - t0
        rels = {s: float(((true_full - f)**2).sum() / (true_full**2).sum())
                for s, f in fields.items()}
        print(f"  [{ci+1}/{len(file_list)}] {fname} | {dt:.1f}s | " +
              "  ".join(f"{s}={r:.4f}" for s, r in rels.items()), flush=True)

        np.savez(os.path.join(args.out_dir, f'{ci:03d}_{fname.replace(".mat","")}.npz'),
                 true_norm=true_full.astype(np.float32),
                 **{s: f.astype(np.float32) for s, f in fields.items()},
                 params=par_vec.cpu().numpy(),
                 t21_mean=np.float32(ds.t21_mean),
                 t21_std =np.float32(ds.t21_std),
                 fname=fname)

    print(f"\nTotal: {time.time()-t_total:.1f}s for {len(file_list)} cubes")


if __name__ == '__main__':
    main()
