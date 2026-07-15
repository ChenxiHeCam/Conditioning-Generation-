"""
Unified 256^3 full-field generation for EVERY model family, one CLI.

  --model latent   : VAE + LDM (any compression; cfg read from VAE ckpt)
  --model pixel    : pixel-space diffusion (train_pixel_diffusion.py ckpt)
  --model vqgan_ar : VQGAN + AR transformer (two ckpts)
  --model stylegan : conditional StyleGAN (EMA weights)

Same Hann-blended tiling (patch 64, stride 32) for all -> outputs directly
comparable. Saves per-cube npz {true_norm, gen_norm, params, ...} plus
gen_seconds (wall-clock per cube) for the speed/quality trade-off figure.
"""
import os, argparse, time, json
import numpy as np
import torch

from dataset import T21Dataset, load_mat
from train_ldm import heun_sample, denoise, edm_precond


def hann3d(size):
    w = np.hanning(size)
    return (w[:, None, None] * w[None, :, None] * w[None, None, :]).astype(np.float32)


@torch.no_grad()
def tile_generate(gen_tile, ic_delta_full, ic_vbv_full, params, redshift,
                  patch=64, stride=32, batch=16, device='cuda'):
    N = ic_delta_full.shape[-1]
    out  = np.zeros((N, N, N), dtype=np.float32)
    wsum = np.zeros((N, N, N), dtype=np.float32)
    win  = hann3d(patch)
    coords = list(range(0, N - patch + 1, stride))
    if coords[-1] != N - patch:
        coords.append(N - patch)
    origins = [(i, j, k) for i in coords for j in coords for k in coords]
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
        final = gen_tile(ic_d, ic_v, par4, zred).float().cpu().numpy()
        for b, (i, j, k) in enumerate(chunk):
            sl = np.s_[i:i+patch, j:j+patch, k:k+patch]
            out[sl] += final[b, 0] * win
            wsum[sl] += win
    return out / np.clip(wsum, 1e-8, None)


def build_latent(args, dev):
    from models.vae import VAE3D
    from ldm_unet import LDMUNet3D
    ck = torch.load(args.vae_ckpt, map_location=dev)
    cfg = ck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'], base_ch=cfg['base_ch'],
                ch_mults=tuple(cfg['ch_mults'])).to(dev).eval()
    vae.load_state_dict(ck['model'])
    lm = ck['latent_mean'].view(1, -1, 1, 1, 1).to(dev)
    ls = ck['latent_std' ].view(1, -1, 1, 1, 1).to(dev)
    ldm = LDMUNet3D(latent_ch=cfg['latent_ch'], ch_mults=(1, 2),
                    attn_levels=(0, 1), ic_stem_downsamples=2).to(dev).eval()
    lck = torch.load(args.ldm_ckpt, map_location=dev)
    key = f'ema_{args.ema}' if args.ema and f'ema_{args.ema}' in lck else 'model'
    ldm.load_state_dict(lck[key])
    sd = lck.get('sigma_data', 1.121)
    def gen(ic_d, ic_v, par4, zred):
        z = heun_sample(ldm, ic_d.shape[0], tuple(cfg['latent_shape']),
                        ic_d, ic_v, par4, zred, sigma_data=sd,
                        num_steps=args.steps, device=ic_d.device)
        return vae.decoder(z * ls + lm)
    return gen


def build_pixel(args, dev):
    from ldm_unet import LDMUNet3D
    ck = torch.load(args.pixel_ckpt, map_location=dev)
    cfg = ck['model_config']
    model = LDMUNet3D(latent_ch=1, ic_stem_downsamples=0,
                      base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults']),
                      attn_levels=tuple(cfg['attn_levels'])).to(dev).eval()
    key = f'ema_{args.ema}' if args.ema and f'ema_{args.ema}' in ck else 'model'
    model.load_state_dict(ck[key])
    sd = ck['sigma_data']
    p = cfg.get('patch_size', 64)
    def gen(ic_d, ic_v, par4, zred):
        return heun_sample(model, ic_d.shape[0], (1, p, p, p),
                           ic_d, ic_v, par4, zred, sigma_data=sd,
                           num_steps=args.steps, device=ic_d.device)
    return gen


def build_vqgan_ar(args, dev):
    from models.vqgan3d import VQGAN3D
    from models.ar_transformer import ARTransformer3D
    vck = torch.load(args.vqgan_ckpt, map_location=dev)
    vcfg = vck['model_config']
    vq = VQGAN3D(in_ch=1, embed_dim=vcfg['embed_dim'], n_embed=vcfg['n_embed'],
                 base_ch=vcfg['base_ch']).to(dev).eval()
    vq.load_state_dict(vck['G'])
    ack = torch.load(args.ar_ckpt, map_location=dev)
    acfg = ack['model_config']
    ar = ARTransformer3D(n_embed=acfg['n_embed'], dim=acfg['dim'],
                         depth=acfg['depth'], heads=acfg['heads']).to(dev).eval()
    ar.load_state_dict(ack['model'])
    def gen(ic_d, ic_v, par4, zred):
        idx = ar.sample(ic_d, ic_v, par4, zred,
                        temperature=args.temperature, top_k=args.top_k)
        return vq.decode_indices(idx)
    return gen


def build_stylegan(args, dev):
    from models.stylegan3d import StyleGenerator3D
    ck = torch.load(args.gan_ckpt, map_location=dev)
    G = StyleGenerator3D(base_ch=ck['model_config']['base_ch']).to(dev).eval()
    G.load_state_dict(ck.get('ema', ck['G']))
    def gen(ic_d, ic_v, par4, zred):
        return G(ic_d, ic_v, par4, zred)
    return gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   choices=['latent', 'pixel', 'vqgan_ar', 'stylegan'])
    p.add_argument('--vae_ckpt'); p.add_argument('--ldm_ckpt')
    p.add_argument('--pixel_ckpt'); p.add_argument('--vqgan_ckpt')
    p.add_argument('--ar_ckpt'); p.add_argument('--gan_ckpt')
    p.add_argument('--ema', default='0.9999')
    p.add_argument('--data_root', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--redshift', type=int, default=10)
    p.add_argument('--n_cubes', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda'
    torch.manual_seed(args.seed)

    gen_tile = {'latent': build_latent, 'pixel': build_pixel,
                'vqgan_ar': build_vqgan_ar, 'stylegan': build_stylegan}[args.model](args, dev)

    ds = T21Dataset(args.data_root, 64, redshifts=[args.redshift], split=args.split)
    files = ds.files[:args.n_cubes]
    print(f"model={args.model}  {args.split} files: {len(ds.files)}; using {len(files)}")

    ic_d_path = os.path.join(ds.ic_dir, [f for f in os.listdir(ds.ic_dir) if 'delta' in f.lower()][0])
    ic_v_path = os.path.join(ds.ic_dir, [f for f in os.listdir(ds.ic_dir) if 'vbv'   in f.lower()][0])
    ic_d_arr = load_mat(ic_d_path); ic_v_arr = load_mat(ic_v_path)
    ic_d_arr = (ic_d_arr - ic_d_arr.mean()) / (ic_d_arr.std() + 1e-8)
    ic_v_arr = (ic_v_arr - ic_v_arr.mean()) / (ic_v_arr.std() + 1e-8)
    ic_d_t = torch.from_numpy(ic_d_arr.astype(np.float32))
    ic_v_t = torch.from_numpy(ic_v_arr.astype(np.float32))

    times = []
    for ci, fname in enumerate(files):
        t0 = time.time()
        true_full = load_mat(os.path.join(ds.t21_dir, fname))
        true_full = (true_full - ds.t21_mean) / ds.t21_std
        par_vec = ds.param_vector(fname).to(dev)
        gen_full = tile_generate(gen_tile, ic_d_t, ic_v_t,
                                 par_vec[:4], par_vec[4].item(),
                                 batch=args.batch, device=dev)
        dt = time.time() - t0
        times.append(dt)
        rel = float(((true_full - gen_full)**2).sum() / (true_full**2).sum())
        print(f"  [{ci+1}/{len(files)}] {fname} | {dt:.1f}s | rel_MSE={rel:.4f}", flush=True)
        np.savez(os.path.join(args.out_dir, f'{ci:03d}_{fname.replace(".mat","")}.npz'),
                 true_norm=true_full.astype(np.float32),
                 gen_norm=gen_full.astype(np.float32),
                 params=par_vec.cpu().numpy(),
                 t21_mean=np.float32(ds.t21_mean), t21_std=np.float32(ds.t21_std),
                 gen_seconds=np.float32(dt), fname=fname)

    with open(os.path.join(args.out_dir, 'timing.json'), 'w') as f:
        json.dump(dict(model=args.model, steps=args.steps,
                       mean_s=float(np.mean(times)), median_s=float(np.median(times)),
                       n=len(times)), f, indent=2)
    print(f"median {np.median(times):.1f}s / cube")


if __name__ == '__main__':
    main()
