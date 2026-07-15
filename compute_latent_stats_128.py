# Compute latent_mean / latent_std at 32^3 latent (from 128^3 patches with v5 VAE).
# v5 VAE is fully-conv so works directly on 128^3 -> 32^3 latent.
import os, sys, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, '/root/21cm_gen')
from dataset import T21Dataset
from models.vae import VAE3D


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt', required=True)
    p.add_argument('--patch_size', type=int, default=128)
    p.add_argument('--n_samples', type=int, default=200,
                   help='number of patches to use for statistics')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    dev = 'cuda'

    ck = torch.load(args.vae_ckpt, map_location=dev)
    cfg = ck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
                base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(dev).eval()
    vae.load_state_dict(ck['model'])
    print(f'loaded VAE: latent_ch={cfg["latent_ch"]}  ch_mults={cfg["ch_mults"]}')

    def mk_ds(root, split='train'):
        return T21Dataset(root, args.patch_size, redshifts=[8,9,10,11,12],
                          split=split, load_ic=False, max_per_z=50, primary_z=10,
                          holdout_frac=0.25, patches_per_cube=1)
    sets = []
    for r in ['/root/autodl-tmp/ASR21cm/varying_IC',
              '/root/autodl-tmp/ASR21cm/varying_astro']:
        try: sets.append(mk_ds(r, 'train'))
        except Exception as e: print('skip', r, e)
    ds = ConcatDataset(sets)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4)
    print(f'dataset size: {len(ds)}')

    # streaming mean/std over latent channels, spatial dims kept
    n_seen = 0
    sum_c   = torch.zeros(cfg['latent_ch'], device=dev)
    sum_c2  = torch.zeros(cfg['latent_ch'], device=dev)
    n_vox   = 0

    t0 = time.time()
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= args.n_samples:
                break
            x = b['patch'].to(dev, non_blocking=True)
            mu, _ = vae.encoder(x)  # (1, C, 32, 32, 32)
            mu = mu.float()
            # accumulate over (batch, spatial)
            sum_c  += mu.sum (dim=(0, 2, 3, 4))
            sum_c2 += mu.pow(2).sum(dim=(0, 2, 3, 4))
            n_vox  += mu.shape[0] * mu.shape[2] * mu.shape[3] * mu.shape[4]
            n_seen += 1
            if (i+1) % 20 == 0:
                print(f'  [{i+1}/{args.n_samples}] {(time.time()-t0):.1f}s')

    mean = (sum_c  / n_vox).cpu()
    var  = (sum_c2 / n_vox).cpu() - mean.pow(2)
    std  = var.clamp(min=0).sqrt() + 1e-6

    print('latent_mean:', mean.numpy())
    print('latent_std :', std.numpy())
    torch.save({'latent_mean': mean, 'latent_std': std,
                'n_samples': n_seen, 'n_vox': n_vox,
                'vae_ckpt': args.vae_ckpt, 'patch_size': args.patch_size,
                'latent_shape': list(mu.shape[1:])}, args.out)
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
