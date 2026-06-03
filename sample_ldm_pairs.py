# Sample LDM K times per (IC, params, z_red) on train+val splits.
# Save (x_low_ldm, real_x, ic_d, ic_v, params5) tuples for cond-residual mode-B training.
# DOES NOT touch test split.
import os, sys, time, argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, "/root/21cm_gen")
from dataset import T21Dataset
from models.vae import VAE3D
from ldm_unet import LDMUNet3D
from train_ldm import heun_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ldm_ckpt', required=True)
    p.add_argument('--vae_ckpt', required=True)
    p.add_argument('--out_path', required=True, help='where to write the sample tensor file')
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--K', type=int, default=2, help='LDM samples per patch')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_steps', type=int, default=18)
    p.add_argument('--max_per_z', type=int, default=50)
    p.add_argument('--primary_z', type=int, default=10)
    p.add_argument('--holdout_frac', type=float, default=0.25)
    p.add_argument('--patches_per_cube', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--val_frac', type=float, default=1.0,
                   help='fraction of val files to include in sampling (rest reserved as clean val)')
    args = p.parse_args()

    dev = 'cuda'

    # ---- frozen VAE ----
    vae_ck = torch.load(args.vae_ckpt, map_location=dev)
    cfg = vae_ck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
                base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(dev).eval()
    vae.load_state_dict(vae_ck['model'])
    latent_mean = vae_ck['latent_mean'].view(1, -1, 1, 1, 1).to(dev)
    latent_std  = vae_ck['latent_std' ].view(1, -1, 1, 1, 1).to(dev)

    # ---- LDM ----
    ldm_ck = torch.load(args.ldm_ckpt, map_location=dev)
    ldm = LDMUNet3D(latent_ch=cfg['latent_ch'],
                    ch_mults=(1, 2), attn_levels=(0, 1), ic_stem_downsamples=2).to(dev).eval()
    ldm.load_state_dict(ldm_ck['model'])
    sigma_data = 1.121
    print(f"LDM ep{ldm_ck.get('epoch','?')} loaded, sigma_data={sigma_data}")

    # ---- data ----
    def mk_ds(root, split):
        ds = T21Dataset(root, 64, redshifts=args.redshifts, split=split,
                          load_ic=True, max_per_z=args.max_per_z,
                          primary_z=args.primary_z, holdout_frac=args.holdout_frac,
                          patches_per_cube=args.patches_per_cube)
        if split == 'val' and args.val_frac < 1.0:
            n = int(len(ds.files) * args.val_frac)
            ds.files = ds.files[:n]
            print(f'  val_frac={args.val_frac}: using first {n}/{len(ds.files) if not n else int(len(ds.files)/args.val_frac)} val files')
        return ds
    sets = []
    for split in ['train', 'val']:
        for root in [args.data_root_ic, args.data_root_astro]:
            try:
                sets.append(mk_ds(root, split))
            except Exception as e:
                print(f"skip {root} {split}: {e}")
    ds = ConcatDataset(sets)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    n_patches = len(ds)
    total_samples = n_patches * args.K
    print(f"patches: {n_patches}  K={args.K}  total LDM samples: {total_samples}")

    # ---- sample ----
    x_low_ldm_all = torch.zeros(total_samples, 1, 64, 64, 64, dtype=torch.float16)
    x_real_all    = torch.zeros(total_samples, 1, 64, 64, 64, dtype=torch.float16)
    ic_d_all      = torch.zeros(total_samples, 1, 64, 64, 64, dtype=torch.float16)
    ic_v_all      = torch.zeros(total_samples, 1, 64, 64, 64, dtype=torch.float16)
    par_all       = torch.zeros(total_samples, 5, dtype=torch.float32)

    idx = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, b in enumerate(loader):
            x   = b['patch'].to(dev)            # (B,1,64,64,64) z-score normalized
            icd = b['ic_delta'].to(dev)
            icv = b['ic_vbv'].to(dev)
            par = b['params'].to(dev)
            params4 = par[:, :4]
            zred    = par[:, 4]
            B = x.shape[0]

            for k in range(args.K):
                z = heun_sample(ldm, B, tuple(cfg['latent_shape']),
                                icd, icv, params4, zred,
                                sigma_data=sigma_data, num_steps=args.num_steps,
                                cfg_ic=1.0, cfg_params=1.0,
                                null_param=None, device=dev)
                xl = vae.decoder(z * latent_std + latent_mean)  # (B,1,64,64,64)
                # store
                end = idx + B
                x_low_ldm_all[idx:end] = xl.cpu().half()
                x_real_all   [idx:end] = x.cpu().half()
                ic_d_all     [idx:end] = icd.cpu().half()
                ic_v_all     [idx:end] = icv.cpu().half()
                par_all      [idx:end] = par.cpu()
                idx = end

            if (bi + 1) % 20 == 0:
                dt = time.time() - t0
                samples_done = idx
                eta = dt / samples_done * (total_samples - samples_done)
                print(f"  batch {bi+1}/{len(loader)}  samples {samples_done}/{total_samples}  "
                      f"elapsed {dt:.0f}s  ETA {eta:.0f}s  ({samples_done/dt:.2f} sample/s)")

    elapsed = time.time() - t0
    print(f"\nTotal: {idx} samples in {elapsed:.1f}s  =  {elapsed/60:.2f} min  ({idx/elapsed:.2f} sample/s)")

    # ---- save ----
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(dict(
        x_low_ldm=x_low_ldm_all[:idx],
        x_real   =x_real_all[:idx],
        ic_d     =ic_d_all[:idx],
        ic_v     =ic_v_all[:idx],
        par      =par_all[:idx],
        meta=dict(ldm_ckpt=args.ldm_ckpt, K=args.K, num_steps=args.num_steps,
                  n_patches=n_patches, sigma_data=sigma_data),
    ), args.out_path)
    print(f"\nSaved to {args.out_path}  ({os.path.getsize(args.out_path)/1e9:.2f} GB)")


if __name__ == '__main__':
    main()
