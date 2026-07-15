# Quick eval of VAE 128 ep151: encode a few test cubes at 128^3, compare PS ratio vs original 64^3 VAE.
import os, sys, glob, json, math
import numpy as np
import torch

sys.path.insert(0, '/root/21cm_gen')
from dataset import T21Dataset, load_mat
from models.vae import VAE3D


def radial_ps(x, Lpix=3.0, n_bins=12, k_min=0.05):
    x = x - x.mean()
    N = x.shape[-1]
    F = np.fft.rfftn(x)
    P = (F.real**2 + F.imag**2) / N**3
    k1  = np.fft.fftfreq(N)  * N
    k1r = np.fft.rfftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    Kphys = Kmag * (2*math.pi/(N*Lpix))
    k_nyq = math.pi/Lpix
    edges = np.logspace(math.log10(k_min), math.log10(k_nyq*0.95), n_bins+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    P_b = np.zeros(n_bins)
    for i in range(n_bins):
        m = (Kphys >= edges[i]) & (Kphys < edges[i+1])
        P_b[i] = P[m].mean() if m.any() else np.nan
    return centers, P_b


def eval_vae(vae_ckpt, patch_size, n_cubes=4, device='cuda', label=''):
    ck = torch.load(vae_ckpt, map_location=device)
    cfg = ck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
                base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(device).eval()
    vae.load_state_dict(ck['model'])
    ds = T21Dataset('/root/autodl-tmp/ASR21cm/varying_astro',
                    patch_size, redshifts=[10], split='test', load_ic=False,
                    max_per_z=n_cubes, primary_z=10,
                    holdout_frac=0.25, patches_per_cube=1)
    # gather a few cubes — sample full 256^3 cubes, take centered patch_size^3
    files = ds.files[:n_cubes]
    rats = []
    pss_t = []
    pss_g = []
    for f in files:
        full = load_mat(os.path.join(ds.t21_dir, f))
        full = (full - ds.t21_mean) / ds.t21_std
        # central crop patch_size
        h = (full.shape[0] - patch_size) // 2
        x = full[h:h+patch_size, h:h+patch_size, h:h+patch_size]
        xt = torch.from_numpy(x.astype(np.float32))[None,None].to(device)
        with torch.no_grad():
            mu, lv = vae.encoder(xt)
            rec = vae.decoder(mu)
        rec_np = rec[0,0].cpu().numpy()
        kc, Pt = radial_ps(x)
        _ , Pg = radial_ps(rec_np)
        rats.append(Pg / np.where(Pt>0, Pt, np.nan))
        pss_t.append(Pt); pss_g.append(Pg)
    rats = np.stack(rats)
    med = np.nanmedian(rats, axis=0)
    print(f"\n=== {label} (patch {patch_size}^3, n={n_cubes} cubes, central crop) ===")
    print("  k [cMpc^-1] | PS ratio median (Pg/Pt), per k-bin:")
    for k, r in zip(kc, med):
        bar = '#' * max(0, int((r - 1) * 20))
        print(f"  {k:8.4f}   {r:.3f}  {bar}")
    # large/mid/small
    m_l = kc < 0.15; m_m = (kc >= 0.15) & (kc < 0.40); m_s = kc >= 0.40
    print(f"  Bands:  large={np.nanmean(med[m_l]):.3f}  "
          f"mid={np.nanmean(med[m_m]):.3f}  small={np.nanmean(med[m_s]):.3f}")


if __name__ == '__main__':
    # baseline: 64^3 patches with original v5 final
    eval_vae('/root/autodl-tmp/checkpoints/vae_v5/vae_v5_final.pt', 64, label='v5 baseline')
    # candidate: 128^3 patches with v5_128 ep151 (1 ep finetune)
    eval_vae('/root/autodl-tmp/checkpoints/vae_v5_128/vae_epoch0151.pt', 128, label='v5_128 ep151 (1 ep finetune)')
