"""
Stage 2 LDM training: EDM denoiser on the v4 latent.

Loss: standard EDM (Karras 2022) v-prediction with preconditioning.
Sampling: deterministic Heun 2nd-order, 32 steps.
CFG: independent dropout of IC channels (p=0.10) and params/redshift (p=0.10),
     joint drop (p=0.05) — compositional guidance at inference.
"""
import os, argparse, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from models.vae import VAE3D
from ldm_dataset import build_train_dataset, make_balanced_sampler
from ldm_unet   import LDMUNet3D
from utils.power_spectrum import power_spectrum


# ---------------------------------------------------------------------------
# v2 ADDITIONS: moment-matching + latent spectral losses
# ---------------------------------------------------------------------------

def moment_match_loss(x_pred, x_target, eps=1e-6):
    """Match (std, skew, kurt) per channel between predicted and target latents."""
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


def latent_spectral_loss(x_pred, x_target):
    """Log-magnitude MSE in 3D Fourier space at the latent level."""
    fr = torch.fft.rfftn(x_pred.float(), dim=(-3,-2,-1), norm='ortho')
    ft = torch.fft.rfftn(x_target.float(), dim=(-3,-2,-1), norm='ortho')
    return F.mse_loss(fr.abs().log1p(), ft.abs().log1p())


# ---------------------------------------------------------------------------
# EDM preconditioning (Karras 2022 Appendix B.6)
# ---------------------------------------------------------------------------

def edm_precond(sigma, sigma_data):
    sigma2 = sigma ** 2
    sd2 = sigma_data ** 2
    c_skip = sd2 / (sigma2 + sd2)
    c_out  = sigma * sigma_data / (sigma2 + sd2).sqrt()
    c_in   = 1.0 / (sigma2 + sd2).sqrt()
    c_noise = sigma                            # passed directly to model
    return c_skip, c_out, c_in, c_noise


def denoise(model, x, sigma, sigma_data, ic_delta, ic_vbv, params, redshift):
    """Apply EDM preconditioning. Returns the predicted clean latent."""
    sigma = sigma.view(-1, 1, 1, 1, 1)
    c_skip, c_out, c_in, c_noise = edm_precond(sigma, sigma_data)
    model_out = model(c_in * x, ic_delta, ic_vbv,
                      params, redshift, c_noise.flatten())
    return c_skip * x + c_out * model_out


def edm_loss(model, x_clean, ic_delta, ic_vbv, params, redshift,
             sigma_data, P_mean, P_std, drop_ic, drop_params, drop_both,
             null_param):
    B = x_clean.shape[0]
    device = x_clean.device

    # Sample log-normal sigma
    rnd = torch.randn(B, device=device)
    sigma = (P_mean + P_std * rnd).exp()

    # Sample noise
    n = torch.randn_like(x_clean) * sigma.view(-1, 1, 1, 1, 1)
    x_noisy = x_clean + n

    # CFG dropout — three independent draws
    u = torch.rand(B, device=device)
    mask_both    = u < drop_both
    mask_ic_only = (u >= drop_both) & (u < drop_both + drop_ic)
    mask_p_only  = (u >= drop_both + drop_ic) & (u < drop_both + drop_ic + drop_params)

    if mask_both.any() or mask_ic_only.any():
        m = (mask_both | mask_ic_only).view(-1, 1, 1, 1, 1).float()
        ic_delta = ic_delta * (1 - m)
        ic_vbv   = ic_vbv   * (1 - m)
    if mask_both.any() or mask_p_only.any():
        m = (mask_both | mask_p_only).view(-1, 1).float()
        params = params * (1 - m) + null_param * m
        # redshift: replace with mean redshift 10 when dropped
        redshift = redshift * (1 - m.squeeze(-1)) + 10.0 * m.squeeze(-1)

    # Forward through preconditioned denoiser
    x_hat = denoise(model, x_noisy, sigma, sigma_data,
                    ic_delta, ic_vbv, params, redshift)

    # EDM loss weighting (Karras 2022 eq. 8): w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
    w = ((sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2)
    w = w.view(-1, 1, 1, 1, 1)
    edm = (w * (x_hat - x_clean) ** 2).mean()

    # v2 additions (conservative weights)
    mm  = moment_match_loss(x_hat, x_clean)
    spec = latent_spectral_loss(x_hat, x_clean)

    loss = edm + 0.1 * mm + 0.05 * spec
    return loss


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(v)


# ---------------------------------------------------------------------------
# Sampling (Heun 2nd-order, EDM noise schedule)
# ---------------------------------------------------------------------------

@torch.no_grad()
def heun_sample(model, n_samples, latent_shape, ic_delta, ic_vbv,
                params, redshift, sigma_data,
                num_steps=32, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                cfg_ic=1.0, cfg_params=2.0,
                null_param=None, device='cuda'):
    """Deterministic Heun 2nd-order sampler, EDM noise schedule.
    With CFG (independent guidance on IC and params)."""
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_t = (sigma_max ** (1 / rho)
               + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t = torch.cat([sigma_t, sigma_t.new_zeros(1)])  # append sigma=0

    # Init noisy latent
    x = torch.randn((n_samples,) + latent_shape, device=device) * sigma_t[0]

    null_ic_d = torch.zeros_like(ic_delta)
    null_ic_v = torch.zeros_like(ic_vbv)
    null_p    = null_param.expand_as(params) if null_param is not None else torch.zeros_like(params)
    null_z    = torch.full_like(redshift, 10.0)

    def D(x_in, sigma):
        sig = sigma.expand(n_samples)
        if cfg_ic == 1.0 and cfg_params == 1.0:
            return denoise(model, x_in, sig, sigma_data,
                           ic_delta, ic_vbv, params, redshift)
        # uncond (everything dropped)
        d_uncond = denoise(model, x_in, sig, sigma_data,
                           null_ic_d, null_ic_v, null_p, null_z)
        d_ic     = denoise(model, x_in, sig, sigma_data,
                           ic_delta,  ic_vbv,  null_p, null_z)
        d_full   = denoise(model, x_in, sig, sigma_data,
                           ic_delta,  ic_vbv,  params, redshift)
        # Compositional CFG: full = uncond + w_ic*(ic - uncond) + w_p*(full - ic)
        return d_uncond + cfg_ic * (d_ic - d_uncond) + cfg_params * (d_full - d_ic)

    for i in range(num_steps):
        sigma_cur = sigma_t[i]
        sigma_next = sigma_t[i + 1]
        d_cur = D(x, sigma_cur)
        # Heun's method
        dxdt = (x - d_cur) / sigma_cur
        x_next = x + (sigma_next - sigma_cur) * dxdt
        if sigma_next > 0:
            d_next = D(x_next, sigma_next)
            dxdt_next = (x_next - d_next) / sigma_next
            x_next = x + 0.5 * (sigma_next - sigma_cur) * (dxdt + dxdt_next)
        x = x_next
    return x


# ---------------------------------------------------------------------------
# Train script
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# T21-space PS callback (decode sampled latents, compare PS to ground truth)
# ---------------------------------------------------------------------------

def _pearson(a, b):
    """Pearson correlation of two arrays (flattened)."""
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum()) + 1e-30
    return float((a*b).sum() / denom)


def _moments(x):
    """Return (mean, std, skew, excess_kurtosis) of x."""
    m = float(x.mean())
    s = float(x.std() + 1e-30)
    c = x - m
    sk = float((c**3).mean() / s**3)
    ku = float((c**4).mean() / s**4 - 3.0)
    return m, s, sk, ku


def _multiscale_var(x, sigmas=(1.0, 2.0, 4.0)):
    """Variance of x after gaussian smoothing at each sigma (in voxel units)."""
    from scipy.ndimage import gaussian_filter
    out = []
    for s in sigmas:
        sm = gaussian_filter(x, sigma=s, mode='wrap')
        out.append(float(sm.var()))
    return out


@torch.no_grad()
def t21_ps_check(model, vae, val_ds, latent_mean, latent_std,
                 sigma_data, device, n_samples=4, num_steps=18, chunk=8):
    """Sample n_samples latents conditioned on val items, decode, and compute
    a battery of quality metrics in T21 space."""
    import numpy as np
    n_samples = min(n_samples, len(val_ds))
    idxs = np.random.choice(len(val_ds), size=n_samples, replace=False)
    items = [val_ds[i] for i in idxs]
    ic_d  = torch.stack([it['ic_delta'] for it in items]).to(device)
    ic_v  = torch.stack([it['ic_vbv']   for it in items]).to(device)
    params   = torch.stack([it['params'][:4] for it in items]).to(device)
    redshift = torch.stack([it['redshift']   for it in items]).to(device)
    x_true   = torch.stack([it['patch']      for it in items]).to(device)

    # Sample in chunks to avoid OOM (sampling batch=128 through U-Net would OOM)
    t21_recon_list = []
    for s in range(0, n_samples, chunk):
        e = min(s + chunk, n_samples)
        z = heun_sample(model, e - s, (8, 32, 32, 32),
                        ic_d[s:e], ic_v[s:e], params[s:e], redshift[s:e],
                        sigma_data=sigma_data, num_steps=num_steps,
                        cfg_ic=1.0, cfg_params=1.0,
                        null_param=None, device=device)
        z_unnorm = z * latent_std + latent_mean
        t21_recon_list.append(vae.decoder(z_unnorm).cpu())
    t21_recon = torch.cat(t21_recon_list).squeeze(1).numpy()
    t21_true  = x_true.cpu().squeeze(1).numpy()
    ic_delta_np = ic_d.cpu().squeeze(1).numpy()

    # --- PS ratio ---
    k, ps_t = power_spectrum(t21_true)
    _, ps_r = power_spectrum(t21_recon)
    valid = ps_t.mean(0) > 1e-20
    ratio = ps_r.mean(0) / np.maximum(ps_t.mean(0), 1e-30)
    ps_large = float(ratio[(k<0.1)&valid].mean()) if ((k<0.1)&valid).any() else float('nan')
    ps_mid   = float(ratio[(k>=0.1)&(k<0.5)&valid].mean()) if ((k>=0.1)&(k<0.5)&valid).any() else float('nan')
    ps_small = float(ratio[(k>=0.5)&valid].mean()) if ((k>=0.5)&valid).any() else float('nan')

    # --- Pearson correlations per sample ---
    pixel_corrs = [_pearson(t21_recon[i], t21_true[i]) for i in range(n_samples)]
    ic_corrs_recon = [_pearson(t21_recon[i], ic_delta_np[i]) for i in range(n_samples)]
    ic_corrs_true  = [_pearson(t21_true[i],  ic_delta_np[i]) for i in range(n_samples)]

    # --- Moments ---
    moms_t = [_moments(t21_true[i])  for i in range(n_samples)]
    moms_r = [_moments(t21_recon[i]) for i in range(n_samples)]
    # average across samples
    std_t  = np.mean([m[1] for m in moms_t])
    std_r  = np.mean([m[1] for m in moms_r])
    skew_t = np.mean([m[2] for m in moms_t])
    skew_r = np.mean([m[2] for m in moms_r])
    kurt_t = np.mean([m[3] for m in moms_t])
    kurt_r = np.mean([m[3] for m in moms_r])

    # --- Multiscale variance ratio ---
    msv_t = np.mean([_multiscale_var(t21_true[i])  for i in range(n_samples)], axis=0)
    msv_r = np.mean([_multiscale_var(t21_recon[i]) for i in range(n_samples)], axis=0)
    msv_ratio = (msv_r / np.maximum(msv_t, 1e-30)).tolist()

    return dict(
        ps_large=ps_large, ps_mid=ps_mid, ps_small=ps_small,
        decoded_std=float(std_r), true_std=float(std_t),
        pixel_corr=float(np.mean(pixel_corrs)),       # average per-sample (T21_pred, T21_true) corr
        ic_corr_recon=float(np.mean(ic_corrs_recon)), # spatial alignment of generated T21 with IC
        ic_corr_true =float(np.mean(ic_corrs_true)),  # reference: alignment of true T21 with IC
        skew_true=float(skew_t),  skew_recon=float(skew_r),
        kurt_true=float(kurt_t),  kurt_recon=float(kurt_r),
        msv_ratio_1vox=msv_ratio[0],
        msv_ratio_2vox=msv_ratio[1],
        msv_ratio_4vox=msv_ratio[2],
    )


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt',  default='/root/autodl-tmp/checkpoints/vae_v4/vae_v4_final.pt')
    p.add_argument('--out_dir',   default='/root/autodl-tmp/checkpoints/ldm_v1')
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--epochs',     type=int, default=400)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--wd',         type=float, default=1e-4)
    p.add_argument('--patches_per_cube', type=int, default=4)
    p.add_argument('--P_mean',     type=float, default=-0.4)
    p.add_argument('--P_std',      type=float, default=1.2)
    p.add_argument('--drop_ic',    type=float, default=0.10)
    p.add_argument('--drop_params',type=float, default=0.10)
    p.add_argument('--drop_both',  type=float, default=0.05)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--ps_check_every', type=int, default=10)
    p.add_argument('--val_subset',     type=int, default=20,
                   help='per-epoch val: random subset for trend monitoring')
    p.add_argument('--ps_check_samples', type=int, default=129,
                   help='PS check: full val set for accurate metrics')
    p.add_argument('--resume',     default=None)
    p.add_argument('--finetune', action='store_true',
                   help='When resuming, load only model weights (fresh opt/scheduler).')
    p.add_argument('--ema_decays', nargs='+', type=float, default=[0.9999, 0.999])
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda')
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load frozen VAE ----------
    print(f"Loading frozen VAE from {args.vae_ckpt}")
    ck = torch.load(args.vae_ckpt, map_location=device)
    cfg = ck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
                base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(device)
    vae.load_state_dict(ck['model']); vae.eval()
    for p_ in vae.parameters(): p_.requires_grad_(False)
    latent_mean = ck['latent_mean'].view(1, -1, 1, 1, 1).to(device)
    latent_std  = ck['latent_std' ].view(1, -1, 1, 1, 1).to(device)
    print(f"VAE latent shape: {cfg['latent_shape']}, per-ch std mean = {latent_std.mean().item():.3f}")

    # ---------- Dataset ----------
    train_ds, weights = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        redshifts=(8, 9, 10, 11, 12),
        patches_per_cube=args.patches_per_cube,
        augment=True, split='train',
    )
    sampler = make_balanced_sampler(weights, num_samples=len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_ds, val_weights = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        patches_per_cube=1, augment=False, split='val',
    )
    # val_loader is SHUFFLED so each epoch's val_subset is a fresh random pick
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    print(f"train: {len(train_ds)} patches, val: {len(val_ds)} patches")

    # ---------- Sigma data — empirical std of normalized latent ----------
    print("Estimating sigma_data on a few train batches ...")
    with torch.no_grad():
        accum_var = 0.0
        n_seen = 0
        for i, batch in enumerate(train_loader):
            if i >= 20: break
            x = batch['patch'].to(device)
            mu, _ = vae.encoder(x)
            z = (mu - latent_mean) / latent_std
            accum_var += z.float().pow(2).sum().item()
            n_seen += z.numel()
        sigma_data = math.sqrt(accum_var / n_seen)
    print(f"empirical sigma_data = {sigma_data:.4f}  (expected ~1.0)")

    # ---------- Model ----------
    model = LDMUNet3D(latent_ch=cfg['latent_ch']).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"LDM U-Net params: {n/1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler()
    emas = {d: EMA(model, decay=d) for d in args.ema_decays}

    null_param = torch.zeros(1, 4, device=device)

    # ---------- Resume ----------
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        rc = torch.load(args.resume, map_location=device)
        model.load_state_dict(rc['model'])
        if args.finetune:
            # Fresh optimizer + scheduler so we use args.lr from scratch
            print(f"FINETUNE mode: loading model weights only, fresh opt/scheduler at lr={args.lr}")
            for d in args.ema_decays:
                if f'ema_{d}' in rc:
                    emas[d].shadow = {k: v.to(device) for k, v in rc[f'ema_{d}'].items()}
            start_epoch = 0
        else:
            opt.load_state_dict(rc['opt'])
            if 'scheduler' in rc: scheduler.load_state_dict(rc['scheduler'])
            for d in args.ema_decays:
                if f'ema_{d}' in rc:
                    emas[d].shadow = {k: v.to(device) for k, v in rc[f'ema_{d}'].items()}
            start_epoch = rc['epoch'] + 1
            print(f"resumed from epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,lr,sigma_data\n')
    ps_log_path = os.path.join(args.out_dir, 'ps_check.csv')
    if not os.path.exists(ps_log_path):
        with open(ps_log_path, 'w') as f:
            f.write('epoch,ps_large,ps_mid,ps_small,decoded_std,true_std,'
                    'pixel_corr,ic_corr_recon,ic_corr_true,'
                    'skew_true,skew_recon,kurt_true,kurt_recon,'
                    'msv_ratio_1vox,msv_ratio_2vox,msv_ratio_4vox\n')

    # ---------- Training loop ----------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            x      = batch['patch'].to(device)
            ic_d   = batch['ic_delta'].to(device)
            ic_v   = batch['ic_vbv'].to(device)
            # we use only 4 astro params (drop the 5th = redshift entry from PARAM_KEYS)
            params = batch['params'][:, :4].to(device)
            redshift = batch['redshift'].to(device)

            with torch.no_grad():
                mu, _ = vae.encoder(x)
                z = (mu - latent_mean) / latent_std

            with autocast(dtype=torch.bfloat16):
                loss = edm_loss(model, z, ic_d, ic_v, params, redshift,
                                sigma_data=sigma_data,
                                P_mean=args.P_mean, P_std=args.P_std,
                                drop_ic=args.drop_ic, drop_params=args.drop_params,
                                drop_both=args.drop_both,
                                null_param=null_param)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for e in emas.values(): e.update(model)

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / n_batches

        # ---------- Validation (quick subset for trend) ----------
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            n_seen = 0
            for batch in val_loader:
                x      = batch['patch'].to(device)
                ic_d   = batch['ic_delta'].to(device)
                ic_v   = batch['ic_vbv'].to(device)
                params = batch['params'][:, :4].to(device)
                redshift = batch['redshift'].to(device)
                mu, _ = vae.encoder(x)
                z = (mu - latent_mean) / latent_std
                with autocast(dtype=torch.bfloat16):
                    loss = edm_loss(model, z, ic_d, ic_v, params, redshift,
                                    sigma_data=sigma_data,
                                    P_mean=args.P_mean, P_std=args.P_std,
                                    drop_ic=0.0, drop_params=0.0, drop_both=0.0,
                                    null_param=null_param)
                val_loss_sum += loss.item() * x.shape[0]
                n_seen += x.shape[0]
                if n_seen >= args.val_subset:
                    break
            val_loss = val_loss_sum / max(n_seen, 1)

        dt = time.time() - t0
        print(f"Ep {epoch:4d} | tr {train_loss:.5f} | val {val_loss:.5f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {dt:.0f}s", flush=True)
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                    f"{scheduler.get_last_lr()[0]:.6e},{sigma_data:.6f}\n")

        if (epoch + 1) % args.ps_check_every == 0:
            try:
                ps = t21_ps_check(model, vae, val_ds, latent_mean, latent_std,
                                  sigma_data=sigma_data, device=device,
                                  n_samples=min(args.ps_check_samples, len(val_ds)))
                print(f"  PS check: PS=({ps['ps_large']:.3f},{ps['ps_mid']:.3f},{ps['ps_small']:.3f}) "
                      f"std={ps['decoded_std']:.3f}/{ps['true_std']:.3f} "
                      f"pix_corr={ps['pixel_corr']:.3f} "
                      f"IC_corr={ps['ic_corr_recon']:.3f}/{ps['ic_corr_true']:.3f} "
                      f"skew={ps['skew_recon']:.2f}/{ps['skew_true']:.2f} "
                      f"kurt={ps['kurt_recon']:.2f}/{ps['kurt_true']:.2f} "
                      f"msv={ps['msv_ratio_1vox']:.2f}/{ps['msv_ratio_2vox']:.2f}/{ps['msv_ratio_4vox']:.2f}",
                      flush=True)
                with open(ps_log_path, 'a') as f:
                    f.write(f"{epoch},{ps['ps_large']:.5f},{ps['ps_mid']:.5f},{ps['ps_small']:.5f},"
                            f"{ps['decoded_std']:.5f},{ps['true_std']:.5f},"
                            f"{ps['pixel_corr']:.5f},{ps['ic_corr_recon']:.5f},{ps['ic_corr_true']:.5f},"
                            f"{ps['skew_true']:.5f},{ps['skew_recon']:.5f},"
                            f"{ps['kurt_true']:.5f},{ps['kurt_recon']:.5f},"
                            f"{ps['msv_ratio_1vox']:.5f},{ps['msv_ratio_2vox']:.5f},"
                            f"{ps['msv_ratio_4vox']:.5f}\n")
            except Exception as e:
                print(f"  PS check failed: {e}", flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.out_dir, f'ldm_epoch{epoch:04d}.pt')
            ckpt_out = dict(
                epoch=epoch, model=model.state_dict(), opt=opt.state_dict(),
                scheduler=scheduler.state_dict(),
                args=vars(args), sigma_data=sigma_data,
            )
            for d, e in emas.items():
                ckpt_out[f'ema_{d}'] = e.shadow
            torch.save(ckpt_out, path)
            print(f"  saved {path}")

    print("LDM training complete.")


if __name__ == '__main__':
    main()
