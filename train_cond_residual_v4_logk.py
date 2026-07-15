# CondResidual v4: finetune v3lowk with log-spaced PS bins + ratio MSE.
# Same arch (base_ch=32), so can init from v3lowk ep19 directly.
import os, sys, argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast

sys.path.insert(0, "/root/21cm_gen")
from dataset import T21Dataset
from models.vae import VAE3D
from train_cond_residual import CondResidualUNet
from train_cond_residual_v2 import LDMPairsDataset


def ratio_ps_loss_logk(pred, target, n_bins=12, k_min=0.05, Lpix=3.0):
    """Log-spaced k bins, ratio MSE in log domain.
       Equally weights every decade in k; penalises P_pred/P_target deviating from 1."""
    pred = pred.float().squeeze(1); target = target.float().squeeze(1)
    N = pred.shape[-1]
    Fp = torch.fft.rfftn(pred   - pred  .mean(dim=(-3,-2,-1), keepdim=True), dim=(-3,-2,-1))
    Ft = torch.fft.rfftn(target - target.mean(dim=(-3,-2,-1), keepdim=True), dim=(-3,-2,-1))
    Pp = Fp.abs().pow(2)
    Pt = Ft.abs().pow(2)
    k1  = torch.fft.fftfreq (N, device=pred.device) * N
    k1r = torch.fft.rfftfreq(N, device=pred.device) * N
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag  = (KX**2 + KY**2 + KZ**2).sqrt()
    kbox  = 2 * math.pi / (N * Lpix)
    Kphys = Kmag * kbox
    k_nyq = math.pi / Lpix
    edges = torch.logspace(math.log10(k_min), math.log10(k_nyq * 0.95),
                            n_bins + 1, device=pred.device)
    loss = torch.tensor(0.0, device=pred.device); used = 0
    for i in range(n_bins):
        m = (Kphys >= edges[i]) & (Kphys < edges[i + 1])
        if not m.any():
            continue
        pp = Pp[:, m].mean(-1)
        pt = Pt[:, m].mean(-1)
        log_ratio = torch.log((pp + 1e-12) / (pt + 1e-12))
        loss = loss + (log_ratio ** 2).mean()
        used += 1
    return loss / max(used, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt',  required=True)
    p.add_argument('--init_ckpt', required=True)
    p.add_argument('--ldm_pairs_file', default=None)
    p.add_argument('--out_dir',   required=True)
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',        type=float, default=2e-5)
    p.add_argument('--epochs',    type=int, default=30)
    p.add_argument('--ps_weight', type=float, default=0.5)
    p.add_argument('--ldm_frac',  type=float, default=0.4)
    p.add_argument('--base_ch',   type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_every',  type=int, default=5)
    p.add_argument('--max_per_z',   type=int, default=50)
    p.add_argument('--primary_z',   type=int, default=10)
    p.add_argument('--holdout_frac',type=float, default=0.25)
    p.add_argument('--patches_per_cube', type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda'

    vae_ck = torch.load(args.vae_ckpt, map_location=dev)
    cfg = vae_ck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
                base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(dev).eval()
    vae.load_state_dict(vae_ck['model'])
    for q in vae.parameters(): q.requires_grad_(False)

    model = CondResidualUNet(base_ch=args.base_ch).to(dev)
    ck = torch.load(args.init_ckpt, map_location=dev)
    model.load_state_dict(ck['model'])
    print(f"init from {args.init_ckpt} (ep{ck.get('epoch','?')})  "
          f"log-spaced PS bins + ratio MSE, ps_weight={args.ps_weight}")

    def mk_ds(root, split):
        return T21Dataset(root, 64, redshifts=args.redshifts, split=split,
                          load_ic=True, max_per_z=args.max_per_z, primary_z=args.primary_z,
                          holdout_frac=args.holdout_frac, patches_per_cube=args.patches_per_cube)
    vae_train_sets, vae_val_sets = [], []
    for r in [args.data_root_ic, args.data_root_astro]:
        try: vae_train_sets.append(mk_ds(r, 'train'))
        except Exception as e: print("skip train", r, e)
        try: vae_val_sets  .append(mk_ds(r, 'val'))
        except Exception: pass
    vae_train = ConcatDataset(vae_train_sets)
    vae_val   = ConcatDataset(vae_val_sets)
    vae_train_loader = DataLoader(vae_train, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    vae_val_loader   = DataLoader(vae_val,   args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
    print(f"VAE source: train={len(vae_train)} val={len(vae_val)}")

    if args.ldm_pairs_file:
        ldm_ds = LDMPairsDataset(args.ldm_pairs_file)
        ldm_loader = DataLoader(ldm_ds, args.batch_size, shuffle=True,
                                num_workers=2, pin_memory=True, drop_last=True)
    else:
        ldm_loader = None; args.ldm_frac = 0.0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    log_path = os.path.join(args.out_dir, 'log.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,tr_loss,tr_mse,tr_ps,val_loss,val_mse,val_ps\n")

    def fwd_loss(batch, x_low_pre=None):
        x  = batch['patch'].to(dev, non_blocking=True)
        id_= batch['ic_delta'].to(dev, non_blocking=True)
        iv = batch['ic_vbv'].to(dev, non_blocking=True)
        par= batch['params'].to(dev, non_blocking=True)
        params4 = par[:, :4]; zred = par[:, 4]
        with autocast(dtype=torch.bfloat16):
            if x_low_pre is not None:
                x_low = x_low_pre.to(dev, non_blocking=True)
            else:
                with torch.no_grad():
                    mu, _ = vae.encoder(x); x_low = vae.decoder(mu)
            r_hat = model(x_low, id_, iv, params4, zred)
            final = x_low + r_hat
            l_mse = F.mse_loss(final, x)
            l_ps  = ratio_ps_loss_logk(final, x)
            loss  = l_mse + args.ps_weight * l_ps
        return loss, l_mse, l_ps

    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        tr = dict(loss=0,mse=0,ps=0)
        vae_iter = iter(vae_train_loader)
        ldm_iter = iter(ldm_loader) if ldm_loader is not None else None
        steps = len(vae_train_loader)
        for step in range(steps):
            use_ldm = (ldm_iter is not None) and (torch.rand(1).item() < args.ldm_frac)
            if use_ldm:
                try: b = next(ldm_iter)
                except StopIteration: ldm_iter = iter(ldm_loader); b = next(ldm_iter)
                opt.zero_grad()
                loss, l_mse, l_ps = fwd_loss(b, x_low_pre=b['x_low'])
            else:
                try: b = next(vae_iter)
                except StopIteration: vae_iter = iter(vae_train_loader); b = next(vae_iter)
                opt.zero_grad()
                loss, l_mse, l_ps = fwd_loss(b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr['loss']+=loss.item(); tr['mse']+=l_mse.item(); tr['ps']+=l_ps.item()
        for k in tr: tr[k] /= max(steps, 1)
        sch.step()

        model.eval(); vl = dict(loss=0,mse=0,ps=0); nv=0
        with torch.no_grad():
            for b in vae_val_loader:
                loss, l_mse, l_ps = fwd_loss(b)
                vl['loss']+=loss.item(); vl['mse']+=l_mse.item(); vl['ps']+=l_ps.item(); nv+=1
        for k in vl: vl[k] /= max(nv, 1)

        print(f"Ep {epoch:3d} | tr loss={tr['loss']:.4f} mse={tr['mse']:.4f} ps={tr['ps']:.4f} | "
              f"val loss={vl['loss']:.4f} mse={vl['mse']:.4f} ps={vl['ps']:.4f} | {time.time()-t0:.0f}s",
              flush=True)
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{tr['loss']:.6f},{tr['mse']:.6f},{tr['ps']:.6f},"
                    f"{vl['loss']:.6f},{vl['mse']:.6f},{vl['ps']:.6f}\n")
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.out_dir, f'cond_residual_v4logk_ep{epoch:04d}.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'args': vars(args)}, path)
            print(f"  saved {path}")


if __name__ == '__main__':
    main()
