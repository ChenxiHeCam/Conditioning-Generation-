# Cond-residual trainer V2: mixes VAE encode-decode pairs (mode A) with LDM-sample pairs (mode B).
# Mode A: x_low = vae.decode(vae.encoder(real_x).mu)  — computed on the fly
# Mode B: x_low = vae.decode(LDM.sample(IC,params,z) * std + mean)  — pre-sampled, loaded from file
# Final pipeline:  x_final = x_low + cond_residual(x_low, IC, params, z)
import os, sys, argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.cuda.amp import autocast

sys.path.insert(0, "/root/21cm_gen")
from dataset import T21Dataset
from models.vae import VAE3D
from train_cond_residual import CondResidualUNet, log_ps_loss


class LDMPairsDataset(Dataset):
    """Wraps the pre-sampled LDM pairs file."""
    def __init__(self, path):
        d = torch.load(path, map_location='cpu')
        self.x_low_ldm = d['x_low_ldm']   # (N,1,64,64,64) fp16
        self.x_real    = d['x_real']
        self.ic_d      = d['ic_d']
        self.ic_v      = d['ic_v']
        self.par       = d['par']         # (N, 5)
        self.meta      = d.get('meta', {})
        print(f"LDM pairs loaded from {path}: N={len(self.x_low_ldm)} meta={self.meta}")
    def __len__(self):
        return len(self.x_low_ldm)
    def __getitem__(self, i):
        return dict(
            x_low   =self.x_low_ldm[i].float(),
            patch   =self.x_real[i].float(),
            ic_delta=self.ic_d[i].float(),
            ic_vbv  =self.ic_v[i].float(),
            params  =self.par[i],
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt',  required=True)
    p.add_argument('--init_ckpt', default=None, help='start from existing cond_residual ckpt (finetune)')
    p.add_argument('--ldm_pairs_file', default=None, help='pre-sampled (x_low_ldm, real) pairs')
    p.add_argument('--out_dir',   required=True)
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5)   # smaller for finetune
    p.add_argument('--epochs', type=int, default=40)   # shorter for finetune
    p.add_argument('--ps_weight', type=float, default=0.3)
    p.add_argument('--ldm_frac', type=float, default=0.4,
                   help='fraction of batches drawn from LDM pairs (rest from VAE on-the-fly)')
    p.add_argument('--base_ch', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--max_per_z', type=int, default=50)
    p.add_argument('--primary_z', type=int, default=10)
    p.add_argument('--holdout_frac', type=float, default=0.25)
    p.add_argument('--patches_per_cube', type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda'

    # ---- frozen VAE ----
    vae_ck = torch.load(args.vae_ckpt, map_location=dev)
    cfg = vae_ck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'],
                base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(dev).eval()
    vae.load_state_dict(vae_ck['model'])
    for q in vae.parameters(): q.requires_grad_(False)
    print(f"frozen VAE: latent_ch={cfg['latent_ch']} ep{vae_ck.get('epoch','?')}")

    # ---- model (finetune-init) ----
    model = CondResidualUNet(base_ch=args.base_ch).to(dev)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=dev)
        model.load_state_dict(ck['model'])
        print(f"loaded init from {args.init_ckpt} (ep{ck.get('epoch','?')})")
    print(f"cond-residual params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ---- data: VAE source (on-the-fly) ----
    def mk_ds(root, split):
        return T21Dataset(root, 64, redshifts=args.redshifts, split=split,
                          load_ic=True, max_per_z=args.max_per_z, primary_z=args.primary_z,
                          holdout_frac=args.holdout_frac, patches_per_cube=args.patches_per_cube)
    vae_train_sets, vae_val_sets = [], []
    for r in [args.data_root_ic, args.data_root_astro]:
        try: vae_train_sets.append(mk_ds(r, 'train'))
        except Exception as e: print("skip", r, e)
        try: vae_val_sets  .append(mk_ds(r, 'val'))
        except Exception as e: pass
    vae_train = ConcatDataset(vae_train_sets)
    vae_val   = ConcatDataset(vae_val_sets)
    vae_train_loader = DataLoader(vae_train, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    vae_val_loader   = DataLoader(vae_val,   args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)
    print(f"VAE source: train={len(vae_train)} val={len(vae_val)}")

    # ---- data: LDM source (precomputed file) ----
    if args.ldm_pairs_file:
        ldm_ds = LDMPairsDataset(args.ldm_pairs_file)
        ldm_loader = DataLoader(ldm_ds, args.batch_size, shuffle=True,
                                num_workers=2, pin_memory=True, drop_last=True)
        print(f"LDM source: {len(ldm_ds)} samples,  mixing at ldm_frac={args.ldm_frac}")
    else:
        ldm_loader = None
        args.ldm_frac = 0.0
        print("LDM source: NONE (mode A only)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    log_path = os.path.join(args.out_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("epoch,train_loss,train_mse,train_ps,val_loss,val_mse,val_ps,n_vae,n_ldm\n")

    def fwd_loss(batch, x_low_precomputed=None):
        x  = batch['patch'].to(dev, non_blocking=True)
        id_= batch['ic_delta'].to(dev, non_blocking=True)
        iv = batch['ic_vbv'].to(dev, non_blocking=True)
        par= batch['params'].to(dev, non_blocking=True)
        params4 = par[:, :4]; zred = par[:, 4]
        with autocast(dtype=torch.bfloat16):
            if x_low_precomputed is not None:
                x_low = x_low_precomputed.to(dev, non_blocking=True)
            else:
                with torch.no_grad():
                    mu, _ = vae.encoder(x)
                    x_low = vae.decoder(mu)
            r_hat = model(x_low, id_, iv, params4, zred)
            final = x_low + r_hat
            l_mse = F.mse_loss(final, x)
            l_ps  = log_ps_loss(final, x)
            loss = l_mse + args.ps_weight * l_ps
        return loss, l_mse, l_ps

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tr = dict(loss=0, mse=0, ps=0); n_vae = 0; n_ldm = 0

        vae_iter = iter(vae_train_loader)
        ldm_iter = iter(ldm_loader) if ldm_loader is not None else None
        steps_per_epoch = len(vae_train_loader)

        for step in range(steps_per_epoch):
            use_ldm = (ldm_iter is not None) and (torch.rand(1).item() < args.ldm_frac)
            if use_ldm:
                try:
                    b = next(ldm_iter)
                except StopIteration:
                    ldm_iter = iter(ldm_loader); b = next(ldm_iter)
                opt.zero_grad()
                loss, l_mse, l_ps = fwd_loss(b, x_low_precomputed=b['x_low'])
                n_ldm += 1
            else:
                try:
                    b = next(vae_iter)
                except StopIteration:
                    vae_iter = iter(vae_train_loader); b = next(vae_iter)
                opt.zero_grad()
                loss, l_mse, l_ps = fwd_loss(b)
                n_vae += 1
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr['loss'] += loss.item(); tr['mse'] += l_mse.item(); tr['ps'] += l_ps.item()
        nb = max(steps_per_epoch, 1)
        for k in tr: tr[k] /= nb
        sch.step()

        # ---- val on VAE source ----
        model.eval(); vl = dict(loss=0, mse=0, ps=0); nv = 0
        with torch.no_grad():
            for b in vae_val_loader:
                loss, l_mse, l_ps = fwd_loss(b)
                vl['loss'] += loss.item(); vl['mse'] += l_mse.item(); vl['ps'] += l_ps.item(); nv += 1
        for k in vl: vl[k] /= max(nv, 1)

        print(f"Ep {epoch:4d} | tr {tr['loss']:.4f} (mse {tr['mse']:.4f} ps {tr['ps']:.4f}) | "
              f"val {vl['loss']:.4f} (mse {vl['mse']:.4f} ps {vl['ps']:.4f}) | "
              f"n_vae={n_vae} n_ldm={n_ldm} | {time.time()-t0:.0f}s", flush=True)
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{tr['loss']:.6f},{tr['mse']:.6f},{tr['ps']:.6f},"
                    f"{vl['loss']:.6f},{vl['mse']:.6f},{vl['ps']:.6f},{n_vae},{n_ldm}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.out_dir, f'cond_residual_v2_ep{epoch:04d}.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'args': vars(args)}, path)
            print(f"  saved {path}")


if __name__ == '__main__':
    main()
