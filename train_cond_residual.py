# Conditional residual decoder for LDM post-processing
# Input:  x_low (1,64^3) + ic_delta (1,64^3) + ic_vbv (1,64^3) + params (4) + redshift (1)
# Output: r_hat (1,64^3)  per-voxel correction; zero-init -> initial output is 0
# Pipeline: x_final = x_low + r_hat
import os, sys, argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast

sys.path.insert(0, "/root/21cm_gen")
from dataset import T21Dataset
from models.vae import VAE3D


def sinusoidal_embed(x, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    args = x[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class FiLM(nn.Module):
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * ch)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
    def forward(self, x, cond):
        s, sh = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + s[:, :, None, None, None]) + sh[:, :, None, None, None]


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_dim):
        super().__init__()
        gn = lambda c: nn.GroupNorm(min(8, c), c)
        self.n1 = gn(in_c); self.c1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.film = FiLM(cond_dim, out_c)
        self.n2 = gn(out_c); self.c2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.skip = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x)))
        h = self.film(h, cond)
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class CondResidualUNet(nn.Module):
    def __init__(self, base_ch=32, cond_dim=128, n_params=4, z_emb=64):
        super().__init__()
        self.z_emb = z_emb
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_params + z_emb, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim))
        self.proj_in = nn.Conv3d(3, base_ch, 3, padding=1)
        self.d1 = ResBlock(base_ch,   base_ch*2, cond_dim)
        self.p1 = nn.Conv3d(base_ch*2, base_ch*2, 4, 2, 1)
        self.d2 = ResBlock(base_ch*2, base_ch*4, cond_dim)
        self.p2 = nn.Conv3d(base_ch*4, base_ch*4, 4, 2, 1)
        self.m1 = ResBlock(base_ch*4, base_ch*4, cond_dim)
        self.m2 = ResBlock(base_ch*4, base_ch*4, cond_dim)
        self.u2 = nn.ConvTranspose3d(base_ch*4, base_ch*4, 4, 2, 1)
        self.dec2 = ResBlock(base_ch*4 + base_ch*4, base_ch*2, cond_dim)
        self.u1 = nn.ConvTranspose3d(base_ch*2, base_ch*2, 4, 2, 1)
        self.dec1 = ResBlock(base_ch*2 + base_ch*2, base_ch, cond_dim)
        self.no = nn.GroupNorm(min(8, base_ch), base_ch)
        self.co = nn.Conv3d(base_ch, 1, 3, padding=1)
        nn.init.zeros_(self.co.weight); nn.init.zeros_(self.co.bias)
    def forward(self, x_low, ic_d, ic_v, params, redshift):
        cond = self.cond_mlp(torch.cat([params, sinusoidal_embed(redshift, self.z_emb)], dim=-1))
        x = torch.cat([x_low, ic_d, ic_v], dim=1)
        x = self.proj_in(x)
        s1 = self.d1(x, cond); h = self.p1(s1)
        s2 = self.d2(h, cond); h = self.p2(s2)
        h = self.m1(h, cond); h = self.m2(h, cond)
        h = self.u2(h); h = self.dec2(torch.cat([h, s2], dim=1), cond)
        h = self.u1(h); h = self.dec1(torch.cat([h, s1], dim=1), cond)
        h = F.silu(self.no(h))
        return self.co(h)


def log_ps_loss(pred, target, n_bins=20):
    pred = pred.float().squeeze(1); target = target.float().squeeze(1)
    N = pred.shape[-1]
    Fp = torch.fft.rfftn(pred - pred.mean(dim=(-3,-2,-1), keepdim=True), dim=(-3,-2,-1))
    Ft = torch.fft.rfftn(target - target.mean(dim=(-3,-2,-1), keepdim=True), dim=(-3,-2,-1))
    Pp = Fp.abs().pow(2) / N**3
    Pt = Ft.abs().pow(2) / N**3
    k1 = torch.fft.fftfreq(N, device=pred.device); k1r = torch.fft.rfftfreq(N, device=pred.device)
    KX, KY, KZ = torch.meshgrid(k1, k1, k1r, indexing='ij')
    Kmag = (KX**2 + KY**2 + KZ**2).sqrt()
    edges = torch.linspace(0, Kmag.max().item()+1e-6, n_bins+1, device=pred.device)
    loss = torch.tensor(0.0, device=pred.device); used = 0
    for i in range(n_bins):
        m = (Kmag >= edges[i]) & (Kmag < edges[i+1])
        if not m.any(): continue
        pp = Pp[:, m].mean(-1); pt = Pt[:, m].mean(-1)
        loss = loss + F.mse_loss(torch.log(pp + 1e-12), torch.log(pt + 1e-12))
        used += 1
    return loss / max(used, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vae_ckpt', required=True)
    p.add_argument('--out_dir',  required=True)
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--redshifts', nargs='+', type=int, default=[8,9,10,11,12])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--ps_weight', type=float, default=0.3)
    p.add_argument('--base_ch', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--max_per_z', type=int, default=50)
    p.add_argument('--primary_z', type=int, default=10)
    p.add_argument('--holdout_frac', type=float, default=0.25)
    p.add_argument('--patches_per_cube', type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda"

    ck = torch.load(args.vae_ckpt, map_location=dev)
    cfg = ck['model_config']
    vae = VAE3D(in_ch=1, latent_ch=cfg['latent_ch'], base_ch=cfg['base_ch'], ch_mults=tuple(cfg['ch_mults'])).to(dev).eval()
    vae.load_state_dict(ck['model'])
    for q in vae.parameters(): q.requires_grad_(False)
    print(f"frozen VAE: latent_ch={cfg['latent_ch']} ep{ck.get('epoch','?')}")

    def mk_ds(root, split):
        return T21Dataset(root, 64, redshifts=args.redshifts, split=split,
                          load_ic=True, max_per_z=args.max_per_z, primary_z=args.primary_z,
                          holdout_frac=args.holdout_frac, patches_per_cube=args.patches_per_cube)
    train_sets = []
    for r in [args.data_root_ic, args.data_root_astro]:
        try: train_sets.append(mk_ds(r, 'train'))
        except Exception as e: print("skip", r, e)
    train_ds = ConcatDataset(train_sets) if len(train_sets)>1 else train_sets[0]
    val_sets = []
    for r in [args.data_root_ic, args.data_root_astro]:
        try: val_sets.append(mk_ds(r, 'val'))
        except Exception as e: pass
    val_ds = ConcatDataset(val_sets) if len(val_sets)>1 else val_sets[0]
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    model = CondResidualUNet(base_ch=args.base_ch).to(dev)
    print(f"cond-residual params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    log_path = os.path.join(args.out_dir, 'log.csv')
    with open(log_path, 'w') as f: f.write("epoch,train_loss,train_mse,train_ps,val_loss,val_mse,val_ps\n")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tr = dict(loss=0, mse=0, ps=0); nb=0
        for b in train_loader:
            x  = b['patch'].to(dev)
            id_= b['ic_delta'].to(dev)
            iv = b['ic_vbv'].to(dev)
            par= b['params'].to(dev)
            params4 = par[:, :4]
            zred    = par[:, 4]
            opt.zero_grad()
            with autocast(dtype=torch.bfloat16):
                with torch.no_grad():
                    mu, _ = vae.encoder(x)
                    x_low = vae.decoder(mu)
                r_hat = model(x_low, id_, iv, params4, zred)
                final = x_low + r_hat
                l_mse = F.mse_loss(final, x)
                l_ps  = log_ps_loss(final, x)
                loss = l_mse + args.ps_weight * l_ps
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr['loss'] += loss.item(); tr['mse'] += l_mse.item(); tr['ps'] += l_ps.item(); nb += 1
        for k in tr: tr[k] /= max(nb,1)
        sch.step()

        model.eval(); vl = dict(loss=0, mse=0, ps=0); nv=0
        with torch.no_grad():
            for b in val_loader:
                x  = b['patch'].to(dev); id_= b['ic_delta'].to(dev); iv = b['ic_vbv'].to(dev)
                par= b['params'].to(dev); params4 = par[:, :4]; zred = par[:, 4]
                with autocast(dtype=torch.bfloat16):
                    mu, _ = vae.encoder(x); x_low = vae.decoder(mu)
                    r_hat = model(x_low, id_, iv, params4, zred)
                    final = x_low + r_hat
                    l_mse = F.mse_loss(final, x); l_ps = log_ps_loss(final, x)
                    loss = l_mse + args.ps_weight * l_ps
                vl['loss'] += loss.item(); vl['mse'] += l_mse.item(); vl['ps'] += l_ps.item(); nv += 1
        for k in vl: vl[k] /= max(nv,1)
        print(f"Ep {epoch:4d} | tr {tr['loss']:.4f} (mse {tr['mse']:.4f} ps {tr['ps']:.4f}) | val {vl['loss']:.4f} (mse {vl['mse']:.4f} ps {vl['ps']:.4f}) | {time.time()-t0:.0f}s", flush=True)
        with open(log_path,'a') as f:
            f.write(f"{epoch},{tr['loss']:.6f},{tr['mse']:.6f},{tr['ps']:.6f},{vl['loss']:.6f},{vl['mse']:.6f},{vl['ps']:.6f}\n")

        if (epoch+1) % args.save_every == 0 or epoch == args.epochs-1:
            path = os.path.join(args.out_dir, f'cond_residual_ep{epoch:04d}.pt')
            torch.save({'epoch':epoch, 'model':model.state_dict(), 'args':vars(args)}, path)
            print(f"  saved {path}")


if __name__ == '__main__':
    main()
