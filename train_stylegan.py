"""
Conditional StyleGAN baseline trainer (IC -> T21, single forward inference).

Hinge GAN + R1 every 16 steps + 0.1*L1 anchor + 0.25*log-PS.
EMA of G is the eval model. Known failure mode is D overpowering G —
watch d_loss -> 0 with g_gan exploding; lower --d_lr if so.
"""
import os, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from ldm_dataset import build_train_dataset, make_balanced_sampler
from models.stylegan3d import StyleGenerator3D, CondDiscriminator3D
from train_vqgan import log_ps_loss
from train_ldm import EMA


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='/root/autodl-tmp/checkpoints/stylegan')
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--epochs',     type=int, default=200)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--g_lr', type=float, default=2e-4)
    p.add_argument('--d_lr', type=float, default=2e-4)
    p.add_argument('--base_ch', type=int, default=32)
    p.add_argument('--l1_weight', type=float, default=0.1)
    p.add_argument('--ps_weight', type=float, default=0.25)
    p.add_argument('--r1_weight', type=float, default=1.0)
    p.add_argument('--r1_every',  type=int, default=16)
    p.add_argument('--patches_per_cube', type=int, default=4)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--resume', default=None)
    args = p.parse_args()
    dev = torch.device('cuda')
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, weights = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        redshifts=tuple(args.redshifts),
        patches_per_cube=args.patches_per_cube, augment=True, split='train')
    sampler = make_balanced_sampler(weights, num_samples=len(train_ds))
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True)
    val_ds, _ = build_train_dataset(
        args.data_root_ic, args.data_root_astro,
        redshifts=tuple(args.redshifts),
        patches_per_cube=1, augment=False, split='val')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"train {len(train_ds)} / val {len(val_ds)} patches")

    G = StyleGenerator3D(base_ch=args.base_ch).to(dev)
    D = CondDiscriminator3D(base_ch=args.base_ch).to(dev)
    print(f"G {sum(q.numel() for q in G.parameters())/1e6:.1f}M, "
          f"D {sum(q.numel() for q in D.parameters())/1e6:.1f}M")
    optG = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.0, 0.99))
    optD = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.0, 0.99))
    ema = EMA(G, decay=0.999)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        rc = torch.load(args.resume, map_location=dev)
        G.load_state_dict(rc['G']); D.load_state_dict(rc['D'])
        optG.load_state_dict(rc['optG']); optD.load_state_dict(rc['optD'])
        if 'ema' in rc:
            ema.shadow = {k: v.to(dev) for k, v in rc['ema'].items()}
        start_epoch = rc['epoch'] + 1
        print(f"resumed from epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,g_gan,l1,ps,d_loss,r1,val_l1,sec\n')

    step = 0
    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train(); t0 = time.time()
        s = dict(g_gan=0, l1=0, ps=0, d=0, r1=0); n_b = 0
        for batch in loader:
            x      = batch['patch'].to(dev)
            ic_d   = batch['ic_delta'].to(dev)
            ic_v   = batch['ic_vbv'].to(dev)
            params = batch['params'][:, :4].to(dev)
            redshift = batch['redshift'].to(dev)

            # ---- D step ----
            with autocast(dtype=torch.bfloat16):
                fake = G(ic_d, ic_v, params, redshift)
                lf = D(fake.detach(), ic_d, ic_v, params, redshift)
                lr_ = D(x, ic_d, ic_v, params, redshift)
            d_loss = F.relu(1 - lr_.float()).mean() + F.relu(1 + lf.float()).mean()
            r1 = torch.tensor(0.0, device=dev)
            if step % args.r1_every == 0:
                x_r = x.detach().requires_grad_(True)
                lr_r1 = D(x_r, ic_d, ic_v, params, redshift)
                grad = torch.autograd.grad(lr_r1.sum(), x_r, create_graph=True)[0]
                r1 = grad.pow(2).sum(dim=(1, 2, 3, 4)).mean()
                d_loss = d_loss + 0.5 * args.r1_weight * r1 * args.r1_every
            optD.zero_grad(); d_loss.backward(); optD.step()

            # ---- G step ----
            with autocast(dtype=torch.bfloat16):
                fake = G(ic_d, ic_v, params, redshift)
                g_gan = -D(fake, ic_d, ic_v, params, redshift).float().mean()
                l1 = F.l1_loss(fake.float(), x.float())
            ps = log_ps_loss(fake.float(), x.float())
            g_loss = g_gan + args.l1_weight * l1 + args.ps_weight * ps
            optG.zero_grad(); g_loss.backward(); optG.step()
            ema.update(G)

            s['g_gan'] += float(g_gan); s['l1'] += float(l1); s['ps'] += float(ps)
            s['d'] += float(d_loss); s['r1'] += float(r1); n_b += 1; step += 1

        G.eval(); val_l1, n_v = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 20: break
                x      = batch['patch'].to(dev)
                ic_d   = batch['ic_delta'].to(dev)
                ic_v   = batch['ic_vbv'].to(dev)
                params = batch['params'][:, :4].to(dev)
                redshift = batch['redshift'].to(dev)
                with autocast(dtype=torch.bfloat16):
                    fake = G(ic_d, ic_v, params, redshift)
                val_l1 += F.l1_loss(fake.float(), x.float()).item(); n_v += 1

        m = {k: v / max(n_b, 1) for k, v in s.items()}
        dt = time.time() - t0
        print(f"ep {epoch:03d} | gGAN {m['g_gan']:.3f} L1 {m['l1']:.4f} "
              f"ps {m['ps']:.3f} | D {m['d']:.3f} r1 {m['r1']:.3f} "
              f"| val L1 {val_l1/max(n_v,1):.4f} | {dt:.0f}s", flush=True)
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{m['g_gan']:.5f},{m['l1']:.5f},{m['ps']:.5f},"
                    f"{m['d']:.5f},{m['r1']:.5f},{val_l1/max(n_v,1):.5f},{dt:.1f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(dict(G=G.state_dict(), D=D.state_dict(),
                            optG=optG.state_dict(), optD=optD.state_dict(),
                            ema=ema.shadow, epoch=epoch,
                            model_config=dict(base_ch=args.base_ch)),
                       os.path.join(args.out_dir, f'stylegan_epoch{epoch:04d}.pt'))
            print(f"  saved epoch {epoch}")


if __name__ == '__main__':
    main()
