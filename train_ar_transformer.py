"""
Stage 2 of the VQGAN+AR baseline: conditional GPT on VQGAN code indices.

Codes are computed on the fly from the frozen stage-1 VQGAN (one encoder
forward per batch — cheap). CE next-token loss over the 4096 code tokens.
"""
import os, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from ldm_dataset import build_train_dataset, make_balanced_sampler
from models.vqgan3d import VQGAN3D
from models.ar_transformer import ARTransformer3D


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vqgan_ckpt', required=True)
    p.add_argument('--out_dir', default='/root/autodl-tmp/checkpoints/ar_gpt')
    p.add_argument('--data_root_ic',    default='/root/autodl-tmp/ASR21cm/varying_IC')
    p.add_argument('--data_root_astro', default='/root/autodl-tmp/ASR21cm/varying_astro')
    p.add_argument('--redshifts', nargs='+', type=int, default=[8, 9, 10, 11, 12])
    p.add_argument('--epochs',     type=int, default=150)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--dim',   type=int, default=512)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--patches_per_cube', type=int, default=4)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--resume', default=None)
    args = p.parse_args()
    dev = torch.device('cuda')
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.vqgan_ckpt, map_location=dev)
    vcfg = ck['model_config']
    vq = VQGAN3D(in_ch=1, embed_dim=vcfg['embed_dim'], n_embed=vcfg['n_embed'],
                 base_ch=vcfg['base_ch']).to(dev).eval()
    vq.load_state_dict(ck['G'])
    for q in vq.parameters(): q.requires_grad_(False)
    print(f"frozen VQGAN from {args.vqgan_ckpt} (epoch {ck['epoch']})")

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

    model = ARTransformer3D(n_embed=vcfg['n_embed'], dim=args.dim,
                            depth=args.depth, heads=args.heads).to(dev)
    print(f"AR params {sum(q.numel() for q in model.parameters())/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        rc = torch.load(args.resume, map_location=dev)
        model.load_state_dict(rc['model']); opt.load_state_dict(rc['opt'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.epochs - (rc['epoch'] + 1)))
        start_epoch = rc['epoch'] + 1
        print(f"resumed from epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_ce,val_ce,lr,sec\n')

    for epoch in range(start_epoch, args.epochs):
        model.train(); t0 = time.time(); tr, n_b = 0.0, 0
        for batch in loader:
            x      = batch['patch'].to(dev)
            ic_d   = batch['ic_delta'].to(dev)
            ic_v   = batch['ic_vbv'].to(dev)
            params = batch['params'][:, :4].to(dev)
            redshift = batch['redshift'].to(dev)
            with torch.no_grad(), autocast(dtype=torch.bfloat16):
                idx = vq.encode_indices(x)
            with autocast(dtype=torch.bfloat16):
                logits = model(idx, ic_d, ic_v, params, redshift)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                       idx.flatten())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item(); n_b += 1
        scheduler.step()

        model.eval(); va, n_v = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 20: break
                x      = batch['patch'].to(dev)
                ic_d   = batch['ic_delta'].to(dev)
                ic_v   = batch['ic_vbv'].to(dev)
                params = batch['params'][:, :4].to(dev)
                redshift = batch['redshift'].to(dev)
                with autocast(dtype=torch.bfloat16):
                    idx = vq.encode_indices(x)
                    logits = model(idx, ic_d, ic_v, params, redshift)
                va += F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    idx.flatten()).item()
                n_v += 1

        dt = time.time() - t0
        lr_now = opt.param_groups[0]['lr']
        print(f"ep {epoch:03d} | CE {tr/max(n_b,1):.4f} | val {va/max(n_v,1):.4f} "
              f"| lr {lr_now:.2e} | {dt:.0f}s", flush=True)
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{tr/max(n_b,1):.6f},{va/max(n_v,1):.6f},{lr_now:.6e},{dt:.1f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            epoch=epoch,
                            model_config=dict(dim=args.dim, depth=args.depth,
                                              heads=args.heads,
                                              n_embed=vcfg['n_embed'],
                                              vqgan_ckpt=args.vqgan_ckpt)),
                       os.path.join(args.out_dir, f'ar_epoch{epoch:04d}.pt'))
            print(f"  saved epoch {epoch}")


if __name__ == '__main__':
    main()
