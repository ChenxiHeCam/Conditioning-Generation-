"""
Train the Conditional Flow Matching model in VAE latent space.
Requires a trained VAE checkpoint.
Usage:  python train_flow.py --vae_ckpt PATH [options]
"""
import os, argparse, time, copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from dataset import T21Dataset
from models.vae import VAE3D
from models.flow_matching import FlowUNet3D, ICEncoder, ConditionalFlowMatcher


# -------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',    default=None,
                   help='varying_astro root (T21+params+shared IC). If omitted, only varying_IC is used.')
    p.add_argument('--data_root_ic', default='/home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_IC',
                   help='varying_IC root (T21+IC per sim, fixed params). Set to empty string to disable.')
    p.add_argument('--out_dir',    default='/home/ch2067/rds/hpc-work/21cm_gen/checkpoints/flow')
    p.add_argument('--vae_ckpt',   required=True, help='Path to trained VAE checkpoint')
    p.add_argument('--redshifts',  nargs='+', type=int, default=[10])
    p.add_argument('--patch_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--epochs',     type=int, default=500)
    p.add_argument('--cfg_dropout',type=float, default=0.1)
    p.add_argument('--cfg_scale',  type=float, default=3.0)
    p.add_argument('--base_ch',    type=int, default=128)
    p.add_argument('--latent_ch',  type=int, default=4)
    p.add_argument('--vae_base_ch',type=int, default=64)
    p.add_argument('--param_dim',  type=int, default=8)
    p.add_argument('--save_every', type=int, default=50)
    p.add_argument('--resume',     default=None)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--num_steps',    type=int,   default=100,      help='ODE steps at eval')
    # Rectified Flow / physics loss
    p.add_argument('--time_mode',    default='lognormal',
                   choices=['lognormal', 'uniform'],
                   help='t-sampling: lognormal=Rectified-Flow style, uniform=vanilla CFM')
    p.add_argument('--ps_weight',    type=float, default=0.01,
                   help='Weight for latent-space power-spectrum physics loss (0=off)')
    p.add_argument('--ps_image',     action='store_true',
                   help='Compute PS in image space (decode z→x); more physical but ~2× slower')
    p.add_argument('--ic_cond_mode', default='cross_attn',
                   choices=['concat', 'cross_attn'],
                   help='How IC features are injected: concat at input or cross-attn in decoder')
    p.add_argument('--reflow_steps', type=int,   default=0,
                   help='If >0, replace each batch\'s z1 with ODE-generated reflow target '
                        '(use after main training converges; typical: 5-10 steps)')
    # EMA + 训练稳定性
    p.add_argument('--ema_decay',    type=float, default=0.9999,
                   help='EMA 衰减率（0=禁用）。EDM2 推荐 0.9999')
    p.add_argument('--weight_norm',  action='store_true',
                   help='超球面权重约束：每步后将 Conv3d/Linear 权重行归一化到单位球面（EDM2）')
    return p.parse_args()


# -------------------------------------------------------------------------
# EMA 辅助函数（手动实现，不依赖外部库）
# -------------------------------------------------------------------------
@torch.no_grad()
def ema_update(ema_state, model_state, decay):
    """指数移动平均更新：ema = decay * ema + (1-decay) * model"""
    for k in ema_state:
        if ema_state[k].is_floating_point():
            ema_state[k].mul_(decay).add_(model_state[k], alpha=1.0 - decay)
        else:
            ema_state[k].copy_(model_state[k])


@torch.no_grad()
def apply_weight_norm(model):
    """超球面权重约束（EDM2）：将 Conv3d/Linear 权重每行归一化到单位球面"""
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            w = m.weight.data
            # 沿输出维度（dim=0 以外的所有维度）归一化
            norm = w.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
            w.div_(norm.view(-1, *([1] * (w.dim() - 1))))


# -------------------------------------------------------------------------
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device: {device}")

    # Load frozen VAE
    vae = VAE3D(in_ch=1, latent_ch=args.latent_ch,
                base_ch=args.vae_base_ch, ch_mults=(1, 2)).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(ckpt['model'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"Loaded VAE from {args.vae_ckpt}")

    # Data — support varying_astro, varying_IC, or both combined
    def make_split(split):
        parts = []
        if args.data_root:
            parts.append(T21Dataset(args.data_root, args.patch_size,
                                    redshifts=args.redshifts, split=split))
        if args.data_root_ic:
            parts.append(T21Dataset(args.data_root_ic, args.patch_size,
                                    redshifts=args.redshifts, split=split))
        if not parts:
            raise ValueError("No data_root or data_root_ic specified.")
        return ConcatDataset(parts) if len(parts) > 1 else parts[0]

    train_ds = make_split('train')
    val_ds   = make_split('val')
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Flow model
    ic_encoder = ICEncoder(in_ch=2, out_ch=4).to(device)
    unet = FlowUNet3D(
        latent_ch=args.latent_ch, ic_ch=4, base_ch=args.base_ch, ch_mults=(1, 2),
        param_dim=args.param_dim, param_embed_dim=256,
        ic_cond_mode=args.ic_cond_mode,
    ).to(device)
    flow = ConditionalFlowMatcher(unet, ic_encoder,
                                  cfg_dropout=args.cfg_dropout,
                                  time_mode=args.time_mode).to(device)

    # Decoder for physics loss (frozen VAE decode; grads flow through input only)
    decoder_fn = vae.decode if (args.ps_weight > 0 and args.ps_image) else None
    print(f"Time mode: {args.time_mode} | PS weight: {args.ps_weight} "
          f"({'image' if decoder_fn else 'latent'} space) | "
          f"Reflow steps: {args.reflow_steps}")

    n_params = sum(p.numel() for p in flow.parameters())
    print(f"Flow model parameters: {n_params/1e6:.2f}M")

    # EMA 初始化
    use_ema = args.ema_decay > 0
    ema_state = copy.deepcopy(flow.state_dict()) if use_ema else None
    if use_ema:
        print(f"EMA enabled: decay={args.ema_decay}")
    if args.weight_norm:
        print("Spherical weight normalization enabled (EDM2)")

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        flow.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1
        # 恢复 EMA 状态（向后兼容旧 checkpoint）
        if use_ema:
            if 'ema' in ckpt:
                ema_state = ckpt['ema']
            else:
                ema_state = copy.deepcopy(flow.state_dict())
                print("  Warning: no EMA state in checkpoint, initialized from model weights")
        print(f"Resumed from epoch {start_epoch}")

    log_path = os.path.join(args.out_dir, 'log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,train_mse,train_ps,val_loss,val_ema_loss\n')

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        flow.train()
        t0 = time.time()
        tr_loss = 0.0
        tr_mse  = 0.0
        tr_ps   = 0.0

        for batch in train_loader:
            x      = batch['patch'].to(device)   # (B,1,64,64,64)
            params = batch['params'].to(device)   # (B, param_dim)

            ic_delta = batch['ic_delta'].to(device)  # (B,1,64,64,64)
            ic_vbv   = batch['ic_vbv'].to(device)

            with torch.no_grad():
                z, _, _ = vae.encode(x)              # (B,4,16,16,16)

            if args.reflow_steps > 0:
                with torch.no_grad():
                    ic_enc_rf = flow.ic_encoder(ic_delta, ic_vbv)
                    z0_rf     = torch.randn_like(z)
                    z1_rf     = flow.reflow_sample_z1(z0_rf, params, ic_enc_rf,
                                                      num_steps=args.reflow_steps)
                loss, loss_dict = flow.training_loss(z1_rf, params, ic_delta, ic_vbv,
                                          z0=z0_rf,
                                          decoder=decoder_fn,
                                          ps_weight=args.ps_weight)
            else:
                loss, loss_dict = flow.training_loss(z, params, ic_delta, ic_vbv,
                                          decoder=decoder_fn,
                                          ps_weight=args.ps_weight)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()

            # EMA 更新（在 optimizer step 之后）
            if use_ema:
                ema_update(ema_state, flow.state_dict(), args.ema_decay)

            # 超球面权重约束（在 EMA 更新之后）
            if args.weight_norm:
                apply_weight_norm(flow)

            tr_loss += loss.item()
            tr_mse  += loss_dict['mse']
            tr_ps   += loss_dict['ps']

        scheduler.step()
        n_batches = len(train_loader)
        tr_loss /= n_batches
        tr_mse  /= n_batches
        tr_ps   /= n_batches

        # Validation（用当前模型权重）
        flow.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x        = batch['patch'].to(device)
                params   = batch['params'].to(device)
                ic_delta = batch['ic_delta'].to(device)
                ic_vbv   = batch['ic_vbv'].to(device)
                z, _, _  = vae.encode(x)
                l, _ = flow.training_loss(z, params, ic_delta, ic_vbv,
                                          decoder=decoder_fn,
                                          ps_weight=args.ps_weight)
                vl_loss += l.item()
        vl_loss /= len(val_loader)

        # Validation（用 EMA 权重）
        vl_ema_loss = 0.0
        if use_ema:
            # 暂存当前权重 → 加载 EMA → 验证 → 恢复
            model_state_backup = copy.deepcopy(flow.state_dict())
            flow.load_state_dict(ema_state)
            with torch.no_grad():
                for batch in val_loader:
                    x        = batch['patch'].to(device)
                    params   = batch['params'].to(device)
                    ic_delta = batch['ic_delta'].to(device)
                    ic_vbv   = batch['ic_vbv'].to(device)
                    z, _, _  = vae.encode(x)
                    l, _ = flow.training_loss(z, params, ic_delta, ic_vbv,
                                              decoder=decoder_fn,
                                              ps_weight=args.ps_weight)
                    vl_ema_loss += l.item()
            vl_ema_loss /= len(val_loader)
            flow.load_state_dict(model_state_backup)

        elapsed = time.time() - t0
        ema_str = f" | val_ema {vl_ema_loss:.6f}" if use_ema else ""
        print(f"Epoch {epoch:4d} | train {tr_loss:.6f} (mse {tr_mse:.6f} ps {tr_ps:.6f}) "
              f"| val {vl_loss:.6f}{ema_str} | {elapsed:.1f}s")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{tr_loss:.8f},{tr_mse:.8f},{tr_ps:.8f},"
                    f"{vl_loss:.8f},{vl_ema_loss:.8f}\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            path = os.path.join(args.out_dir, f'flow_epoch{epoch:04d}.pt')
            save_dict = {'epoch': epoch, 'model': flow.state_dict(),
                         'opt': opt.state_dict()}
            if use_ema:
                save_dict['ema'] = ema_state
            torch.save(save_dict, path)
            print(f"  Saved {path}")

    print("Flow matching training complete.")


if __name__ == '__main__':
    main()
