#!/bin/bash
# Launch the 6 training chains across 8 GPUs (GPUs 6/7 reserved for residual).
# Each chain runs in its own tmux session so they survive SSH disconnects.
# Logs land in /root/autodl-tmp/logs/<chain>.log.
set -e
cd /root/21cm_gen
export PATH=/root/miniconda3/bin:$PATH
mkdir -p /root/autodl-tmp/logs /root/autodl-tmp/ckpt

R=/root/autodl-tmp                # short alias
COMMON="--data_root_ic $R/ASR21cm/varying_IC --data_root_astro $R/ASR21cm/varying_astro --redshifts 7 8 9 10 11 12 13 --save_every 25 --num_workers 4"

# GPU 0 — pixel-space diffusion (compression 0x)
tmux new-session -d -s c_pixel "
  export CUDA_VISIBLE_DEVICES=0; export PATH=/root/miniconda3/bin:\$PATH
  python -u train_pixel_diffusion.py --out_dir $R/ckpt/pixel_diff $COMMON \
    --epochs 250 --batch_size 8 --save_every 25 \
    2>&1 | tee $R/logs/pixel.log
"

# GPU 1 — 4x compression chain (latent_ch=16): VAE then LDM
tmux new-session -d -s c_4x "
  export CUDA_VISIBLE_DEVICES=1; export PATH=/root/miniconda3/bin:\$PATH
  python -u train_vae.py --out_dir $R/ckpt/vae_4x --base_ch 128 --latent_ch 16 \
    --ch_mults 1 2 --epochs 150 --batch_size 8 \
    --data_root_ic $R/ASR21cm/varying_IC --data_root_astro $R/ASR21cm/varying_astro \
    --redshifts 7 8 9 10 11 12 13 --save_every 25 \
    && python -u finalize_vae.py --ckpt_in $R/ckpt/vae_4x/vae_epoch0149.pt \
         --ckpt_out $R/ckpt/vae_4x/vae_final.pt --latent_ch 16 \
    && python -u train_ldm.py --vae_ckpt $R/ckpt/vae_4x/vae_final.pt \
       --out_dir $R/ckpt/ldm_4x $COMMON --epochs 250 --batch_size 16 \
    2>&1 | tee $R/logs/4x.log
"

# GPU 2 — 8x compression chain (latent_ch=8, the deployed config) on fresh data
tmux new-session -d -s c_8x "
  export CUDA_VISIBLE_DEVICES=2; export PATH=/root/miniconda3/bin:\$PATH
  python -u train_vae.py --out_dir $R/ckpt/vae_8x --base_ch 128 --latent_ch 8 \
    --ch_mults 1 2 --epochs 150 --batch_size 8 \
    --data_root_ic $R/ASR21cm/varying_IC --data_root_astro $R/ASR21cm/varying_astro \
    --redshifts 7 8 9 10 11 12 13 --save_every 25 \
    && python -u finalize_vae.py --ckpt_in $R/ckpt/vae_8x/vae_epoch0149.pt \
         --ckpt_out $R/ckpt/vae_8x/vae_final.pt --latent_ch 8 \
    && python -u train_ldm.py --vae_ckpt $R/ckpt/vae_8x/vae_final.pt \
       --out_dir $R/ckpt/ldm_8x $COMMON --epochs 250 --batch_size 16 \
    2>&1 | tee $R/logs/8x.log
"

# GPU 3 — 16x compression chain (latent_ch=4)
tmux new-session -d -s c_16x "
  export CUDA_VISIBLE_DEVICES=3; export PATH=/root/miniconda3/bin:\$PATH
  python -u train_vae.py --out_dir $R/ckpt/vae_16x --base_ch 128 --latent_ch 4 \
    --ch_mults 1 2 --epochs 150 --batch_size 8 \
    --data_root_ic $R/ASR21cm/varying_IC --data_root_astro $R/ASR21cm/varying_astro \
    --redshifts 7 8 9 10 11 12 13 --save_every 25 \
    && python -u finalize_vae.py --ckpt_in $R/ckpt/vae_16x/vae_epoch0149.pt \
         --ckpt_out $R/ckpt/vae_16x/vae_final.pt --latent_ch 4 \
    && python -u train_ldm.py --vae_ckpt $R/ckpt/vae_16x/vae_final.pt \
       --out_dir $R/ckpt/ldm_16x $COMMON --epochs 250 --batch_size 16 \
    2>&1 | tee $R/logs/16x.log
"

# GPU 4 — VQGAN then AR transformer
tmux new-session -d -s c_ar "
  export CUDA_VISIBLE_DEVICES=4; export PATH=/root/miniconda3/bin:\$PATH
  python -u train_vqgan.py --out_dir $R/ckpt/vqgan $COMMON --epochs 120 --batch_size 8 \
    && python -u train_ar_transformer.py --vqgan_ckpt $R/ckpt/vqgan/vqgan_epoch0119.pt \
       --out_dir $R/ckpt/ar_gpt $COMMON --epochs 150 --batch_size 8 \
    2>&1 | tee $R/logs/ar.log
"

# GPU 5 — conditional StyleGAN
tmux new-session -d -s c_gan "
  export CUDA_VISIBLE_DEVICES=5; export PATH=/root/miniconda3/bin:\$PATH
  python -u train_stylegan.py --out_dir $R/ckpt/stylegan $COMMON --epochs 200 --batch_size 8 \
    2>&1 | tee $R/logs/gan.log
"

# GPU 6 — 2x compression chain (latent_ch=32) — extra ablation point
tmux new-session -d -s c_2x "
  export CUDA_VISIBLE_DEVICES=6; export PATH=/root/miniconda3/bin:\$PATH
  python -u train_vae.py --out_dir $R/ckpt/vae_2x --base_ch 128 --latent_ch 32 \
    --ch_mults 1 2 --epochs 150 --batch_size 8 \
    --data_root_ic $R/ASR21cm/varying_IC --data_root_astro $R/ASR21cm/varying_astro \
    --redshifts 7 8 9 10 11 12 13 --save_every 25 \
    && python -u finalize_vae.py --ckpt_in $R/ckpt/vae_2x/vae_epoch0149.pt \
         --ckpt_out $R/ckpt/vae_2x/vae_final.pt --latent_ch 32 \
    && python -u train_ldm.py --vae_ckpt $R/ckpt/vae_2x/vae_final.pt \
       --out_dir $R/ckpt/ldm_2x $COMMON --epochs 250 --batch_size 16 \
    2>&1 | tee $R/logs/2x.log
"

# GPU 7 — 32x compression chain (latent_ch=2) — extreme ablation end point
tmux new-session -d -s c_32x "
  export CUDA_VISIBLE_DEVICES=7; export PATH=/root/miniconda3/bin:\$PATH
  python -u train_vae.py --out_dir $R/ckpt/vae_32x --base_ch 128 --latent_ch 2 \
    --ch_mults 1 2 --epochs 150 --batch_size 8 \
    --data_root_ic $R/ASR21cm/varying_IC --data_root_astro $R/ASR21cm/varying_astro \
    --redshifts 7 8 9 10 11 12 13 --save_every 25 \
    && python -u finalize_vae.py --ckpt_in $R/ckpt/vae_32x/vae_epoch0149.pt \
         --ckpt_out $R/ckpt/vae_32x/vae_final.pt --latent_ch 2 \
    && python -u train_ldm.py --vae_ckpt $R/ckpt/vae_32x/vae_final.pt \
       --out_dir $R/ckpt/ldm_32x $COMMON --epochs 250 --batch_size 16 \
    2>&1 | tee $R/logs/32x.log
"

sleep 4
tmux ls
echo "---"
echo "Sessions launched. Tail any log with:"
echo "  tail -f $R/logs/<pixel|4x|8x|16x|ar|gan>.log"
