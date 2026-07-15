#!/bin/bash
#SBATCH -J flow_21cm
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/flow_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/flow_%j.err

mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/logs
mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/checkpoints/flow

module purge
# module load python (venv has correct python)
module load cuda/11.8
source /home/ch2067/rds/hpc-work/21cm_gen/venv/bin/activate

cd /home/ch2067/rds/hpc-work/21cm_gen

# Set --vae_ckpt to the latest saved VAE checkpoint
python train_flow.py \
  --data_root_ic /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_IC \
  --out_dir      /home/ch2067/rds/hpc-work/21cm_gen/checkpoints/flow \
  --vae_ckpt     /home/ch2067/rds/hpc-work/21cm_gen/checkpoints/vae/vae_epoch0299.pt \
  --redshifts    10 \
  --patch_size   64 \
  --batch_size   8 \
  --epochs       500 \
  --lr           1e-4 \
  --cfg_dropout  0.1 \
  --cfg_scale    3.0 \
  --base_ch      128 \
  --latent_ch    4 \
  --param_dim    5 \
  --save_every   50 \
  --num_workers  4 \
  --time_mode    lognormal \
  --ps_weight    0.01 \
  --ic_cond_mode cross_attn \
  --ema_decay    0.9999
