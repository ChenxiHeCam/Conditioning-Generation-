#!/bin/bash
#SBATCH -J reflow_21cm
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/reflow_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/reflow_%j.err

mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/logs
mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/checkpoints/flow_reflow

module purge
# module load python (venv has correct python)
module load cuda/11.8
source /home/ch2067/rds/hpc-work/21cm_gen/venv/bin/activate

cd /home/ch2067/rds/hpc-work/21cm_gen

# Reflow phase: run AFTER main flow training converges.
# Uses ODE-generated (z0, z1_hat) pairs for straighter trajectories.
# Typical: reflow_steps=5, epochs=100, lr=3e-5.
#
# Set --resume to the best flow checkpoint from main training.
python train_flow.py \
  --data_root    /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro \
  --out_dir      /home/ch2067/rds/hpc-work/21cm_gen/checkpoints/flow_reflow \
  --vae_ckpt     /home/ch2067/rds/hpc-work/21cm_gen/checkpoints/vae/vae_epoch0299.pt \
  --resume       /home/ch2067/rds/hpc-work/21cm_gen/checkpoints/flow/flow_epoch0499.pt \
  --redshifts    10 \
  --patch_size   64 \
  --batch_size   8 \
  --epochs       100 \
  --lr           3e-5 \
  --cfg_dropout  0.1 \
  --base_ch      128 \
  --latent_ch    4 \
  --param_dim    5 \
  --save_every   25 \
  --num_workers  4 \
  --time_mode    lognormal \
  --ps_weight    0.01 \
  --reflow_steps 5 \
  --ema_decay    0.9999
