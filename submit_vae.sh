#!/bin/bash
#SBATCH -J vae_21cm
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/vae_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/vae_%j.err

mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/logs
mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/checkpoints/vae

module purge
# module load python (venv has correct python)
module load cuda/11.8
source /home/ch2067/rds/hpc-work/21cm_gen/venv/bin/activate

cd /home/ch2067/rds/hpc-work/21cm_gen

python train_vae.py \
  --data_root  /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro \
  --out_dir    /home/ch2067/rds/hpc-work/21cm_gen/checkpoints/vae \
  --redshifts  10 \
  --patch_size 64 \
  --batch_size 4 \
  --epochs     300 \
  --lr         1e-4 \
  --kl_weight  1e-4 \
  --kl_anneal  50 \
  --base_ch    64 \
  --latent_ch  4 \
  --save_every 25 \
  --num_workers 4 \
  --redshifts 10
