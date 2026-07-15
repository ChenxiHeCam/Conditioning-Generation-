#!/bin/bash
#SBATCH -J infer_8x
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/infer_8x_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/infer_8x_%j.err

mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/eval_results/ff_8x
module purge
module load miniconda/3
module load cuda/11.8
source activate ldm

cd /home/ch2067/rds/hpc-work/21cm_gen

/home/ch2067/.conda/envs/ldm/bin/python infer_baselines.py \
  --model latent \
  --vae_ckpt /home/ch2067/rds/hpc-work/21cm_gen/ckpts/vae_8x_final.pt \
  --ldm_ckpt /home/ch2067/rds/hpc-work/21cm_gen/ckpts/ldm_8x/ldm_epoch0089.pt \
  --data_root /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro \
  --out_dir   /home/ch2067/rds/hpc-work/21cm_gen/eval_results/ff_8x \
  --split test \
  --redshift 10 \
  --n_cubes 100 \
  --steps 30
