#!/bin/bash
#SBATCH -J eval_2x
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/eval_2x_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/eval_2x_%j.err

mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/logs /home/ch2067/rds/hpc-work/21cm_gen/eval_results
module purge
module load miniconda/3
module load cuda/11.8
source activate ldm

cd /home/ch2067/rds/hpc-work/21cm_gen

/home/ch2067/.conda/envs/ldm/bin/python ldm_eval.py \
  --ldm_ckpt /home/ch2067/rds/hpc-work/21cm_gen/ckpts/ldm_2x/ldm_epoch0089.pt \
  --vae_ckpt /home/ch2067/rds/hpc-work/21cm_gen/ckpts/vae_2x_final.pt \
  --out      /home/ch2067/rds/hpc-work/21cm_gen/eval_results/eval_2x_ep89.json \
  --data_root_ic    /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_IC \
  --data_root_astro /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro \
  --split test
