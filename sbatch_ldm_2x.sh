#!/bin/bash
#SBATCH -J ldm_2x
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/ldm_2x_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/ldm_2x_%j.err

mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/logs /home/ch2067/rds/hpc-work/21cm_gen/ckpts/ldm_2x

module purge
module load miniconda/3
module load cuda/11.8
source activate ldm

cd /home/ch2067/rds/hpc-work/21cm_gen

RESUME=""
LAST_CKPT=$(ls -t /home/ch2067/rds/hpc-work/21cm_gen/ckpts/ldm_2x/ldm_epoch*.pt 2>/dev/null | head -1)
if [ -n "$LAST_CKPT" ]; then
    RESUME="--resume $LAST_CKPT"
    echo "Resuming from $LAST_CKPT"
fi

/home/ch2067/.conda/envs/ldm/bin/python train_ldm.py \
  --vae_ckpt /home/ch2067/rds/hpc-work/21cm_gen/ckpts/vae_2x_final.pt \
  --out_dir  /home/ch2067/rds/hpc-work/21cm_gen/ckpts/ldm_2x \
  --data_root_ic    /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_IC \
  --data_root_astro /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro \
  --redshifts 7 8 9 10 11 12 13 \
  --epochs     125 \
  --batch_size 8 \
  --save_every 5 \
  --num_workers 8 \
  $RESUME
