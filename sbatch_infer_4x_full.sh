#!/bin/bash
#SBATCH -J infer_4x_full
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/infer_4x_full_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/infer_4x_full_%j.err

module purge
module load miniconda/3
module load cuda/11.8
source activate ldm
cd /home/ch2067/rds/hpc-work/21cm_gen

for split in test; do
    for z in 7 8 9 10 11 12 13; do
        if [ $z -eq 10 ]; then NC=100; else NC=10; fi
        OUT=/home/ch2067/rds/hpc-work/21cm_gen/eval_results/ff_4x_${split}_z${z}
        if [ -f $OUT/timing.json ]; then
            echo "=== 4x $split z=$z already done, skip ==="
            continue
        fi
        mkdir -p $OUT
        echo "=== 4x $split z=$z n_cubes=$NC ==="
        /home/ch2067/.conda/envs/ldm/bin/python infer_baselines.py \
            --model latent \
            --vae_ckpt /home/ch2067/rds/hpc-work/21cm_gen/ckpts/vae_4x_final.pt \
            --ldm_ckpt /home/ch2067/rds/hpc-work/21cm_gen/ckpts/ldm_4x/ldm_epoch0124.pt \
            --data_root /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro \
            --out_dir   $OUT \
            --split $split \
            --redshift $z \
            --n_cubes $NC \
            --steps 30
    done
done
echo "=== DONE all splits and z for 4x ==="
