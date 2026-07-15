#!/bin/bash
#SBATCH -J infer_2x_mz
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/infer_2x_mz_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/infer_2x_mz_%j.err

module purge
module load miniconda/3
module load cuda/11.8
source activate ldm
cd /home/ch2067/rds/hpc-work/21cm_gen

for z in 7 8 9 10 11 12 13; do
    if [ $z -eq 10 ]; then NC=100; else NC=10; fi
    OUT=/home/ch2067/rds/hpc-work/21cm_gen/eval_results/ff_2x_z${z}
    if [ -f $OUT/metrics.json ]; then
        echo "=== z=$z already done, skip ==="
        continue
    fi
    mkdir -p $OUT
    echo "=== z=$z n_cubes=$NC ==="
    /home/ch2067/.conda/envs/ldm/bin/python infer_baselines.py \
        --model latent \
        --vae_ckpt /home/ch2067/rds/hpc-work/21cm_gen/ckpts/vae_2x_final.pt \
        --ldm_ckpt /home/ch2067/rds/hpc-work/21cm_gen/ckpts/ldm_2x/ldm_epoch0089.pt \
        --data_root /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro \
        --out_dir   $OUT \
        --split test \
        --redshift $z \
        --n_cubes $NC \
        --steps 30
done
echo "=== DONE all z for 2x ==="
