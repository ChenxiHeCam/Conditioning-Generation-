#!/bin/bash
#SBATCH -J vae_eval
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/vae_eval_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/vae_eval_%j.err

mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/logs /home/ch2067/rds/hpc-work/21cm_gen/eval_results/vae_only
module purge
module load miniconda/3
module load cuda/11.8
source activate ldm
cd /home/ch2067/rds/hpc-work/21cm_gen

for z in 7 8 9 10 11 12 13; do
    if [ $z -eq 10 ]; then NC=100; else NC=10; fi
    OUT=/home/ch2067/rds/hpc-work/21cm_gen/eval_results/vae_only/z${z}
    if [ -f $OUT/summary.json ]; then
        echo "=== z=$z already done, skip ==="
        continue
    fi
    mkdir -p $OUT
    echo "=== vae-only z=$z n_cubes=$NC ==="
    /home/ch2067/.conda/envs/ldm/bin/python vae_only_eval.py \
        --vae_ckpts \
          2x:/home/ch2067/rds/hpc-work/21cm_gen/ckpts/vae_2x_final.pt \
          4x:/home/ch2067/rds/hpc-work/21cm_gen/ckpts/vae_4x_final.pt \
          8x:/home/ch2067/rds/hpc-work/21cm_gen/ckpts/vae_8x_final.pt \
          16x:/home/ch2067/rds/hpc-work/21cm_gen/ckpts/vae_16x_final.pt \
        --data_root /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro \
        --out_dir   $OUT \
        --redshift $z \
        --n_cubes $NC
done
echo "=== DONE all z ==="
