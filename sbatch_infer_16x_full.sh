#!/bin/bash
#SBATCH -J infer_16x_full
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/infer_16x_full_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/infer_16x_full_%j.err

module purge
module load miniconda/3
module load cuda/11.8
source activate ldm
cd /home/ch2067/rds/hpc-work/21cm_gen

# Auto-pick latest 16x ckpt (will be ep89 now, ep124 after training finishes)
LDM_CKPT=$(ls -t /home/ch2067/rds/hpc-work/21cm_gen/ckpts/ldm_16x/ldm_epoch*.pt 2>/dev/null | head -1)
echo "Using LDM ckpt: $LDM_CKPT"
EP=$(basename $LDM_CKPT | sed 's/ldm_epoch\([0-9]*\).pt/\1/')
echo "Epoch: $EP"

for z in 7 8 9 10 11 12 13; do
    if [ $z -eq 10 ]; then NC=100; else NC=10; fi
    OUT=/home/ch2067/rds/hpc-work/21cm_gen/eval_results/ff_16x_test_z${z}_ep${EP}
    if [ -f $OUT/timing.json ]; then
        echo "=== 16x ep$EP z=$z already done, skip ==="
        continue
    fi
    mkdir -p $OUT
    echo "=== 16x ep$EP z=$z n_cubes=$NC ==="
    /home/ch2067/.conda/envs/ldm/bin/python infer_baselines.py \
        --model latent \
        --vae_ckpt /home/ch2067/rds/hpc-work/21cm_gen/ckpts/vae_16x_final.pt \
        --ldm_ckpt $LDM_CKPT \
        --data_root /home/ch2067/rds/hpc-work/ASR21cm/datasets/varying_astro \
        --out_dir   $OUT \
        --split test \
        --redshift $z \
        --n_cubes $NC \
        --steps 30
done
echo "=== DONE all z for 16x ==="
