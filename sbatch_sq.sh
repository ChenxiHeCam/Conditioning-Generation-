#!/bin/bash
#SBATCH -J sq_plot
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/sq_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/sq_%j.err

module purge
module load miniconda/3
module load cuda/11.8
source activate ldm
cd /home/ch2067/rds/hpc-work/21cm_gen
/home/ch2067/.conda/envs/ldm/bin/python plot_squeezed_quantile.py --out plots/
