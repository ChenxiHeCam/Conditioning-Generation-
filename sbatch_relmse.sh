#!/bin/bash
#SBATCH -J relmse -A FIALKOV-SL3-GPU -p ampere --qos=intr
#SBATCH --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:20:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/relmse_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/relmse_%j.err
module purge; module load miniconda/3
source activate ldm
cd /home/ch2067/rds/hpc-work/21cm_gen
python recompute_relmse.py
