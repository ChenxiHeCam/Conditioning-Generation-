#!/bin/bash
#SBATCH -J mat2npy
#SBATCH -A PENG-SL3-CPU
#SBATCH -p icelake
#SBATCH --qos=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/mat2npy_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/mat2npy_%j.err

mkdir -p /home/ch2067/rds/hpc-work/21cm_gen/logs
module purge
source /home/ch2067/rds/hpc-work/21cm_gen/venv/bin/activate
cd /home/ch2067/rds/hpc-work/21cm_gen
python convert_mat_npy.py
