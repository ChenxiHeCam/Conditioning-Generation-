#!/bin/bash
#SBATCH -J imp3
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/imp3_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/imp3_%j.err

cd /home/ch2067/rds/hpc-work/21cm_gen

echo === venv python info ===
file venv/bin/python 2>&1
ldd venv/bin/python 2>&1 | head -5

echo
echo === try direct system python ===
which python3
python3 --version
python3 -c "import sys; print(sys.version_info)" 2>&1

echo
echo === available modules ===
module avail 2>&1 | head -40

echo
echo === try loading python module ===
module load python/3.10 2>&1 | head -3
python3 -c "print('basic OK')" 2>&1
python3 -c "import numpy; print('numpy OK', numpy.__version__)" 2>&1 || true

echo
echo === check venv site-packages numpy ===
ls venv/lib/python*/site-packages/numpy/core/_multiarray_umath* 2>&1 | head
file venv/lib/python*/site-packages/numpy/core/_multiarray_umath*.so 2>&1 | head
