#!/bin/bash
#SBATCH -J imp4
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/imp4_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/imp4_%j.err

cd /home/ch2067/rds/hpc-work/21cm_gen

echo === search python modules ===
module avail python 2>&1 | head -40

echo === search miniconda module ===
module avail miniconda 2>&1 | head -10
module avail conda 2>&1 | head -10
module avail mamba 2>&1 | head -10

echo === check if any locally accessible python in /usr/local/software ===
ls /usr/local/software/spack/spack-views 2>&1 | head -10
find /usr/local/software -maxdepth 5 -name "python3.1*" -type f 2>/dev/null | head -5

echo === find venv numpy .so ===
find venv -name "*.so" 2>/dev/null | head -10

echo === try minimal numpy import with show_runtime ===
source venv/bin/activate
python << 'PY' 2>&1
import os
os.environ['NPY_DISABLE_CPU_FEATURES'] = 'AVX512_SKX AVX512F AVX512CD'
try:
    import numpy as np
    print('numpy OK', np.__version__)
except Exception as e:
    print('numpy FAIL:', type(e).__name__, str(e)[:200])
PY
