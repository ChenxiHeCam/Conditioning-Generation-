#!/bin/bash
#SBATCH -J imp5
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/imp5_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/imp5_%j.err

echo === explore rhel8-a100 spack views ===
ls /usr/local/software/spack/spack-views/ 2>&1 | grep a100

echo
echo === pick latest a100 view ===
LATEST=$(ls -d /usr/local/software/spack/spack-views/rhel8-a100-* 2>/dev/null | tail -1)
echo "Using $LATEST"
ls "$LATEST/bin/" 2>&1 | grep -E "^python|^pip" | head
ls "$LATEST/bin/" 2>&1 | head -30

echo
echo === miniconda3 module ===
module load miniconda/3 2>&1 | head -3
which python
python --version
python -c "import numpy; print('miniconda numpy OK', numpy.__version__)" 2>&1
python -c "import torch; print('miniconda torch OK')" 2>&1
module unload miniconda/3 2>&1

echo
echo === try a100 view python directly ===
A100_PY=$(ls /usr/local/software/spack/spack-views/rhel8-a100-*/bin/python3 2>/dev/null | tail -1)
echo "Using $A100_PY"
if [ -n "$A100_PY" ]; then
    "$A100_PY" -c "print('a100 view python OK')" 2>&1
    "$A100_PY" -c "import sys; print(sys.version)" 2>&1
    "$A100_PY" -c "import numpy; print('numpy', numpy.__version__)" 2>&1 || echo "(no numpy)"
fi
