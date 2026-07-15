#!/bin/bash
#SBATCH -J imp2
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/imp2_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/imp2_%j.err

cd /home/ch2067/rds/hpc-work/21cm_gen
module purge
module load cuda/11.8
source venv/bin/activate

# Disable AVX-512 paths
export NPY_DISABLE_CPU_FEATURES="AVX512_SKX AVX512_CNL AVX512_ICL AVX512F AVX512CD AVX512_KNL AVX512_KNM"
export OPENBLAS_CORETYPE=ZEN

echo === numpy ===
python -c "import numpy; print('numpy OK', numpy.__version__)" 2>&1
echo === scipy ===
python -c "import scipy; print('scipy OK', scipy.__version__)" 2>&1
echo === torch ===
python -c "import torch; print('torch OK', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1
echo === train_ldm imports ===
python -c "from ldm_unet import LDMUNet3D; from ldm_dataset import build_train_dataset; print('imports OK')" 2>&1

echo
echo === wheel info ===
python -c "import numpy; print(numpy.show_runtime())" 2>&1
