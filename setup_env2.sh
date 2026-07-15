#!/bin/bash
#SBATCH -J setup_env2
#SBATCH -A FIALKOV-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=intr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /home/ch2067/rds/hpc-work/21cm_gen/logs/setup2_%j.out
#SBATCH -e /home/ch2067/rds/hpc-work/21cm_gen/logs/setup2_%j.err

cd /home/ch2067/rds/hpc-work/21cm_gen
module purge
module load miniconda/3
module load cuda/11.8

echo "=== host: $(hostname) ==="
grep "model name" /proc/cpuinfo | head -1

conda env remove -n ldm -y 2>/dev/null
conda create -y -n ldm -c conda-forge python=3.11 pip

ENV=/home/ch2067/.conda/envs/ldm
$ENV/bin/pip install --upgrade pip
$ENV/bin/pip install \
    torch==2.4.0 \
    numpy==1.26.4 \
    scipy==1.11.4 \
    matplotlib \
    scikit-learn \
    --index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple

echo "=== test imports ==="
$ENV/bin/python -c "
import torch, numpy, scipy
print('torch:', torch.__version__)
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
import sys; sys.path.insert(0, '/home/ch2067/rds/hpc-work/21cm_gen')
from ldm_unet import LDMUNet3D
from ldm_dataset import build_train_dataset
print('repo imports OK')
"
