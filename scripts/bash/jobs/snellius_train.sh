#!/bin/bash

#SBATCH --job-name=Training_SwinUnet
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=d.koopmans@sron.nl
#SBATCH --output=/home/dkoopmans/snellius_training_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --begin=now

# Model name
MODEL_NAME="SwinUnet"

# Config file
CONFIG_FILE="TrainConfig_snellius.yaml"

# Directory to configuration files
CONFIG_DIR="/home/dkoopmans/DeepSPIRE/configs/${MODEL_NAME}/train/${CONFIG_FILE}"

# Snellius
# module load 2023 
# module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1 # This installation has a memory leak!
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

PYENV_DIR=$HOME/.venvs/deepSPIRE_tf_cuda_2024
source ${PYENV_DIR}/bin/activate # contains all our Python packages

source ~/.bashrc # important for using legacy keras

# Figure out where nvcc (and thus CUDA) lives
CUDA_ROOT="$(dirname "$(dirname "$(which nvcc)")")"

# Point XLA at the libdevice folder in your 12.6.0 install:
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_ROOT}"

# 4) XLA & GPU flags
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"
export TF_ENABLE_XLA=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

python "/home/dkoopmans/DeepSPIRE/scripts/train/train.py" --config  "${CONFIG_DIR}"