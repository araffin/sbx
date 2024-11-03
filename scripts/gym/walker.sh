#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate sbx

module load cuDNN/8.9.2.26-CUDA-12.2.0

python3 train_torch.py --benchmark=gym --env_name=Walker2d-v4

wait
