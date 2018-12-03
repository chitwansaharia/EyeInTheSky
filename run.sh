#!/bin/bash
##SBATCH --time=2  # wall-clock time limit in minutes
#SBATCH -p special
#SBATCH --gres=gpu:4,gpu_mem:8000M  # number of GPUs (keep it at 3) and memory limit
#SBATCH --cpus-per-task=2            # number of CPU cores
#SBATCH --output=logging/LargerModel_BS15.txt       # output file
##SBATCH --error=error.txt # error file
CUDA_VISIBLE_DEVICES=1
python3 train.py --model LargeModel_BS15 --batch-size  15