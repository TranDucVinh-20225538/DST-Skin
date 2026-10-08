#!/bin/bash
#SBATCH --job-name=r3-5b-v2k10
#SBATCH --exclude=node002
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/r3_5b_v2k10_%j.out
set -euo pipefail
# Virchow2 knn K = 10 timed out at 7 h on CPU (job 64379_6); rerun on GPU after the exact GPU/CPU equality check (job 64626).
R=$HOME/DST-Skin
export PYTHONPATH=$R OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export R3_INPUTS=$HOME/r3work/item1/inputs
cd $R
$HOME/.venvs/crossfit-r3/bin/python scripts/r3/item5b_stepB.py --fm virchow2 --scorer knn_mean_cosine --ks 10 --gpu-knn --tag _k10
