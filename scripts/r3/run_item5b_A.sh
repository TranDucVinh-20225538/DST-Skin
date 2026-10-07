#!/bin/bash
#SBATCH --job-name=r3-5b-A
#SBATCH --exclude=node002
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --output=logs/r3_5b_A_%j.out
set -euo pipefail
cd $HOME/DST-Skin
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
$HOME/.conda/envs/torch-env/bin/python scripts/r3/item5b_stepA.py
