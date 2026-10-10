#!/bin/bash
#SBATCH --job-name=t1_itemI
#SBATCH --partition=defq
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/itemI_%j.out
# T1 item I (CPU): registry, inventory with sha256, Camelyon manifest / MANIFEST_OK.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $HOME/DST-Skin
$HOME/.venvs/crossfit-r3/bin/python scripts/t1/item_I.py
echo DONE
