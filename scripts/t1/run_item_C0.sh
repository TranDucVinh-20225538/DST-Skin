#!/bin/bash
#SBATCH --job-name=t1_C0
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=75G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/C0_%j.out
# T1 item C0: bundle toy reproduction (>= 64 replicates) and seconds per replicate by (d, N).
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $HOME/DST-Skin
[ -s results/t1/C/c0.json ] && exit 0
$HOME/.venvs/crossfit-r3/bin/python scripts/t1/item_C0.py
echo DONE
