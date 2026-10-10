#!/bin/bash
#SBATCH --job-name=t1_H
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/H_%A_%a.out
# T1 item H (H1-H3): units of PRECOMMIT_T1_addendum_H in list order, unit i -> task i % 2; resumable.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $HOME/DST-Skin
$HOME/.venvs/crossfit-r3/bin/python scripts/t1/item_H_run.py $SLURM_ARRAY_TASK_ID 2
echo DONE
