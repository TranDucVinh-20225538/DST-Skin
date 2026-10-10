#!/bin/bash
#SBATCH --job-name=t1_A0
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/A0_%j.out
# T1 item A0: A0a toy reproduction, A0c unit conversion, A0b Track A Delta reproduction.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=$HOME/.venvs/crossfit-r3/bin/python
S=$HOME/DST-Skin/scripts/t1/item_A0.py
O=$HOME/DST-Skin/results/t1/A
cd $HOME/DST-Skin
[ -s $O/a0a.json ] || $PY $S a0a
[ -s $O/a0c.json ] || $PY $S a0c
[ -s $O/a0b.json ] || $PY $S a0b
echo DONE
