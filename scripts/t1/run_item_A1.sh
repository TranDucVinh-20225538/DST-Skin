#!/bin/bash
#SBATCH --job-name=t1_A1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=75G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --array=0-3
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/A1_%A_%a.out
# T1 item A1: per-cell descriptors / predictions; cells taken from the item-A work list execution order (a1 units),
# shard = position % 4. Resumable (cells with an output are skipped).
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=$HOME/.venvs/crossfit-r3/bin/python
cd $HOME/DST-Skin
CELLS=$($PY -c "
import json
o=[u.split('|')[1] for u in json.load(open('results/t1/PRECOMMIT_T1_addendum_A.json'))['work_lists'][0]['execution_order'] if u.startswith('a1|')]
print(' '.join(c for i,c in enumerate(o) if i % 4 == $SLURM_ARRAY_TASK_ID))")
$PY scripts/t1/item_A1_cell.py $CELLS
echo DONE
