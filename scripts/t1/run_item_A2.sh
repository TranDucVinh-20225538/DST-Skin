#!/bin/bash
#SBATCH --job-name=t1_A2
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=75G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --array=0-25%3
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/A2_%A_%a.out
# T1 item A2: one array task per (backbone, fold); tasks ordered by first appearance in the item-A work-list order.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=$HOME/.venvs/crossfit-r3/bin/python
cd $HOME/DST-Skin
read BB F < <($PY -c "
import json
seen=[]
for u in json.load(open('results/t1/PRECOMMIT_T1_addendum_A.json'))['work_lists'][0]['execution_order']:
    if u.startswith('a2|'):
        k=tuple(u.split('|')[1:3])
        if k not in seen: seen.append(k)
b,f=seen[$SLURM_ARRAY_TASK_ID]; print(b, f[1:])")
$PY scripts/t1/item_A2.py $BB $F
echo DONE
