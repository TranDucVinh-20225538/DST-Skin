#!/bin/bash
#SBATCH --job-name=t1_D_real
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/D_real_%A_%a.out
# T1 item D (D3-D5): real cells, two tasks (unit j -> task j % 2) in the item-D work-list order.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $HOME/DST-Skin
[ -f results/t1/D/raw/dreal_$SLURM_ARRAY_TASK_ID.jsonl.done ] && exit 0
$HOME/.venvs/crossfit-r3/bin/python scripts/t1/item_D_real.py $SLURM_ARRAY_TASK_ID 2
echo DONE
