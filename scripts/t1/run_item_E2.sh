#!/bin/bash
#SBATCH --job-name=t1_E2
#SBATCH --partition=defq
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/E2_%j.out
# T1 item E2 (CPU only, no GPU): label-free group recovery, 75 configs in the item-E list-0 order.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $HOME/DST-Skin
[ -f results/t1/E/raw/e2_0.jsonl.done ] && exit 0
$HOME/.venvs/crossfit-r3/bin/python scripts/t1/item_E2.py run 0 1 $SLURM_CPUS_PER_TASK
echo DONE
