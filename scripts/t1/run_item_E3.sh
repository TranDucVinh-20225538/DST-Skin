#!/bin/bash
#SBATCH --job-name=t1_E3
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/E3_%j.out
# T1 item E3 (CPU only, no GPU): real-feature group recovery, 144 cells in the item-E list-0 order.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $HOME/DST-Skin
[ -f results/t1/E/raw/e3.jsonl.done ] && exit 0
$HOME/.venvs/crossfit-r3/bin/python scripts/t1/item_E3.py run $SLURM_CPUS_PER_TASK
echo DONE
