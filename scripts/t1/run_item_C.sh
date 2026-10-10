#!/bin/bash
#SBATCH --job-name=t1_C
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --array=0-3%4
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/C_%A_%a.out
# T1 item C (C1 + C3 + C4 interleaved in the item-C work-list order; PHASE=core) or C5 (PHASE=c5). Usage: sbatch run_item_C.sh PHASE
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=$HOME/.venvs/crossfit-r3/bin/python
cd $HOME/DST-Skin
PHASE=${1:-core}
$PY -c "
import json, sys
c0 = json.load(open('results/t1/C/c0.json'))
sys.exit(0 if c0['pass_C0'] else 'C0 failed: S3, C is not run')"
mkdir -p results/t1/C/raw
[ -s results/t1/C/raw/shard_c0.secs ] || $PY -c "import json; print(json.load(open('results/t1/C/c0.json'))['seconds'])" > results/t1/C/raw/shard_c0.secs
$PY scripts/t1/item_C_run.py $PHASE $SLURM_ARRAY_TASK_ID 4
echo DONE
