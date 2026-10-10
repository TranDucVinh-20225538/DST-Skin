#!/bin/bash
#SBATCH --job-name=t1_E
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/E1_%A_%a.out
# T1 item E (GPU part): unit test (task 0), then the GPU units (E1, E4) of list j = 0 (regime a, a'), then of list j = 1 (regime b); unit i -> task i % 2.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $HOME/DST-Skin
PY=$HOME/.venvs/crossfit-r3/bin/python
if [ ! -f results/t1/E/unit_test.json ]; then
  if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then $PY scripts/t1/item_E1.py unittest || { echo "unit test failed"; exit 1; }
  else while [ ! -f results/t1/E/unit_test.json ]; do sleep 30; done; fi
fi
grep -q '"pass_": true' results/t1/E/unit_test.json || { echo "unit test failed"; exit 1; }
for L in 0 1; do
  [ -f results/t1/E/raw/egpu_${L}_$SLURM_ARRAY_TASK_ID.jsonl.done ] || $PY scripts/t1/item_E1.py run $L $SLURM_ARRAY_TASK_ID 2
done
echo DONE
