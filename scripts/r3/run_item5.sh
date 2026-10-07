#!/bin/bash
#SBATCH --job-name=r3-5-crossfit
#SBATCH --exclude=node002
#SBATCH --nice=10000
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --array=0-1%2
#SBATCH --output=logs/r3_5_%A_%a.out
# R3 item 5 (post-hoc), CPU. Task 0: part (a) + part (b) fold 0; task 1: part (b) fold 1.
set -euo pipefail
R=$HOME/DST-Skin
PY=$HOME/.conda/envs/torch-env/bin/python
export PYTHONPATH=$R OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
JOBLIB_TEMP_FOLDER=$(mktemp -d /tmp/r3_5_XXXXXX)
export JOBLIB_TEMP_FOLDER
trap 'rm -rf "$JOBLIB_TEMP_FOLDER"' EXIT
cd $R
F=$SLURM_ARRAY_TASK_ID
if [ "$F" = 0 ]; then $PY scripts/r3/item5_a_camelyon_fm.py || echo "ITEM5A FAILED"; fi
for s in 42 43 44; do
  for a in resnet50 convnext_tiny densenet121; do
    $PY scripts/r3/item5_b_isbi_patch2.py --arch $a --seed $s --fold $F
  done
done
