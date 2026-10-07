#!/bin/bash
#SBATCH --job-name=r3-8-pilot
#SBATCH --exclude=node002
#SBATCH --nice=10000
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --output=logs/r3_8_pilot_%j.out
# R3 item 8 / P2-a OOD ceiling pilot (CPU), plan: results/r3/8/pilot_plan.md
set -euo pipefail
R=$HOME/DST-Skin
PY=$HOME/.venvs/crossfit-r3/bin/python
export PYTHONPATH=$R OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PILOT_RAW=${PILOT_RAW:-$HOME/r3work/item8/pilot_raw}
TMPDIR=$(mktemp -d /tmp/r3_8_XXXXXX)
export TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT
cd $R
$PY scripts/r3/item8_ood_pilot.py
