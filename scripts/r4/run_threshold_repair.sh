#!/bin/bash
# R4 analysis 1, one backbone per array task, CPU only, BLAS threads = CPUs.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$HOME/DST-Skin
MODELS=(uni virchow2 dinov2_vitb14 dinov2_vitl14 conch_v1_5 resnet50 convnext_tiny)
cd "$HOME/DST-Skin"
"$HOME/.venvs/crossfit-r3/bin/python" scripts/r4/threshold_repair.py run "${MODELS[$SLURM_ARRAY_TASK_ID]}"
