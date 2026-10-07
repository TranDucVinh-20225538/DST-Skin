#!/bin/bash
# R3 item 6 on one GPU (GPU kNN), BLAS threads = CPUs.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$HOME/DST-Skin
cd "$HOME/DST-Skin"
"$HOME/.venvs/crossfit-r3/bin/python" scripts/r3/item6_dose.py
