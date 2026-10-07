#!/bin/bash
# R3 item 3 whitening stage, one backbone per array task.
set -euo pipefail
m=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HOME/r3work/item3_models.txt")
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$HOME/DST-Skin
cd "$HOME/DST-Skin"
"$HOME/.conda/envs/torch-env/bin/python" scripts/r3/item3_mechanism.py whiten --model "$m"
