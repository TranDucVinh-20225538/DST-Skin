#!/bin/bash
# R3 item 1 stop-check array runner: one cell per task (line SLURM_ARRAY_TASK_ID+1 of $LIST).
set -euo pipefail
export OMP_NUM_THREADS=${BLAS:-4} OPENBLAS_NUM_THREADS=${BLAS:-4} MKL_NUM_THREADS=${BLAS:-4}
cd $HOME/DST-Skin
exec $HOME/.venvs/crossfit-r3/bin/python scripts/r3/check_tracka_match.py ${EXTRA:-}
