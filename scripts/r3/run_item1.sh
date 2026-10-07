#!/bin/bash
# R3 item 1: recompute_paper_ci.py --protocol paper_2fold for one cell (line SLURM_ARRAY_TASK_ID+1 of $LIST).
# Env: LIST (cell names), WORK (item1 dir with inputs/), NJOBS, SCORERS (default: mahalanobis knn vim).
set -euo pipefail
cell=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
seed=$(grep -oP '_s\K[0-9]+' <<< "$cell" | tail -n 1)
export OMP_NUM_THREADS=${BLAS:-1} OPENBLAS_NUM_THREADS=${BLAS:-1} MKL_NUM_THREADS=${BLAS:-1}
export JOBLIB_TEMP_FOLDER=$(mktemp -d /tmp/r3_joblib_XXXXXX)
trap 'rm -rf "$JOBLIB_TEMP_FOLDER"' EXIT
head=()
[ -f "$WORK/inputs/${cell}_head.npz" ] && head=(--vim-head "$WORK/inputs/${cell}_head.npz")
out="$WORK/out/${cell}${OUTSUF:-}"
mkdir -p "$out"
[ -s "$out/paper_ci.csv" ] && { echo "done $cell"; exit 0; }
export CROSSFIT_DIR="$HOME/handoff/crossfit-ood_v3"
cd "$CROSSFIT_DIR"
"$HOME/.venvs/crossfit-r3/bin/python" "${SCRIPT:-scripts/recompute_paper_ci.py}" --input "$WORK/inputs/${cell}.npz" \
  --protocol paper_2fold --scorers ${SCORERS:-mahalanobis knn vim} "${head[@]}" --threshold 0.02 \
  --seed "$seed" --n-jobs "${NJOBS:-4}" --out "$out" ${EXTRA:-}
