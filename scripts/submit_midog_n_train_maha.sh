#!/usr/bin/env bash
# CPU-only H4 test. Does not take the Camelyon zoo GPU (59493).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/midog_n_train_maha.log"
mkdir -p "${ROOT}/logs"
echo "=== MIDOG n-train Maha submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:0 -c 8 --mem=48G -t 06:00:00 \
  --job-name=dst-maha-n1896 \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== MIDOG n-train Maha start \$(date) ==='
    ${PY} scripts/midog_n_train_maha.py
    echo '=== MIDOG n-train Maha done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
