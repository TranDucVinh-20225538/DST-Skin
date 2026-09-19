#!/usr/bin/env bash
# CPU BN-shift check. Does not take the CIFAR GPU.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/bn_shift_check.log"
mkdir -p "${ROOT}/logs"
echo "=== BN-shift submit $(date) ===" | tee -a "${LOG}"
nohup srun --gres=gpu:0 -c 4 --mem=16G -t 02:00:00 \
  --job-name=dst-bn-shift \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== BN-shift start \$(date) ==='
    ${PY} scripts/bn_shift_check.py
    echo '=== BN-shift done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 3
squeue -u "$(whoami)" | tee -a "${LOG}"
