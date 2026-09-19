#!/usr/bin/env bash
# Job A: T* / logit-norm vs MSP-jump on saved Camelyon-8 logits. CPU only.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/camelyon_temp_jump.log"
mkdir -p "${ROOT}/logs"
echo "=== temp-jump submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun -c 8 --mem=48G -t 04:00:00 \
  --job-name=dst-temp-jump \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== temp-jump start \$(date) ==='
    ${PY} scripts/camelyon_temp_jump_corr.py
    echo '=== temp-jump done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
