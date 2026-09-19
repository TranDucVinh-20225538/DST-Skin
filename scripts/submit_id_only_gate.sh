#!/usr/bin/env bash
# Direction #2 ID-only Maha/MSP gate. CPU only. Do not take a GPU from Job B.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/id_only_maha_msp_gate.log"
mkdir -p "${ROOT}/logs"
echo "=== ID-only gate submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun -c 8 --mem=48G -t 06:00:00 \
  --job-name=dst-id-gate \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== ID-only gate start' \$(date)
    ${PY} scripts/id_only_maha_msp_gate.py
    echo '=== ID-only gate done' \$(date)
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
