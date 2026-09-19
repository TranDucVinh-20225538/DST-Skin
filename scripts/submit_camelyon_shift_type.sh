#!/usr/bin/env bash
# Shift-type extract-only. Does not train. Does not touch official W or seed 42 ckpts.
# Do not scancel seed-variance / MIDOG zoo. Submit when a GPU is free.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/camelyon_shift_type.log"
mkdir -p "${ROOT}/logs"
echo "=== shift-type submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=48G -t 12:00:00 \
  --job-name=dst-shift-type \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== shift-type start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/camelyon_shift_type_ood.py --stage all --arm all --backbone all --num-workers 4
    ${PY} scripts/camelyon_shift_type_read.py
    echo '=== shift-type done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
