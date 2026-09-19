#!/usr/bin/env bash
# #4 ViT-B/16 @224 Camelyon. Skip official W mix (rebuild ignores vit_b_16).
# Queues if A100 busy. Precommit: MSP >= 0.6947.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/camelyon17_vit_b_16.log"
mkdir -p "${ROOT}/logs"
echo "=== Camelyon ViT-B/16 submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=48G -t 24:00:00 \
  --job-name=dst-vit-cam \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== Camelyon ViT-B/16 start' \$(date)
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/camelyon17_pilot.py --stage all --backbone vit_b_16 --seed 42 --train-frac 1.0 --num-workers 4
    echo '=== Camelyon ViT-B/16 done' \$(date)
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
