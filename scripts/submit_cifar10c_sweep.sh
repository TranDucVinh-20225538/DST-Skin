#!/usr/bin/env bash
# CIFAR-10-C severity sweep: inference only. Does not touch Camelyon/MIDOG jobs.
set -euo pipefail

ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/cifar10c_sweep.log"
MARKER="${ROOT}/data/raw/cifar10c/labels.npy"

mkdir -p "${ROOT}/logs"

echo "=== CIFAR-10-C sweep submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" -o "%.10i %.12P %.20j %.8T %.10M %R" | tee -a "${LOG}"

nohup srun --gres=gpu:1 -c 4 --mem=16G -t 06:00:00 \
  --job-name=dst-cifar10c \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    echo '=== CIFAR-10-C sweep start \$(date) ==='
    while [ ! -f ${MARKER} ]; do
      echo 'waiting for CIFAR-10-C download...'
      sleep 30
    done
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/cifar10c_severity_sweep.py --backbone all --batch-size 256 --num-workers 4
    ${PY} scripts/domain_shift_auc.py
    echo '=== CIFAR-10-C sweep done \$(date) ==='
  " >> "${LOG}" 2>&1 &

echo "Submitted dst-cifar10c; log: ${LOG}"
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
