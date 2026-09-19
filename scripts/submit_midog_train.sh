#!/usr/bin/env bash
# Train MIDOG ResNet-18 + EffB3 on the existing 1a adapter; do not touch frozen OpenMIBOOD R50.
# Queues a second GPU; Camelyon full (59482) stays running.
set -euo pipefail

ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/midog_train_r18_effb3.log"
MARKER="${ROOT}/data/raw/midog/RELEASE_id_csid.txt"

mkdir -p "${ROOT}/logs"

if [ ! -f "${MARKER}" ]; then
  echo "MIDOG crops not ready: ${MARKER}" >&2
  exit 1
fi

echo "=== MIDOG R18/EffB3 train submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" -o "%.10i %.12P %.20j %.8T %.10M %R" | tee -a "${LOG}"

nohup srun --gres=gpu:1 -c 8 --mem=32G -t 08:00:00 \
  --job-name=dst-midog-r18b3 \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    echo '=== MIDOG R18/EffB3 start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/midog_pilot.py --stage all --backbone train --seed 42 --batch-size 64 --num-workers 4
    echo '=== MIDOG R18/EffB3 done \$(date) ==='
  " >> "${LOG}" 2>&1 &

echo "Submitted dst-midog-r18b3; log: ${LOG}"
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
