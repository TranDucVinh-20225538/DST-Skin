#!/usr/bin/env bash
# Camelyon full x 5 new CNN backbones. Queues behind CIFAR-10-C; does not cancel it.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/camelyon17_newcnn.log"
mkdir -p "${ROOT}/logs"
echo "=== Camelyon NEW-CNN submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=48G -t 72:00:00 \
  --job-name=dst-camel-newcnn \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== Camelyon NEW-CNN start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/camelyon17_pilot.py --stage all --backbone new --seed 42 --train-frac 1.0 --num-workers 4
    echo '=== Camelyon NEW-CNN done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
