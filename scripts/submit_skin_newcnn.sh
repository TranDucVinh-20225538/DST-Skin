#!/usr/bin/env bash
# Skin ISIC→PAD x 5 new CNN backbones. One GPU; do not pin to node002
# (59493/59589 keep their allocations). seed=42, per-arch standard input size.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/skin_newcnn.log"
mkdir -p "${ROOT}/logs"
echo "=== Skin NEW-CNN submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=32G -t 24:00:00 \
  --job-name=dst-skin-newcnn \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== Skin NEW-CNN start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} -c \"from src.models.cnn_family import NEW_CNN_BACKBONES, input_size; print({n: input_size(n) for n in NEW_CNN_BACKBONES})\"
    ${PY} scripts/skin_newcnn.py --stage all --backbone new --seed 42 --num-workers 4
    echo '=== Skin NEW-CNN done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
