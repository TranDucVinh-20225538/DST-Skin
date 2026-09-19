#!/usr/bin/env bash
# Frozen pathology FM OOD. Does not cancel Camelyon zoo 59493.
# Token is read from the HuggingFace cache, never from this file.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/fm_ood.log"
HF_TOKEN_FILE="/data2/cmdir/home/toandq/.cache/huggingface/token"
mkdir -p "${ROOT}/logs" "${ROOT}/outputs/reports/fm"
echo "=== FM OOD submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
if [ ! -s "${HF_TOKEN_FILE}" ]; then
  echo "Missing HuggingFace token file; cannot download gated FMs." | tee -a "${LOG}"
  exit 1
fi
nohup srun --gres=gpu:1 -c 8 --mem=48G -t 48:00:00 \
  --job-name=dst-fm-ood \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    export HF_TOKEN=\$(tr -d '[:space:]' < ${HF_TOKEN_FILE})
    export HUGGINGFACE_HUB_TOKEN=\$HF_TOKEN
    echo '=== FM OOD start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/fm_ood_pilot.py --stage all --fm uni gigapath phikon --domain midog camelyon17 --batch-size 32 --num-workers 4
    echo '=== FM OOD done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
