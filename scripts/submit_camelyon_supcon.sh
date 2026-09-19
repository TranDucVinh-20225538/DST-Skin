#!/usr/bin/env bash
# Job B: Camelyon CE+SupCon on R18 + DenseNet121. Does not overwrite CE zoo.
# Artifacts tagged *_supcon; rebuild skips them. Do not mix into official W.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/camelyon_supcon.log"
mkdir -p "${ROOT}/logs"
echo "=== Camelyon CE+SupCon submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=48G -t 24:00:00 \
  --job-name=dst-supcon \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== Camelyon CE+SupCon start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/camelyon_supcon_pilot.py --num-workers 4
    echo '=== Camelyon CE+SupCon done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
