#!/usr/bin/env bash
# #5 iWildCam train R18+R50+DenseNet. Waits for metadata.csv. Queues if GPU busy.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/iwildcam_pilot.log"
mkdir -p "${ROOT}/logs"
echo "=== iWildCam pilot submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=48G -t 24:00:00 \
  --job-name=dst-iwild \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== iWildCam pilot start' \$(date)
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/iwildcam_pilot.py --stage all --backbone all --num-workers 4
    echo '=== iWildCam pilot done' \$(date)
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
