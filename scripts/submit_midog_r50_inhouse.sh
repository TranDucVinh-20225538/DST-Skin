#!/usr/bin/env bash
# In-house MIDOG ResNet-50 (same epochs/seed/split as R18/EffB3).
# Does not wait for Camelyon-8. Does not retrain R18/EffB3.
# OpenMIBOOD R50 scores/features already copied to *_openmibood_*.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/midog_r50_inhouse.log"
mkdir -p "${ROOT}/logs"
echo "=== MIDOG in-house R50 submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=32G -t 08:00:00 \
  --job-name=dst-midog-r50 \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== MIDOG in-house R50 start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/midog_pilot.py --stage all --backbone resnet50 --seed 42 --batch-size 64 --num-workers 4
    echo '=== MIDOG in-house R50 done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
