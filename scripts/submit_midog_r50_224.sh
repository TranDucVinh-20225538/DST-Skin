#!/usr/bin/env bash
# In-house MIDOG R50 at 224² — matches R18 in-house protocol (job 59483).
# Does not overwrite the already-finished native-50 in-house run
# (resnet50_native50_*). Does not touch Camelyon zoo 59493.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/midog_r50_inhouse_224.log"
mkdir -p "${ROOT}/logs"
echo "=== MIDOG in-house R50 224 submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=32G -t 08:00:00 \
  --job-name=dst-midog-r50-224 \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== MIDOG in-house R50 224 start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} -c \"from src.datasets.midog_ood import _input_size; assert _input_size('resnet50')==224, _input_size('resnet50')\"
    ${PY} scripts/midog_pilot.py --stage all --backbone resnet50 --seed 42 --batch-size 64 --num-workers 4
    echo '=== MIDOG in-house R50 224 done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
