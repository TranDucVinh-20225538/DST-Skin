#!/usr/bin/env bash
# Camelyon EffB3 @ 224 (resolution control for cua 2 founding cell).
# Does not cancel zoo 59493 or FM 59589. Does not overwrite EffB3@300.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/camelyon17_effb3_224.log"
mkdir -p "${ROOT}/logs"
echo "=== EffB3@224 submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=48G -t 16:00:00 \
  --job-name=dst-effb3-224 \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== EffB3@224 start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/camelyon17_pilot.py --stage all --backbone effb3 --seed 42 --train-frac 1.0 --input-size 224 --num-workers 4
    ${PY} scripts/compare_effb3_224.py
    echo '=== EffB3@224 done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
