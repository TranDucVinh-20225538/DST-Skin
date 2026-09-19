#!/usr/bin/env bash
# Skin EffV2-S @224. Does not overwrite native-384. Skip zoo W mix.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/skin_effv2_224.log"
mkdir -p "${ROOT}/logs"
echo "=== Skin EffV2-S@224 submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=32G -t 08:00:00 \
  --job-name=dst-skin-effv2-224 \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== Skin EffV2-S@224 start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/skin_newcnn.py --stage all --backbone efficientnet_v2_s --seed 42 --input-size 224 --num-workers 4
    echo '=== Skin EffV2-S@224 done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
