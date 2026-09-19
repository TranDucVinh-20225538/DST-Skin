#!/usr/bin/env bash
# MIDOG covariate-shift pilot: OpenMIBOOD ResNet-50 checkpoint, domains 1a/1b/1c only.
set -euo pipefail

ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
DL_LOG="${ROOT}/logs/midog_download.log"
LOG="${ROOT}/logs/midog_pilot.log"

mkdir -p "${ROOT}/logs"

echo "=== SLURM queue snapshot $(date) ===" | tee "${LOG}"
squeue -u "$(whoami)" -o "%.10i %.12P %.20j %.8T %.10M %.6D %R" | tee -a "${LOG}"

# CPU: checkpoint (~94MB) + 150 WSIs (~20GB) + 50×50 crops
nohup srun --gres=gpu:0 -c 4 --mem=32G -t 08:00:00 \
  --job-name=dst-midog-dl \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    echo '=== MIDOG download start \$(date) ==='
    ${PY} scripts/midog_pilot.py --stage download --num-workers 2
    echo '=== MIDOG download done \$(date) ==='
  " >> "${DL_LOG}" 2>&1 &

# GPU: wait for crops, then extract + gap metrics (no training)
nohup srun --gres=gpu:1 -c 8 --mem=32G -t 08:00:00 \
  --job-name=dst-midog-pilot \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    echo '=== MIDOG extract/analyze start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    MARKER=${ROOT}/data/raw/midog/RELEASE_id_csid.txt
    while [ ! -f \"\${MARKER}\" ]; do
      echo \"waiting for download (\${MARKER})...\"
      sleep 60
    done
    ${PY} scripts/midog_pilot.py --stage extract --seed 42 --batch-size 128 --num-workers 4
    ${PY} scripts/midog_pilot.py --stage analyze --seed 42
    echo '=== MIDOG pilot done \$(date) ==='
  " >> "${LOG}" 2>&1 &

echo "Submitted MIDOG download + extract; logs: ${DL_LOG}, ${LOG}"
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
