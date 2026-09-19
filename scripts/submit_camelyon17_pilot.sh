#!/usr/bin/env bash
# Camelyon17 covariate-shift pilot: download (~10GB) + 1 seed × 3 backbones.
set -euo pipefail

ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/camelyon17_pilot.log"
DL_LOG="${ROOT}/logs/camelyon17_download.log"
TRAIN_FRAC="${TRAIN_FRAC:-0.05}"

mkdir -p "${ROOT}/logs"

echo "=== SLURM queue snapshot $(date) ===" | tee "${LOG}"
squeue -u "$(whoami)" -o "%.10i %.12P %.20j %.8T %.10M %.6D %R" | tee -a "${LOG}"

# CPU download job (10GB tarball, no GPU)
nohup srun --gres=gpu:0 -c 4 --mem=16G -t 04:00:00 \
  --job-name=dst-camelyon-dl \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    echo '=== Camelyon17 download start \$(date) ==='
    ${PY} scripts/camelyon17_pilot.py --stage download --download --batch-size 8 --num-workers 2
    ${PY} scripts/camelyon17_pilot.py --stage probe
    echo '=== Camelyon17 download done \$(date) ==='
  " >> "${DL_LOG}" 2>&1 &

# GPU pilot — waits for RELEASE marker, then train/extract/analyze
nohup srun --gres=gpu:1 -c 8 --mem=32G -t 24:00:00 \
  --job-name=dst-camelyon-pilot \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    echo '=== Camelyon17 pilot start \$(date) train_frac=${TRAIN_FRAC} ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    MARKER=${ROOT}/data/raw/wilds/camelyon17_v1.0/RELEASE_v1.0.txt
    while [ ! -f \"\${MARKER}\" ]; do
      echo \"waiting for download (\${MARKER})...\"
      sleep 120
    done
    ${PY} scripts/camelyon17_pilot.py --stage all --backbone all --seed 42 \
      --batch-size 64 --num-workers 4 --train-frac ${TRAIN_FRAC}
    echo '=== Camelyon17 pilot done \$(date) ==='
  " >> "${LOG}" 2>&1 &

echo "Submitted Camelyon17 download + pilot; logs: ${DL_LOG}, ${LOG}"
sleep 3
squeue -u "$(whoami)" | tee -a "${LOG}"
