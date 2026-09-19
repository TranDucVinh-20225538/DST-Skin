#!/usr/bin/env bash
# Camelyon17 full train: train_frac=1.0, multiple seeds, does not clobber 5% pilot.
set -euo pipefail

ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/camelyon17_full.log"
SEEDS="${SEEDS:-42 43 44}"
TRAIN_FRAC="${TRAIN_FRAC:-1.0}"
MARKER="${ROOT}/data/raw/wilds/camelyon17_v1.0/RELEASE_v1.0.txt"

mkdir -p "${ROOT}/logs"

if [ ! -f "${MARKER}" ]; then
  echo "Camelyon17 data not ready: ${MARKER}" >&2
  exit 1
fi

echo "=== Camelyon17 FULL submit $(date) seeds=${SEEDS} train_frac=${TRAIN_FRAC} ===" | tee -a "${LOG}"
squeue -u "$(whoami)" -o "%.10i %.12P %.20j %.8T %.10M %.6D %R" | tee -a "${LOG}"

nohup srun --gres=gpu:1 -c 8 --mem=48G -t 48:00:00 \
  --job-name=dst-camelyon-full \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    echo '=== Camelyon17 FULL start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    for seed in ${SEEDS}; do
      echo \"=== FULL seed=\${seed} train_frac=${TRAIN_FRAC} \$(date) ===\"
      ${PY} scripts/camelyon17_pilot.py --stage all --backbone all --seed \${seed} \
        --batch-size 64 --num-workers 4 --train-frac ${TRAIN_FRAC}
      echo \"=== FULL seed=\${seed} done \$(date) ===\"
    done
    echo '=== Camelyon17 FULL all seeds done \$(date) ==='
  " >> "${LOG}" 2>&1 &

echo "Submitted dst-camelyon-full; log: ${LOG}"
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
