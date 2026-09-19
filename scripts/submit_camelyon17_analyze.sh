#!/usr/bin/env bash
# Camelyon17 full-scale ANALYZE only (seed42). Does not retrain, does not
# touch CIFAR-10-C or MIDOG. CPU job — GPU left for 59485.
set -euo pipefail

ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/camelyon17_analyze.log"
FEAT="${ROOT}/outputs/features/camelyon17/frac1/seed42/resnet50_features.pt"

mkdir -p "${ROOT}/logs"

if [ ! -f "${FEAT}" ]; then
  echo "Missing extracted features: ${FEAT}" >&2
  exit 1
fi

echo "=== Camelyon17 ANALYZE submit $(date) seed=42 train_frac=1.0 ===" | tee -a "${LOG}"
squeue -u "$(whoami)" -o "%.10i %.12P %.20j %.8T %.10M %R" | tee -a "${LOG}"

nohup srun --gres=gpu:0 -c 8 --mem=48G -t 12:00:00 \
  --job-name=dst-camel-analyze \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    echo '=== Camelyon17 ANALYZE start \$(date) ==='
    ${PY} scripts/camelyon17_pilot.py --stage analyze --backbone all --seed 42 \
      --train-frac 1.0
    echo '=== Camelyon17 ANALYZE done \$(date) ==='
  " >> "${LOG}" 2>&1 &

echo "Submitted dst-camel-analyze; log: ${LOG}"
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
