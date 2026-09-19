#!/usr/bin/env bash
# MIDOG zoo: 5 missing CNN backbones (DenseNet, ConvNeXt, MobileNet, RegNet, EffV2-S@224).
# Does not retrain R18 / in-house R50 / EffB3. Does not touch OpenMIBOOD public R50.
# Then re-analyze the original 3 with 7 methods so Kendall W is defined on the same scores.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/midog_newcnn.log"
MARKER="${ROOT}/data/raw/midog/RELEASE_id_csid.txt"
mkdir -p "${ROOT}/logs"
if [ ! -f "${MARKER}" ]; then
  echo "MIDOG crops not ready: ${MARKER}" >&2
  exit 1
fi
echo "=== MIDOG NEW-CNN zoo submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup srun --gres=gpu:1 -c 8 --mem=48G -t 12:00:00 \
  --job-name=dst-midog-zoo \
  bash -lc "
    set -e
    cd ${ROOT}
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1
    echo '=== MIDOG NEW-CNN start \$(date) ==='
    ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
    ${PY} scripts/midog_pilot.py --stage all --backbone new --seed 42 --num-workers 4
    echo '=== Re-analyze original 3 with 7 methods \$(date) ==='
    ${PY} scripts/midog_pilot.py --stage analyze --backbone resnet18,resnet50,effb3 --seed 42 --rebuild
    echo '=== Phao B coverage n=8 \$(date) ==='
    ${PY} scripts/pathology_risk_coverage.py --domain midog
    echo '=== MIDOG NEW-CNN done \$(date) ==='
  " >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
