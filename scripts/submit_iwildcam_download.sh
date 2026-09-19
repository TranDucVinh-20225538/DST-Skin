#!/usr/bin/env bash
# #5: download WILDS iWildCam (~12GB). CPU only. Do not take a GPU from Job B.
# After metadata.csv exists, queue the GPU train from the login node (not nested srun).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/iwildcam_download.log"
mkdir -p "${ROOT}/logs"
echo "=== iWildCam download submit $(date) ===" | tee -a "${LOG}"
squeue -u "$(whoami)" | tee -a "${LOG}"
nohup bash -lc "
  set -e
  cd ${ROOT}
  echo '=== iWildCam download start' \$(date)
  srun -c 4 --mem=16G -t 12:00:00 --job-name=dst-iwild-dl \
    bash -lc '
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      export PYTHONUNBUFFERED=1
      ${PY} scripts/iwildcam_pilot.py --stage download
    '
  echo '=== iWildCam download done' \$(date)
  echo '=== queue iWildCam GPU train'
  bash ${ROOT}/scripts/submit_iwildcam_pilot.sh
" >> "${LOG}" 2>&1 &
sleep 4
squeue -u "$(whoami)" | tee -a "${LOG}"
echo "log: ${LOG}"
