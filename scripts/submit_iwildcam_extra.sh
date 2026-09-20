#!/usr/bin/env bash
# iWildCam extra 5 CNNs. Existing R18/R50/DenseNet ckpts are skipped.
# Does not mix into Camelyon/skin Kendall W.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
mkdir -p "${ROOT}/logs"
echo "=== iWildCam extra submit $(date) ==="
squeue -u "$(whoami)"

submit_one () {
  local bb="$1"
  local log="${ROOT}/logs/iwildcam_${bb}.log"
  echo "=== iWildCam ${bb} submit $(date) ===" | tee -a "${log}"
  nohup srun --gres=gpu:1 -c 8 --mem=48G -t 24:00:00 \
    --job-name="dst-iwild-${bb}" \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      export PYTHONUNBUFFERED=1
      echo '=== iWildCam ${bb} start' \$(date)
      ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
      ${PY} scripts/iwildcam_pilot.py --stage all --backbone ${bb} --num-workers 4
      echo '=== iWildCam ${bb} done' \$(date)
    " >> "${log}" 2>&1 &
  sleep 3
}

submit_one convnext_tiny
submit_one mobilenet_v3_large
submit_one regnet_y_3_2gf
submit_one effb3
submit_one efficientnet_v2_s
sleep 4
squeue -u "$(whoami)"
echo "logs: ${ROOT}/logs/iwildcam_*.log"
