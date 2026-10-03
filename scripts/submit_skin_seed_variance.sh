#!/usr/bin/env bash
# Skin ISIC→PAD official zoo seeds 43–46. Writes seed{N}/ — never overwrites seed 42.
# EffV2-S forced @224 inside skin_newcnn zoo loop.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
mkdir -p "${ROOT}/logs"
echo "=== Skin seed-variance submit $(date) ==="
squeue -u "$(whoami)"

submit_seed () {
  local s="$1"
  local log="${ROOT}/logs/skin_seed${s}.log"
  echo "=== Skin seed ${s} submit $(date) ===" | tee -a "${log}"
  nohup srun --gres=gpu:1 -c 8 --mem=48G -t 24:00:00 \
    --job-name="dst-skin-s${s}" \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      export PYTHONUNBUFFERED=1
      echo '=== Skin seed ${s} start' \$(date)
      ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
      ${PY} scripts/skin_newcnn.py --stage all --backbone zoo --seed ${s} --num-workers 4
      echo '=== Skin seed ${s} done' \$(date)
    " >> "${log}" 2>&1 &
  sleep 3
}

for s in 43 44 45 46; do
  submit_seed "${s}"
done
sleep 4
squeue -u "$(whoami)"
echo "logs: ${ROOT}/logs/skin_seed*.log"
