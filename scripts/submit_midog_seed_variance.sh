#!/usr/bin/env bash
# MIDOG in-house zoo seeds 43–46. Does not touch seed 42 / official W.
# 1896 train images — cheap. One GPU per seed, all 8 backbones.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
mkdir -p "${ROOT}/logs"
echo "=== MIDOG seed-variance submit $(date) ==="
squeue -u "$(whoami)"

submit_seed () {
  local s="$1"
  local log="${ROOT}/logs/midog_seed${s}.log"
  echo "=== MIDOG seed ${s} submit $(date) ===" | tee -a "${log}"
  nohup srun --gres=gpu:1 -c 8 --mem=32G -t 12:00:00 \
    --job-name="dst-midog-s${s}" \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      export PYTHONUNBUFFERED=1
      echo '=== MIDOG seed ${s} start' \$(date)
      ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
      ${PY} scripts/midog_pilot.py --stage all --backbone all --seed ${s} --num-workers 4
      echo '=== MIDOG seed ${s} done' \$(date)
    " >> "${log}" 2>&1 &
  sleep 3
}

for s in 43 44 45 46; do
  submit_seed "${s}"
done
sleep 4
squeue -u "$(whoami)"
echo "logs: ${ROOT}/logs/midog_seed*.log"
