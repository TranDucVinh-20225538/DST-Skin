#!/usr/bin/env bash
# 4-anchor seed-variance: seeds 43–46. Does not touch seed 42.
# One GPU job per backbone. Precommit: decision_precommit_seed_variance.md
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
mkdir -p "${ROOT}/logs"
echo "=== Camelyon seed-variance submit $(date) ==="
squeue -u "$(whoami)"

submit_one () {
  local bb="$1"
  local log="${ROOT}/logs/camelyon_seed_${bb}.log"
  echo "=== ${bb} seeds 43-46 submit $(date) ===" | tee -a "${log}"
  nohup srun --gres=gpu:1 -c 8 --mem=48G -t 24:00:00 \
    --job-name="dst-seed-${bb}" \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      export PYTHONUNBUFFERED=1
      echo '=== ${bb} seed-variance start' \$(date)
      ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
      for s in 43 44 45 46; do
        echo '--- ${bb} seed' \$s \$(date)
        ${PY} scripts/camelyon17_pilot.py --stage all --backbone ${bb} --seed \$s \
          --train-frac 1.0 --num-workers 4 --log-msp-epoch
      done
      echo '=== ${bb} seed-variance done' \$(date)
    " >> "${log}" 2>&1 &
  sleep 3
}

submit_one resnet18
submit_one resnet50
submit_one densenet121
submit_one effb3
sleep 4
squeue -u "$(whoami)"
echo "logs: ${ROOT}/logs/camelyon_seed_*.log"
echo "precommit: ${ROOT}/decision_precommit_seed_variance.md"
