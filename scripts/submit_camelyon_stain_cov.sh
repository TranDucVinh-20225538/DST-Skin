#!/usr/bin/env bash
# Viên 1: HED stain-covariate on R18 / R50 / DenseNet. Does not touch official seed 42.
# Precommit: decision_precommit_stain_cov.md
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
mkdir -p "${ROOT}/logs"
echo "=== stain-cov submit $(date) ==="
squeue -u "$(whoami)"

submit_one () {
  local bb="$1"
  local log="${ROOT}/logs/camelyon_stain_cov_${bb}.log"
  echo "=== stain-cov ${bb} submit $(date) ===" | tee -a "${log}"
  nohup srun --gres=gpu:1 -c 8 --mem=48G -t 24:00:00 \
    --job-name="dst-stain-${bb}" \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      export PYTHONUNBUFFERED=1
      echo '=== stain-cov ${bb} start' \$(date)
      ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
      ${PY} scripts/camelyon17_pilot.py --stage all --backbone ${bb} --seed 42 \
        --train-frac 1.0 --num-workers 4 --log-msp-epoch \
        --artifact-tag stain_cov --stain-cov
      ${PY} scripts/camelyon_stain_cov_eval.py --backbone ${bb} --num-workers 4
      ${PY} scripts/camelyon_stain_cov_read.py || true
      echo '=== stain-cov ${bb} done' \$(date)
    " >> "${log}" 2>&1 &
  sleep 3
}

submit_one resnet18
submit_one resnet50
submit_one densenet121
sleep 4
squeue -u "$(whoami)"
echo "logs: ${ROOT}/logs/camelyon_stain_cov_*.log"
echo "precommit: ${ROOT}/decision_precommit_stain_cov.md"
