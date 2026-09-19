#!/usr/bin/env bash
# Matched-recipe: M1 four jumpers → ResNet Adam; M2 R18/R50 → ConvNeXt AdamW.
# Does not overwrite frac1/seed42. Does not scancel seed-variance.
# Precommit: decision_precommit_matched_recipe.md
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
mkdir -p "${ROOT}/logs"
echo "=== matched-recipe submit $(date) ==="
squeue -u "$(whoami)"

submit_m1 () {
  local bb="$1"
  local extra="$2"
  local log="${ROOT}/logs/camelyon_matched_adam_${bb}.log"
  echo "=== M1 ${bb} submit $(date) ===" | tee -a "${log}"
  nohup srun --gres=gpu:1 -c 8 --mem=48G -t 24:00:00 \
    --job-name="dst-madam-${bb}" \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      export PYTHONUNBUFFERED=1
      echo '=== M1 ${bb} start' \$(date)
      ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
      ${PY} scripts/camelyon17_pilot.py --stage all --backbone ${bb} --seed 42 \
        --train-frac 1.0 --num-workers 4 --log-msp-epoch \
        --artifact-tag matched_adam --optim adam --lr 1e-4 --wd 1e-4 \
        --recipe-batch-size 64 ${extra}
      echo '=== M1 ${bb} done' \$(date)
    " >> "${log}" 2>&1 &
  sleep 3
}

submit_m2 () {
  local bb="$1"
  local log="${ROOT}/logs/camelyon_matched_adamw_${bb}.log"
  echo "=== M2 ${bb} submit $(date) ===" | tee -a "${log}"
  nohup srun --gres=gpu:1 -c 8 --mem=48G -t 24:00:00 \
    --job-name="dst-madamw-${bb}" \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      export PYTHONUNBUFFERED=1
      echo '=== M2 ${bb} start' \$(date)
      ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
      ${PY} scripts/camelyon17_pilot.py --stage all --backbone ${bb} --seed 42 \
        --train-frac 1.0 --num-workers 4 --log-msp-epoch \
        --artifact-tag matched_adamw --optim adamw --lr 1e-4 --wd 0.05 \
        --recipe-batch-size 64
      echo '=== M2 ${bb} done' \$(date)
    " >> "${log}" 2>&1 &
  sleep 3
}

submit_m1 convnext_tiny ""
submit_m1 mobilenet_v3_large ""
submit_m1 regnet_y_3_2gf ""
submit_m1 efficientnet_v2_s "--input-size 224"
submit_m2 resnet18
submit_m2 resnet50
sleep 4
squeue -u "$(whoami)"
echo "logs: ${ROOT}/logs/camelyon_matched_*.log"
echo "precommit: ${ROOT}/decision_precommit_matched_recipe.md"
echo "After CSVs: ${PY} ${ROOT}/scripts/camelyon_matched_recipe_read.py"
