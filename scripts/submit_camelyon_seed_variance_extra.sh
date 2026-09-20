#!/usr/bin/env bash
# Extra seed-variance: ConvNeXt / MobileNet / RegNet / EffV2-S@224, seeds 43–46.
# Does not touch seed 42 or official W. EffV2 is @224 (official cell).
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
mkdir -p "${ROOT}/logs"
echo "=== Camelyon seed-variance extra submit $(date) ==="
squeue -u "$(whoami)"

submit_one () {
  local bb="$1"
  local extra="${2:-}"
  local log="${ROOT}/logs/camelyon_seed_${bb}.log"
  echo "=== ${bb} seeds 43-46 submit $(date) extra='${extra}' ===" | tee -a "${log}"
  nohup srun --gres=gpu:1 -c 8 --mem=48G -t 24:00:00 \
    --job-name="dst-seed-${bb}" \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      export PYTHONUNBUFFERED=1
      echo '=== ${bb} seed-variance extra start' \$(date)
      ${PY} -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\"
      for s in 43 44 45 46; do
        echo '--- ${bb} seed' \$s \$(date)
        ${PY} scripts/camelyon17_pilot.py --stage all --backbone ${bb} --seed \$s \
          --train-frac 1.0 --num-workers 4 --log-msp-epoch ${extra}
      done
      echo '=== ${bb} seed-variance extra done' \$(date)
    " >> "${log}" 2>&1 &
  sleep 3
}

submit_one convnext_tiny
submit_one mobilenet_v3_large
submit_one regnet_y_3_2gf
submit_one efficientnet_v2_s "--input-size 224"
sleep 4
squeue -u "$(whoami)"
echo "logs: ${ROOT}/logs/camelyon_seed_{convnext_tiny,mobilenet_v3_large,regnet_y_3_2gf,efficientnet_v2_s}.log"
echo "after CSVs: ${PY} scripts/camelyon_seed_jump.py"
