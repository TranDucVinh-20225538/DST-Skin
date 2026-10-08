#!/bin/bash
#SBATCH --job-name=r3-5b-cnn
#SBATCH --exclude=node002
#SBATCH --output=logs/r3_5b_cnn_%A_%a.out
# R3 item 5b, post hoc: K = 2 / 5 / 10 drift for the Camelyon CNN cells (seed 42).
# MODE=maha : mahalanobis_l2 on CPU, one CNN per array task.
# MODE=knn  : GPU knn_mean_cosine. First an equality check against the CPU Step B result (conch_v1_5, K = 2:
#             X_K and leaky equal to 1e-6); on failure no GPU number is produced. Then all CNNs.
set -euo pipefail
R=$HOME/DST-Skin
PY=$HOME/.venvs/crossfit-r3/bin/python
export PYTHONPATH=$R OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export R3_INPUTS=$HOME/r3work/item1/inputs
cd $R
CNNS=(resnet50 convnext_tiny densenet121 effb3 efficientnet_v2_s mobilenet_v3_large regnet_y_3_2gf resnet18)
if [ "$MODE" = maha ]; then
  $PY scripts/r3/item5b_stepB.py --fm ${CNNS[$SLURM_ARRAY_TASK_ID]} --scorer mahalanobis_l2
else
  $PY scripts/r3/item5b_stepB.py --fm conch_v1_5 --scorer knn_mean_cosine --ks 2 --gpu-knn --tag _gpucheck
  $PY - <<'EOF'
import json, sys
c = json.load(open("results/r3/5b/stepB/conch_v1_5_knn_mean_cosine.json"))["K2"]
g = json.load(open("results/r3/5b/stepB/conch_v1_5_knn_mean_cosine_gpucheck.json"))["K2"]
d = max(abs(c["X_K"] - g["X_K"]), abs(c["auroc_leaky"] - g["auroc_leaky"]))
print("GPU check max abs diff", d)
sys.exit(0 if d <= 1e-6 else 1)
EOF
  for m in "${CNNS[@]}"; do $PY scripts/r3/item5b_stepB.py --fm $m --scorer knn_mean_cosine --gpu-knn; done
fi
