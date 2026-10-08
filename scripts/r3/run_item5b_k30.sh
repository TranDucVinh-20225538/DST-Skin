#!/bin/bash
#SBATCH --job-name=r3-5b-k30
#SBATCH --exclude=node002
#SBATCH --output=logs/r3_5b_k30_%A_%a.out
# R3 item 5b, post hoc: leave-one-slide-out (K = 30, one slide per fold), same estimator as Step B.
# MODE=maha: CPU, one cell per array task (5 FMs then 8 CNNs). MODE=knn: GPU knn_mean_cosine, all cells
# (submit with --dependency=afterok on the GPU equality-check job).
set -euo pipefail
R=$HOME/DST-Skin
PY=$HOME/.venvs/crossfit-r3/bin/python
export PYTHONPATH=$R OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export R3_INPUTS=$HOME/r3work/item1/inputs
cd $R
CELLS=(uni virchow2 dinov2_vitb14 dinov2_vitl14 conch_v1_5 resnet50 convnext_tiny densenet121 effb3 efficientnet_v2_s mobilenet_v3_large regnet_y_3_2gf resnet18)
if [ "$MODE" = maha ]; then
  $PY scripts/r3/item5b_stepB.py --fm ${CELLS[$SLURM_ARRAY_TASK_ID]} --scorer mahalanobis_l2 --ks 30 --tag _k30
else
  for m in "${CELLS[@]}"; do
    [ -s results/r3/5b/stepB/${m}_knn_mean_cosine_k30.json ] || \
      $PY scripts/r3/item5b_stepB.py --fm $m --scorer knn_mean_cosine --ks 30 --gpu-knn --tag _k30
  done
fi
