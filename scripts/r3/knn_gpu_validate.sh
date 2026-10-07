#!/bin/bash
# R3 item 1: GPU-kNN equality checks against the sklearn CPU scorer (run on one GPU).
#  1. score-level: k-th-NN distance, sklearn vs GPU, on two medical cells and on Camelyon ResNet-50 (fold-size fit set)
#  2. full recompute_paper_ci (knn) with the GPU scorer on two medical cells (CPU outputs exist from the main run)
#  3. Camelyon ResNet-50 s42 reduced (3 jackknife blocks) and full, GPU scorer
set -euo pipefail
W=$HOME/r3work/item1
PY=$HOME/.venvs/crossfit-r3/bin/python
G=$HOME/DST-Skin/scripts/r3/recompute_gpu_knn.py
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name --format=csv,noheader
for c in isic2019_resnet50_s42_std kermany_convnext_tiny_s42_std; do $PY $G --check $W/inputs/$c.npz; done
$PY $G --check $W/inputs/camelyon_resnet50_s42.npz --n-fit 150000
run() {  # cell outsuf extra
  printf '%s\n' "$1" > $W/list_tmp_$2.txt
  LIST=$W/list_tmp_$2.txt WORK=$W NJOBS=1 BLAS=$SLURM_CPUS_PER_TASK SCORERS=knn OUTSUF=$2 EXTRA="$3" SCRIPT=$G \
    SLURM_ARRAY_TASK_ID=0 bash $HOME/DST-Skin/scripts/r3/run_item1.sh
}
run isic2019_resnet50_s42_std _knn_gpu ""
run kermany_convnext_tiny_s42_std _knn_gpu ""
run camelyon_resnet50_s42 _knnval_gpu "--n-jackknife-blocks 3"
run camelyon_resnet50_s42 _knn_gpu_full ""
