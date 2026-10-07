#!/bin/bash
# R3 item 1: GPU knn_mean_cosine (Track A kNN) equality checks against the sklearn CPU scorer.
# MODE=gpu: score-level check (ISIC ResNet-50; Camelyon ResNet-50 with a fold-size fit set), then recompute_paper_ci
#           with the GPU scorer on ISIC ResNet-50 (full) and Camelyon ResNet-50 s42 (3 jackknife blocks).
# MODE=cpu: the same two recompute_paper_ci runs with the sklearn scorer.
# Pass: score |diff| <= 1e-9; every Delta / CI field equal to the 6th decimal (knn_gpu_compare.py).
set -euo pipefail
W=$HOME/r3work/item1
PY=$HOME/.venvs/crossfit-r3/bin/python
G=$HOME/DST-Skin/scripts/r3/recompute_gpu_knn.py
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
run() {  # cell outsuf extra script
  printf '%s\n' "$1" > $W/list_tmp_$2.txt
  LIST=$W/list_tmp_$2.txt WORK=$W OUTROOT=out_v2_gpucheck NJOBS=1 BLAS=$SLURM_CPUS_PER_TASK SCORERS=knn_mean_cosine \
    OUTSUF=$2 EXTRA="$3" SCRIPT=$4 SLURM_ARRAY_TASK_ID=0 bash $HOME/DST-Skin/scripts/r3/run_item1.sh
}
if [ "$MODE" = gpu ]; then
  nvidia-smi --query-gpu=name --format=csv,noheader
  $PY $G --check $W/inputs/isic2019_resnet50_s42_std.npz
  $PY $G --check $W/inputs/camelyon_resnet50_s42.npz --n-fit 150000
  run isic2019_resnet50_s42_std _gpu "" $G
  run camelyon_resnet50_s42 _val3_gpu "--n-jackknife-blocks 3" $G
else
  run isic2019_resnet50_s42_std _cpu "" scripts/recompute_paper_ci.py
  run camelyon_resnet50_s42 _val3_cpu "--n-jackknife-blocks 3" scripts/recompute_paper_ci.py
fi
