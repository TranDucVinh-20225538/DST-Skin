#!/bin/bash
#SBATCH --job-name=r3-1-cam-v2
#SBATCH --exclude=node002
#SBATCH --nice=5000
#SBATCH --output=logs/r3_1_cam_v2_%A_%a.out
# R3 item 1 v2, Camelyon cells (list_camelyon.txt), Track A scorers, OUTROOT=out_v2.
# MODE=maha : mahalanobis_l2 on CPU (sklearn), suffix _maha.
# MODE=knnmc: knn_mean_cosine with the GPU scorer (passed the strict equality check), suffix _knnmc.
set -euo pipefail
W=$HOME/r3work/item1
common=(LIST=$W/list_camelyon.txt WORK=$W OUTROOT=out_v2 NJOBS=1 BLAS=$SLURM_CPUS_PER_TASK)
if [ "$MODE" = maha ]; then
  env "${common[@]}" SCORERS=mahalanobis_l2 OUTSUF=_maha bash $HOME/DST-Skin/scripts/r3/run_item1.sh
else
  nvidia-smi --query-gpu=name --format=csv,noheader
  env "${common[@]}" SCORERS=knn_mean_cosine OUTSUF=_knnmc SCRIPT=$HOME/DST-Skin/scripts/r3/recompute_gpu_knn.py \
    bash $HOME/DST-Skin/scripts/r3/run_item1.sh
fi
