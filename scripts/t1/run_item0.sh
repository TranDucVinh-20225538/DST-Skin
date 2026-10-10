#!/bin/bash
#SBATCH --job-name=t1_item0
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/item0_%j.out
# T1 item 0 (GPU part): env record, microbenchmark, GPU-kNN equality on 2 cells (k = 1 and k = 50).
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=$HOME/.venvs/crossfit-r3/bin/python
S=$HOME/DST-Skin/scripts/t1/item0.py
O=$HOME/DST-Skin/results/t1/0
cd $HOME/DST-Skin
$PY $S env
ls $O/throughput_*.json >/dev/null 2>&1 || $PY $S bench
[ -s $O/knn_equality_isic2019_resnet50_s42_std.json ] || $PY $S knncheck isic2019_resnet50_s42_std 42
[ -s $O/knn_equality_camelyon_resnet50_s42.json ] || $PY $S knncheck camelyon_resnet50_s42 42
echo DONE
