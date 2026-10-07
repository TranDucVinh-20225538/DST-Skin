#!/bin/bash
#SBATCH --job-name=camp-slid
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/camp_slid_%A_%a.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
M=(resnet18 resnet50 densenet121 convnext_tiny mobilenet_v3_large regnet_y_3_2gf effb3 efficientnet_v2_s dinov2_vitb14 uni conch_v1_5 virchow2 dinov2_vitl14)
echo "HEAD $(git rev-parse HEAD) model ${M[$SLURM_ARRAY_TASK_ID]}"
${DST_PY:-python} scripts/rigor/camp_slide_identity.py probe --model "${M[$SLURM_ARRAY_TASK_ID]}"
