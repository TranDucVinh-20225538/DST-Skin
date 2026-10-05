#!/bin/bash
# Group-leakage multibench (decisions/precommit_group_leakage_multibench_2026-10-06.md).
# Usage: DST_PY=... MB_DS=iwildcam|rxrx1 sbatch [--gres=gpu:1] --array=... run_multibench.sh STAGE
#   extract : GPU, indexed features of the standard seed-42 model, array index = arch in ARCHS_A
#   base    : GPU, RxRx1 seed-42 base models on the full train split, array 0-3 = ARCHS_B
#   train   : GPU, (B) retrain, index = seed_idx*8 + arch_idx*2 + fold (0-15), seeds 42/43
#   scoreA  : CPU, (A)/(C) per arch (array index = arch in ARCHS_A)
#   scoreB  : CPU, 7 scores per retrained model (same index as train)
# GPU stages stage the dataset to node-local /tmp when DST_STAGE_DATA=1.
#SBATCH --job-name=multibench
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/multibench_%x_%A_%a.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
PY=${DST_PY:-python}
export PYTHONPATH="$(pwd)" PYTHONUNBUFFERED=1
T=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${T} OPENBLAS_NUM_THREADS=${T} MKL_NUM_THREADS=${T}
R=scripts/rigor
DS=${MB_DS:?set MB_DS}
I=${SLURM_ARRAY_TASK_ID:-0}
if [[ ${DS} == iwildcam ]]; then
  ARCHS_A=(resnet18 resnet50 densenet121 convnext_tiny mobilenet_v3_large regnet_y_3_2gf effb3 efficientnet_v2_s)
  export DST_STAGE_DS=iwildcam_v2.0
else
  ARCHS_A=(resnet18 resnet50 densenet121 convnext_tiny)
  export DST_STAGE_DS=rxrx1_v1.0
fi
ARCHS_B=(resnet18 resnet50 densenet121 convnext_tiny); SEEDS=(42 43)
stage() { if [[ "${DST_STAGE_DATA:-0}" == 1 ]]; then source ${R}/stage_wilds.sh; RUNW=stage_run; else RUNW=""; fi; }

case "${1:-}" in
  extract) stage; ${RUNW} ${PY} ${R}/multibench_extract.py --ds ${DS} --arch ${ARCHS_A[$I]} ;;
  base)    stage; ${RUNW} ${PY} ${R}/multibench_train.py --ds ${DS} --arch ${ARCHS_B[$I]} --seed 42 --fold full ;;
  train)   stage; ${RUNW} ${PY} ${R}/multibench_train.py --ds ${DS} --arch ${ARCHS_B[$(((I % 8) / 2))]} \
             --seed ${SEEDS[$((I / 8))]} --fold $((I % 2)) ;;
  scoreA)  ${PY} ${R}/multibench_scores.py --ds ${DS} --arch ${ARCHS_A[$I]} --mode A ;;
  scoreB)  ${PY} ${R}/multibench_scores.py --ds ${DS} --arch ${ARCHS_B[$(((I % 8) / 2))]} --mode B \
             --seed ${SEEDS[$((I / 8))]} --fold $((I % 2)) ;;
  *) echo "usage: MB_DS=... $0 {extract|base|train|scoreA|scoreB}"; exit 2 ;;
esac
