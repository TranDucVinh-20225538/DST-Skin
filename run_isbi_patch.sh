#!/bin/bash
# ISBI patch (decisions/precommit_isbi_patch_2026-10-05.md).
# Usage: DST_PY=/path/to/python sbatch run_isbi_patch.sh STAGE
#   leak_smoke : section A on ResNet18 only (writes leakfree_vim_react_smoke.*)
#   leak       : section A, all 8 archs
#   disjoint   : section D (CPU)
#   retrain    : section B, one SLURM array task per (arch, fold); submit with
#                --gres=gpu:1 --mem=64G --array=0-5 and DST_STAGE_DATA=1
#SBATCH --job-name=isbi-patch
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/isbi_patch_%x_%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
PY=${DST_PY:-python}
export PYTHONPATH="$(pwd)${DST_PYDEPS:+:${DST_PYDEPS}}" PYTHONUNBUFFERED=1
T=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${T} OPENBLAS_NUM_THREADS=${T} MKL_NUM_THREADS=${T}
R=scripts/rigor

case "${1:-}" in
  leak_smoke) ${PY} ${R}/leakfree_fit_scores.py --archs resnet18 --out-name leakfree_vim_react_smoke ;;
  leak) ${PY} ${R}/leakfree_fit_scores.py ;;
  disjoint)
    for tree in stable_maha_vim8 stable; do ${PY} ${R}/sample_size_disjoint.py --tree ${tree}; done ;;
  retrain)
    ARCHS=(resnet50 convnext_tiny densenet121)
    A=${ARCHS[$((SLURM_ARRAY_TASK_ID / 2))]}; F=$((SLURM_ARRAY_TASK_ID % 2))
    echo "retrain ${A} fold ${F} $(date)"
    if [[ "${DST_STAGE_DATA:-0}" == 1 ]]; then source ${R}/stage_wilds.sh; RUNW=stage_run; else RUNW=""; fi
    ${RUNW} ${PY} ${R}/logit_retrain_slide_disjoint.py --arch ${A} --fold ${F} --num-workers 8 ;;
  retrain2)
    # patch 2 (precommit_isbi_patch2_2026-10-06.md): index = seed_idx*6 + arch_idx*2 + fold, 0-17
    ARCHS=(resnet50 convnext_tiny densenet121); SEEDS=(42 43 44)
    I=${SLURM_ARRAY_TASK_ID}
    S=${SEEDS[$((I / 6))]}; A=${ARCHS[$(((I % 6) / 2))]}; F=$((I % 2))
    echo "retrain2 ${A} seed ${S} fold ${F} $(date)"
    if [[ "${DST_STAGE_DATA:-0}" == 1 ]]; then source ${R}/stage_wilds.sh; RUNW=stage_run; else RUNW=""; fi
    ${RUNW} ${PY} ${R}/logit_retrain_slide_disjoint_v2.py --arch ${A} --seed ${S} --fold ${F} --num-workers 8 ;;
  score2)
    # CPU, same index mapping as retrain2
    ARCHS=(resnet50 convnext_tiny densenet121); SEEDS=(42 43 44)
    I=${SLURM_ARRAY_TASK_ID}
    ${PY} ${R}/isbi_patch2_scores.py --arch ${ARCHS[$(((I % 6) / 2))]} --seed ${SEEDS[$((I / 6))]} --fold $((I % 2)) ;;
  *) echo "usage: $0 {leak_smoke|leak|disjoint|retrain|retrain2|score2}"; exit 2 ;;
esac
