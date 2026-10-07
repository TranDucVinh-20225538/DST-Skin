#!/bin/bash
#SBATCH --job-name=fm-gate-x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/fm_gate_x_%A_%a.out
# DST_PY=... FMX_CELLS="ds:fm[:limit] | cnnidx:arch:seed ..." sbatch --array=0-(n-1)%3 run_fm_gate_extract.sh
# ds = camelyon | dermamnist | isic2019 | breakhis; cnnidx = existing extract_features_indexed.py (Camelyon CNN).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1
PY=${DST_PY:-python}
echo "HEAD $(git rev-parse HEAD)"
read -ra CELLS <<< "${FMX_CELLS}"
IFS=: read -r DS FM LIMIT <<< "${CELLS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "cell ${DS} ${FM} ${LIMIT} on $(hostname)"
EXTRA=()
[[ -n "${LIMIT}" && ${DS} != cnnidx ]] && EXTRA=(--limit "${LIMIT}")
if [[ ${DS} == camelyon || ${DS} == cnnidx ]]; then
  export DST_STAGE_DS=camelyon17_v1.0 DST_STAGE_DATA=1
  source scripts/rigor/stage_wilds.sh
  if [[ ${DS} == cnnidx ]]; then
    stage_run ${PY} scripts/rigor/extract_features_indexed.py --archs "${FM}" --seeds "${LIMIT}" --task-id 0 --verify --num-workers 8
  else
    stage_run ${PY} scripts/rigor/fm_gate_extract.py --ds "${DS}" --fm "${FM}" "${EXTRA[@]}"
  fi
else
  T=/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
  trap 'rm -rf "${T}"' EXIT
  mkdir -p "${T}/img"
  cp "data/staged/${DS}_256.tar" "${T}/"
  tar -xf "${T}/${DS}_256.tar" -C "${T}/img"
  rm -f "${T}/${DS}_256.tar"
  export DST_IMG_DIR="${T}/img"
  ${PY} scripts/rigor/fm_gate_extract.py --ds "${DS}" --fm "${FM}" "${EXTRA[@]}"
fi
