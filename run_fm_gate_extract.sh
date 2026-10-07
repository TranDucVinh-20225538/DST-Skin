#!/bin/bash
#SBATCH --job-name=fm-gate-x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/fm_gate_x_%A_%a.out
# DST_PY=... FMX_CELLS="ds:fm[:limit] ..." sbatch --array=0-(n-1)%k run_fm_gate_extract.sh
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1
PY=${DST_PY:-python}
echo "HEAD $(git rev-parse HEAD)"
read -ra CELLS <<< "${FMX_CELLS}"
IFS=: read -r DS FM LIMIT <<< "${CELLS[${SLURM_ARRAY_TASK_ID:-0}]}"
EXTRA=()
[[ -n "${LIMIT}" ]] && EXTRA=(--limit "${LIMIT}")
if [[ ${DS} == camelyon ]]; then
  export DST_STAGE_DS=camelyon17_v1.0 DST_STAGE_DATA=1
  source scripts/rigor/stage_wilds.sh
  stage_run ${PY} scripts/rigor/fm_gate_extract.py --ds "${DS}" --fm "${FM}" "${EXTRA[@]}"
else
  ${PY} scripts/rigor/fm_gate_extract.py --ds "${DS}" --fm "${FM}" "${EXTRA[@]}"
fi
