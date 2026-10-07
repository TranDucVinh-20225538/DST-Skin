#!/bin/bash
#SBATCH --job-name=fm-gate-s
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/fm_gate_s_%A_%a.out
# DST_PY=... FMS_CELLS="fm|cnn:model:seed[:smoke] ..." sbatch --array=0-(n-1) run_fm_gate_score.sh
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=${DST_PY:-python}
echo "HEAD $(git rev-parse HEAD)"
read -ra CELLS <<< "${FMS_CELLS}"
IFS=: read -r KIND MODEL SEED SMOKE <<< "${CELLS[${SLURM_ARRAY_TASK_ID:-0}]}"
EXTRA=()
[[ -n "${SMOKE}" ]] && EXTRA=(--smoke)
echo "cell ${KIND} ${MODEL} seed ${SEED} ${SMOKE} on $(hostname)"
${PY} scripts/rigor/fm_gate_score.py --${KIND} "${MODEL}" --seed "${SEED}" "${EXTRA[@]}"
