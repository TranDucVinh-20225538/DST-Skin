#!/bin/bash
#SBATCH --job-name=fm-gate-d
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/fm_gate_d_%A_%a.out
# DST_PY=... FMD_FMS="fm ..." sbatch --array=0-(n-1) run_fm_gate_derma.sh
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=${DST_PY:-python}
echo "HEAD $(git rev-parse HEAD)"
read -ra FMS <<< "${FMD_FMS}"
FM=${FMS[${SLURM_ARRAY_TASK_ID:-0}]}
${PY} scripts/rigor/fm_gate_derma.py --fm "${FM}"
R=outputs/reports/rigor_pack/medbench/dermamnist
mkdir -p outputs/reports/rigor_pack/foundation_gate/dermamnist
for ARM in std b0 b1; do
  ${PY} scripts/rigor/medbench_scores.py --ds dermamnist --arch "fm_${FM}" --seed 42 --arm "${ARM}"
  mv "${R}/scores_fm_${FM}_s42_${ARM}.json" outputs/reports/rigor_pack/foundation_gate/dermamnist/
done
