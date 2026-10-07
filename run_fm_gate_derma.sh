#!/bin/bash
#SBATCH --job-name=fm-gate-d
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/fm_gate_d_%A_%a.out
# DST_PY=... FMD_CELLS="ds:fm ..." sbatch --array=0-(n-1) run_fm_gate_derma.sh   (ds = dermamnist | isic2019 | breakhis)
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=${DST_PY:-python}
echo "HEAD $(git rev-parse HEAD)"
read -ra CELLS <<< "${FMD_CELLS}"
IFS=: read -r DS FM <<< "${CELLS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "cell ${DS} ${FM} on $(hostname)"
${PY} scripts/rigor/fm_gate_derma.py --ds "${DS}" --fm "${FM}"
R=outputs/reports/rigor_pack/medbench/${DS}
O=outputs/reports/rigor_pack/foundation_gate/${DS}
mkdir -p "${O}"
ARMS=$(${PY} -c "import sys; sys.path.insert(0,'scripts/rigor'); from fm_gate_extract import MED_ARMS; print(' '.join(MED_ARMS['${DS}']))")
for ARM in ${ARMS}; do
  ${PY} scripts/rigor/medbench_scores.py --ds "${DS}" --arch "fm_${FM}" --seed 42 --arm "${ARM}"
  mv "${R}/scores_fm_${FM}_s42_${ARM}.json" "${O}/"
done
STD=$(${PY} -c "import sys; sys.path.insert(0,'scripts/rigor'); from fm_gate_extract import MED_ARMS; print(' '.join(a for a in MED_ARMS['${DS}'] if a.startswith('std')))")
${PY} scripts/rigor/camp_medbench_cpu.py --ds "${DS}" --models "fm_${FM}" --arms ${STD}
