#!/bin/bash
#SBATCH --job-name=camp-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/camp_cpu_%A_%a.out
# DST_PY=... CC_CELLS="ds:model1,model2:arm1,arm2 ..." sbatch --array=0-(n-1) run_camp_cpu.sh
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=${DST_PY:-python}
echo "HEAD $(git rev-parse HEAD)"
read -ra CELLS <<< "${CC_CELLS}"
IFS=: read -r DS MODELS ARMS <<< "${CELLS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "cell ${DS} ${MODELS} ${ARMS} on $(hostname)"
${PY} scripts/rigor/camp_medbench_cpu.py --ds "${DS}" --models ${MODELS//,/ } --arms ${ARMS//,/ }
