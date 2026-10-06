#!/bin/bash
#SBATCH --job-name=med-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/medbench_%x_%A_%a.out
# MB_DS=<dataset> MB_RUNS="arch:seed:arm ..." sbatch --array=0-(n-1)%k run_medbench.sh [extra train args]
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=${DST_PY:-python}
read -ra RUNS <<< "${MB_RUNS}"
IFS=: read -r ARCH SEED ARM <<< "${RUNS[${SLURM_ARRAY_TASK_ID}]}"
SIZE=256; [ "${ARCH}" = "effb3" ] && SIZE=343
T=/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
trap 'rm -rf "${T}"' EXIT
mkdir -p "${T}/img"
t0=$(date +%s)
cp "data/staged/${MB_DS}_${SIZE}.tar" "${T}/"
tar -xf "${T}/${MB_DS}_${SIZE}.tar" -C "${T}/img"
rm -f "${T}/${MB_DS}_${SIZE}.tar"
echo "staged ${MB_DS}_${SIZE} in $(( $(date +%s) - t0 ))s on $(hostname)"
${PY} scripts/rigor/medbench_train.py --ds "${MB_DS}" --arch "${ARCH}" --seed "${SEED}" --arm "${ARM}" --img-dir "${T}/img" "$@"
