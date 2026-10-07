#!/bin/bash
#SBATCH --job-name=med-sens
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/medbench_sens_%A_%a.out
# SENS="ds:arch:seed ..." sbatch --array=0-(k-1) run_medbench_sens.sh   (task i runs items i, i+k, ...)
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=${DST_PY:-python}
read -ra R <<< "${SENS}"
K=${SLURM_ARRAY_TASK_COUNT:-1}
for ((i=${SLURM_ARRAY_TASK_ID:-0}; i<${#R[@]}; i+=K)); do
    IFS=: read -r DS ARCH SEED <<< "${R[$i]}"
    [ -s "outputs/reports/rigor_pack/medbench/${DS}/sens_${ARCH}_s${SEED}_std.json" ] && continue
    if ${PY} scripts/rigor/medbench_sensitivity.py --ds "${DS}" --arch "${ARCH}" --seed "${SEED}" > /dev/null; then
        echo "${DS} ${ARCH} s${SEED} ok"
    else
        echo "FAIL ${DS} ${ARCH} s${SEED}"
    fi
done
