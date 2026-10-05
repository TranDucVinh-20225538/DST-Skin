#!/bin/bash
# Sample-size / ranking-stability analysis (decisions/precommit_sample_size_2026-10-05.md). CPU only.
# Usage: DST_PY=/path/to/python sbatch run_sample_size.sh {scores|features}
#   scores   : U matrices, axes, variance components for the primary and appendix trees
#   features : covariance spectrum of the train features (step 3)
#SBATCH --job-name=rigor-samplesize
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/rigor_samplesize_%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
PY=${DST_PY:-python}
# statsmodels (step 2) is not in the base env: DST_PYDEPS=dir installed with pip --target
export PYTHONPATH="$(pwd)${DST_PYDEPS:+:${DST_PYDEPS}}" PYTHONUNBUFFERED=1
T=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${T} OPENBLAS_NUM_THREADS=${T} MKL_NUM_THREADS=${T}
S=scripts/rigor/sample_size.py

varcomp() {
  for tree in stable_maha_vim8 stable; do
    ${PY} ${S} varcomp --tree ${tree}
  done
}

case "${1:-}" in
  scores)
    for tree in stable_maha_vim8 stable; do
      ${PY} ${S} umat --tree ${tree}
      ${PY} ${S} axes --tree ${tree}
    done
    varcomp ;;
  varcomp) varcomp ;;
  features) ${PY} ${S} features ;;
  *) echo "usage: $0 {scores|varcomp|features}"; exit 2 ;;
esac
