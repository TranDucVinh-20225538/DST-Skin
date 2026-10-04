#!/bin/bash
# Numerical-stability addendum (decisions/addendum_numerical_stability_2026-10-05.md).
# 1. scores : sbatch array over `numerical_stability.py list` (CPU, 8 threads max per task)
# 2. aggregate + variants (seconds), then the unchanged rigor stages on each variant tree
#    (array over old_r0..old_r9, stable), then verdicts.
# Usage: DST_PY=/path/to/python bash run_stability_variants.sh {aggregate|stages|verdicts}
#        stages runs one variant per SLURM_ARRAY_TASK_ID (0..12); without SLURM it loops all.
#        DST_STAB_VARIANTS=a,b limits `aggregate` to (re)building those variant trees.
set -eo pipefail
cd "$(dirname "$0")"
PY=${DST_PY:-python}
export PYTHONPATH="$(pwd)" PYTHONUNBUFFERED=1
R=scripts/rigor
VARIANTS=(old_r0 old_r1 old_r2 old_r3 old_r4 old_r5 old_r6 old_r7 old_r8 old_r9 stable stable_maha_vim8 stable_maha_novim)

# exit 2 = a REPRO line differs after all outputs were written; in a variant tree the anchor
# Mahalanobis/ViM values are swapped on purpose, so W REPRO lines are expected to differ.
st() {
  local rc=0
  "$@" || rc=$?
  if [[ ${rc} -eq 2 ]]; then echo "(exit 2: REPRO differs in this variant - expected, outputs written)"
  elif [[ ${rc} -ne 0 ]]; then exit ${rc}; fi
}

run_variant() {
  local V="outputs/rigor_pack/stability/variants/$1"
  [[ -d ${V} ]] || { echo "missing variant tree ${V}; run '$0 aggregate' first"; exit 2; }
  echo "== variant $1 $(date)"
  if [[ $1 == stable_maha_novim ]]; then export DST_EXCLUDE_METHODS=vim; else unset DST_EXCLUDE_METHODS; fi
  st ${PY} ${R}/w_from_csv.py --root "${V}" --n-perm 10000
  st ${PY} ${R}/transfer_regret.py --root "${V}"
  st ${PY} ${R}/w_ties.py --root "${V}"
  st ${PY} ${R}/persample.py --root "${V}" --domain camelyon17 --B 1000
  st ${PY} ${R}/coverage_splits.py --root "${V}"
  st ${PY} ${R}/multiple_testing.py --reports "${V}/outputs/reports/rigor_pack"
  echo "== done $1 $(date)"
}

case "${1:-}" in
  aggregate)
    ${PY} ${R}/numerical_stability.py aggregate
    ${PY} ${R}/numerical_stability.py variants ;;
  stages)
    if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then run_variant "${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
    else for v in "${VARIANTS[@]}"; do run_variant "$v"; done; fi ;;
  verdicts)
    ${PY} ${R}/numerical_stability.py verdicts ;;
  *) echo "usage: $0 {aggregate|stages|verdicts}"; exit 1 ;;
esac
