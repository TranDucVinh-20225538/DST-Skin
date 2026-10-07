#!/bin/bash
#SBATCH --job-name=fm-repro
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/fm_repro_%j.out
# REPRO for the foundation gate on the float64 Ledoit-Wolf scorer, BLAS pinned to 16 threads.
#   MODE=baseline ARCHS="..."  -> outputs/reports/rigor_pack/leakage_lw64/part_${TAG}/leakfree_knn.csv
#   MODE=repro TAG=a|b         -> outputs/rigor_pack/foundation_gate/repro_lw64_${TAG}/ (ResNet18 / ResNet50 seed 42)
#   MODE=compare               -> REPRO a vs b vs baseline part r18r50, 1e-6 on every float column
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=${DST_PY:-python}
echo "HEAD $(git rev-parse HEAD) MODE=${MODE} TAG=${TAG}"
BASE=outputs/reports/rigor_pack/leakage_lw64
case "${MODE}" in
  baseline)
    ${PY} scripts/rigor/leakfree_knn.py --archs ${ARCHS} --seeds 42 --reports "${BASE}/part_${TAG}" ;;
  repro)
    ${PY} scripts/rigor/leakfree_knn.py --archs resnet18 resnet50 --seeds 42 \
      --reports "outputs/rigor_pack/foundation_gate/repro_lw64_${TAG}" ;;
  compare)
    ${PY} - "${BASE}/part_r18r50/leakfree_knn.csv" outputs/rigor_pack/foundation_gate/repro_lw64_{a,b}/leakfree_knn.csv <<'PY'
import sys
import pandas as pd
base, ra, rb = (pd.read_csv(p) for p in sys.argv[1:4])
ok = True
for name, x, y in (("a vs b", ra, rb), ("a vs baseline", ra, base), ("b vs baseline", rb, base)):
    for arch in ("resnet18", "resnet50"):
        u = x[(x.arch == arch) & (x.seed == 42)].reset_index(drop=True)
        v = y[(y.arch == arch) & (y.seed == 42)].reset_index(drop=True)
        cols = [c for c in u.columns if c in v.columns and u[c].dtype.kind == "f"]
        d = (u[cols] - v[cols]).abs().max()
        print(name, arch, "max abs diff %.3g" % d.max())
        ok &= bool((d <= 1e-6).all())
print("REPRO", "PASS" if ok else "FAIL")
PY
    ;;
  *) echo "unknown MODE=${MODE}"; exit 2 ;;
esac
