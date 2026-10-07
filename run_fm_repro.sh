#!/bin/bash
#SBATCH --job-name=fm-repro
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/fm_repro_%j.out
# REPRO for the foundation gate: Camelyon ResNet18 / ResNet50 seed-42 leakfree_knn.py vs the committed CSV
# (1e-6), same configuration as the medbench REPRO that passed (16 CPUs, no BLAS thread variables).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd)
PY=${DST_PY:-python}
echo "HEAD $(git rev-parse HEAD)"
OUT=outputs/rigor_pack/foundation_gate/repro
${PY} scripts/rigor/leakfree_knn.py --archs resnet18 resnet50 --seeds 42 --reports "${OUT}"
${PY} - "${OUT}" <<'PY'
import sys
import pandas as pd
a = pd.read_csv("outputs/reports/rigor_pack/leakage/leakfree_knn.csv")
b = pd.read_csv(sys.argv[1] + "/leakfree_knn.csv")
ok = True
for arch in ("resnet18", "resnet50"):
    x = a[(a.arch == arch) & (a.seed == 42)].reset_index(drop=True)
    y = b[(b.arch == arch) & (b.seed == 42)].reset_index(drop=True)
    cols = [c for c in x.columns if c in y.columns and x[c].dtype.kind == "f"]
    d = (x[cols] - y[cols]).abs().max()
    print(arch, "max abs diff %.3g" % d.max())
    ok &= bool((d <= 1e-6).all())
print("REPRO", "PASS" if ok else "FAIL")
PY
