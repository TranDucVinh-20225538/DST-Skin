#!/usr/bin/env python3
"""R3 item 1: compare kNN rows of recompute_paper_ci outputs, sklearn CPU vs GPU scorer.

Pass = every compared field agrees to the 6th decimal (|diff| < 5e-7). Writes a CSV and prints a table.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

W = Path.home() / "r3work/item1/out"
FIELDS = ("auroc_seen", "auroc_unseen", "delta", "old_bootstrap_lo", "old_bootstrap_hi", "old_bootstrap_se",
          "new_jackknife_lo", "new_jackknife_hi", "new_jackknife_se")
PAIRS = [("isic2019_resnet50_s42_std", "", "_knn_gpu"),
         ("kermany_convnext_tiny_s42_std", "", "_knn_gpu"),
         ("camelyon_resnet50_s42", "_knnval_cpu", "_knnval_gpu"),
         ("camelyon_resnet50_s42", "_knn_cpu_full", "_knn_gpu_full")]
TOL = 5e-7


def knn_row(path):
    for r in csv.DictReader(open(path)):
        if r["scorer"] == "knn":
            return r
    raise SystemExit(f"no knn row in {path}")


def main() -> int:
    rows = []
    for cell, cpu, gpu in PAIRS:
        pc, pg = W / f"{cell}{cpu}/paper_ci.csv", W / f"{cell}{gpu}/paper_ci.csv"
        if not (pc.exists() and pg.exists()):
            print(f"missing: {pc if not pc.exists() else pg}")
            continue
        a, b = knn_row(pc), knn_row(pg)
        diffs = {f: abs(float(a[f]) - float(b[f])) for f in FIELDS}
        f_max = max(diffs, key=diffs.get)
        rows.append(dict(cell=cell, run=gpu.strip("_").replace("knn_gpu", "full").replace("knnval_gpu", "reduced_3_blocks")
                         .replace("full_full", "full"), max_abs_diff=diffs[f_max], field_of_max=f_max,
                         status_same=a["status_thr"] == b["status_thr"] and a["status_gt0"] == b["status_gt0"],
                         passed=all(d < TOL for d in diffs.values())))
    out = W.parent / "knn_gpu_compare.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    return 0 if rows and all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
