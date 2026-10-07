#!/usr/bin/env python3
"""R3 item 5 report: results/r3/5/{REPORT.md, a_camelyon_fm.csv (from item5_a), b_isbi_patch2_cells.csv,
b_isbi_patch2_summary.csv}. Part (b) summary = median over the (arch, seed) cells of each fold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/r3/5"
FIT = ("Mahalanobis", "kNN", "ViM", "ReAct")


def md(df: pd.DataFrame) -> str:
    f = lambda v: ("%.4f" % v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> int:
    parts, stops = ["# R3 item 5: cross-fit closure", "", "commit: <filled by parent>", ""]
    A = pd.read_csv(OUT / "a_camelyon_fm.csv") if (OUT / "a_camelyon_fm.csv").exists() else None
    if A is not None:
        parts += ["## (a) Camelyon17 FMs: leaky vs F2 vs F1 (bar |F2-F1| <= 0.02)", "", md(A), ""]
        a3 = A[A.scorer != "ReAct"]
        ar = A[A.scorer == "ReAct"]
        va = "a: %d/%d FM x scorer cells pass |F2-F1|<=0.02 (Mahalanobis, kNN, ViM); ReAct %d/%d" % (
            a3["pass"].sum(), len(a3), ar["pass"].sum(), len(ar))
    else:
        va = "a: not computed"
        stops.append("STOP (a): a_camelyon_fm.csv missing")
    rows = []
    for p in sorted((OUT / "cells").glob("b_*.json")):
        j = json.loads(p.read_text())
        for m in FIT:
            rows.append({"arch": j["arch"], "seed": j["seed"], "fold": j["fold"], "scorer": m,
                         "auroc_truth": j[m]["truth"], "auroc_leaky": j[m]["leaky"], "auroc_F2": j[m]["F2"],
                         "gap_leaky_minus_truth": j[m]["gap"], "frac_closed": j[m]["frac_closed"],
                         "n_groups_fit": "%d/%d" % tuple(j["n_groups_fit"]), "K": 2, "d": j["d"]})
    vb = "b: not computed"
    if rows:
        B = pd.DataFrame(rows).sort_values(["fold", "arch", "seed", "scorer"])
        B.to_csv(OUT / "b_isbi_patch2_cells.csv", index=False)
        S = (B.groupby(["fold", "scorer"])
             .agg(n_cells=("frac_closed", "size"), median_truth=("auroc_truth", "median"),
                  median_leaky=("auroc_leaky", "median"), median_F2=("auroc_F2", "median"),
                  median_frac_closed=("frac_closed", "median"), min_frac_closed=("frac_closed", "min"),
                  max_frac_closed=("frac_closed", "max")).reset_index())
        S.to_csv(OUT / "b_isbi_patch2_summary.csv", index=False)
        parts += ["## (b) isbi_patch2 retrain: fraction of gap closed = (leaky - F2) / (leaky - truth)", "",
                  "### Summary (median over arch x seed cells)", "", md(S), ""]
        for f in sorted(B.fold.unique()):
            parts += ["### Fold %d cells" % f, "", md(B[B.fold == f].drop(columns="fold")), ""]
        s0 = S[S.fold == 0].set_index("scorer")
        vb = "b (fold 0): median fraction of gap closed " + ", ".join(
            "%s %.2f" % (m, s0.loc[m, "median_frac_closed"]) for m in FIT if m in s0.index)
        n = B.groupby("fold").size() // len(FIT)
        if (n < 9).any():
            stops.append("STOP (b): only %s of 9 cells per fold completed" % dict(n))
    else:
        stops.append("STOP (b): no cells completed")
    parts += ["verdict: %s; %s" % (va, vb), "", "caveats:", "- post-hoc (not in precommit)"]
    parts += ["- %s" % s for s in stops]
    (OUT / "REPORT.md").write_text("\n".join(parts) + "\n")
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
