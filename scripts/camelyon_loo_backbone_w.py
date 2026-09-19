#!/usr/bin/env python3
"""Leave-one-backbone Kendall W on the frozen Camelyon-8 seed-42 table.

Structural sensitivity, not a seed CI. Do not report as 95% CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.rebuild_architecture_invariance import (
    METHODS_ORDER,
    kendall_w,
    _ranks_high_is_1_rows,
)

RANKS = Path("outputs/reports/architecture_invariance_ranks.csv")
OUT = Path("outputs/reports/architecture_invariance_loo_backbone_w.csv")


def main() -> None:
    df = pd.read_csv(RANKS)
    cam = df[df["domain"] == "camelyon17"].copy()
    backbones = sorted(cam["backbone"].unique())
    methods = [m for m in METHODS_ORDER if m in set(cam["method"])]
    mat = np.full((len(backbones), len(methods)), np.nan)
    for i, bb in enumerate(backbones):
        for j, method in enumerate(methods):
            hit = cam[(cam["backbone"] == bb) & (cam["method"] == method)]
            mat[i, j] = float(hit["auroc"].iloc[0])
    full = kendall_w(_ranks_high_is_1_rows(mat))
    rows = [
        {
            "held_out": "(none — official W)",
            "n_backbones": len(backbones),
            "kendall_w": full,
            "delta_vs_full": 0.0,
            "note": "single-seed point estimate; not a CI",
        }
    ]
    for i, held in enumerate(backbones):
        keep = [k for k in range(len(backbones)) if k != i]
        w = kendall_w(_ranks_high_is_1_rows(mat[keep]))
        rows.append(
            {
                "held_out": held,
                "n_backbones": len(keep),
                "kendall_w": w,
                "delta_vs_full": w - full,
                "note": "leave-one-backbone; structural, not seed-CI",
            }
        )
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {OUT}")
    print(
        "Do not quote a 95% CI from this table. "
        "Official W is the '(none)' row."
    )


if __name__ == "__main__":
    main()
