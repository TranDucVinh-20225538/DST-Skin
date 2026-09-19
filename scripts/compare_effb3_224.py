#!/usr/bin/env python3
"""Compare Camelyon EffB3@224 vs EffB3@300 vs R18/R50@224.

Founding cell of cua 2: EffB3@300 MSP ~0.80 vs R18/R50 ~0.51-0.57.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("outputs/reports/camelyon17/frac1/seed42")
PATHS = {
    "resnet18@224": ROOT / "resnet18_score_comparison.csv",
    "resnet50@224": ROOT / "resnet50_score_comparison.csv",
    "effb3@300": ROOT / "effb3_score_comparison.csv",
    "effb3@224": ROOT / "effb3_224_score_comparison.csv",
}
R18_R50_BAND = (0.51, 0.57)
EFFB3_300_MSP = 0.80


def auroc(path: Path, method: str) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    hit = df[df["Method"].astype(str) == method]
    if hit.empty:
        return None
    return float(hit.iloc[0]["AUROC"])


def main() -> None:
    rows = []
    for name, path in PATHS.items():
        rec = {"model": name, "path": str(path), "exists": path.exists()}
        for m in ("msp", "energy", "mahalanobis", "knn"):
            rec[m] = auroc(path, m)
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = ROOT / "effb3_224_vs_300_comparison.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")

    msp224 = auroc(PATHS["effb3@224"], "msp")
    msp300 = auroc(PATHS["effb3@300"], "msp")
    msp_r18 = auroc(PATHS["resnet18@224"], "msp")
    msp_r50 = auroc(PATHS["resnet50@224"], "msp")
    if msp224 is None:
        print("\nEffB3@224 scores not ready yet.")
        return

    lo, hi = R18_R50_BAND
    still_outlier = msp224 >= (EFFB3_300_MSP - 0.08)  # still near 0.80
    dropped_to_resnet = lo - 0.05 <= msp224 <= hi + 0.05
    print(
        f"\nMSP  R18@224={msp_r18:.3f}  R50@224={msp_r50:.3f}  "
        f"EffB3@300={msp300:.3f}  EffB3@224={msp224:.3f}"
    )
    if still_outlier and not dropped_to_resnet:
        print(
            "VERDICT: architecture — EffB3@224 MSP still an outlier vs R18/R50. "
            "Cua 2 founding observation survives."
        )
    elif dropped_to_resnet:
        print(
            "VERDICT: resolution artifact — EffB3@224 MSP fell back to the "
            "R18/R50 band. Cua 2 founding cell was not architecture."
        )
    else:
        print(
            f"VERDICT: mixed — EffB3@224 MSP={msp224:.3f} is between "
            f"R18/R50 ~{lo:.2f}-{hi:.2f} and EffB3@300 ~{msp300:.3f}. "
            "Do not treat as clean architecture or clean resolution."
        )


if __name__ == "__main__":
    main()
