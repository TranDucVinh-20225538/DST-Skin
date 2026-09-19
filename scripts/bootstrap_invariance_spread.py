#!/usr/bin/env python3
"""Bootstrap CI on cross-backbone AUROC spread (architecture invariance).

Uses the existing 95% AUROC CIs (1000-resample) to draw Gaussian AUROCs,
recompute per-domain std across backbones, then mean spread across domains.
Does not touch CIFAR-10-C / BN / ECE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("outputs/reports")
OUT = ROOT / "architecture_invariance_spread_ci.csv"
N_BOOT = 1000
SEED = 42
Z95 = 1.959963984540054

METHODS = ("msp", "energy", "logit_norm", "vim", "react_energy", "mahalanobis", "knn")
DISPLAY = {
    "msp": "MSP",
    "energy": "Energy",
    "logit_norm": "ELogitNorm",
    "vim": "ViM",
    "react_energy": "ReAct",
    "mahalanobis": "Mahalanobis",
    "knn": "kNN",
}
BACKBONES = ("resnet18", "resnet50", "effb3")
DOMAINS = ("skin_isic_pad", "camelyon17", "midog")

SCORE_PATHS = {
    ("skin_isic_pad", "resnet18"): ROOT / "resnet18_score_comparison.csv",
    ("skin_isic_pad", "resnet50"): ROOT / "resnet50_score_comparison.csv",
    ("skin_isic_pad", "effb3"): ROOT / "effb3_score_comparison.csv",
    ("camelyon17", "resnet18"): ROOT / "camelyon17/frac1/seed42/resnet18_score_comparison.csv",
    ("camelyon17", "resnet50"): ROOT / "camelyon17/frac1/seed42/resnet50_score_comparison.csv",
    ("camelyon17", "effb3"): ROOT / "camelyon17/frac1/seed42/effb3_score_comparison.csv",
    ("midog", "resnet18"): ROOT / "midog/seed42/resnet18_score_comparison.csv",
    ("midog", "resnet50"): ROOT / "midog/seed42/resnet50_score_comparison.csv",
    ("midog", "effb3"): ROOT / "midog/seed42/effb3_score_comparison.csv",
}


def se_from_ci(lo, hi) -> float | None:
    if pd.isna(lo) or pd.isna(hi):
        return None
    return float(hi - lo) / (2.0 * Z95)


def load_cells() -> dict:
    wide = pd.read_csv(ROOT / "architecture_invariance.csv")
    cells = {}
    for method in METHODS:
        disp = DISPLAY[method]
        for domain in DOMAINS:
            row = wide[(wide["method"] == disp) & (wide["domain"] == domain)]
            if row.empty:
                continue
            r = row.iloc[0]
            for bb, col in (
                ("resnet18", "auroc_r18"),
                ("resnet50", "auroc_r50"),
                ("effb3", "auroc_effb3"),
            ):
                path = SCORE_PATHS[(domain, bb)]
                se = None
                protocol = "point_only"
                if path.exists():
                    sdf = pd.read_csv(path)
                    hit = sdf[sdf["Method"] == method]
                    if not hit.empty and "AUROC_CI_low" in sdf.columns:
                        se = se_from_ci(
                            hit.iloc[0].get("AUROC_CI_low"),
                            hit.iloc[0].get("AUROC_CI_high"),
                        )
                        if se is not None:
                            protocol = "auroc_ci_gaussian"
                cells[(domain, method, bb)] = {
                    "auroc": float(r[col]),
                    "se": 0.0 if se is None else float(se),
                    "protocol": protocol,
                }
    return cells


def mean_spread(draws: dict[tuple[str, str], float], method: str) -> float:
    spreads = []
    for domain in DOMAINS:
        vals = [draws[(domain, bb)] for bb in BACKBONES]
        spreads.append(float(np.std(vals, ddof=1)))
    return float(np.mean(spreads))


def main() -> None:
    cells = load_cells()
    rng = np.random.default_rng(SEED)
    boot = {m: np.empty(N_BOOT) for m in METHODS}

    for i in range(N_BOOT):
        draws = {}
        for domain in DOMAINS:
            for bb in BACKBONES:
                for method in METHODS:
                    c = cells[(domain, method, bb)]
                    a = c["auroc"] if c["se"] <= 0 else rng.normal(c["auroc"], c["se"])
                    draws[(domain, bb, method)] = a
        for method in METHODS:
            d = {(dom, bb): draws[(dom, bb, method)] for dom in DOMAINS for bb in BACKBONES}
            boot[method][i] = mean_spread(
                {(dom, bb): d[(dom, bb)] for dom in DOMAINS for bb in BACKBONES},
                method,
            )

    # point estimate from original AUROCs
    point = {}
    for method in METHODS:
        d = {(dom, bb): cells[(dom, method, bb)]["auroc"] for dom in DOMAINS for bb in BACKBONES}
        point[method] = mean_spread(d, method)

    rows = []
    for method in METHODS:
        arr = boot[method]
        rows.append({
            "method": DISPLAY[method],
            "mean_spread": point[method],
            "spread_ci_lo": float(np.percentile(arr, 2.5)),
            "spread_ci_hi": float(np.percentile(arr, 97.5)),
            "n_boot": N_BOOT,
            "n_backbones": 3,
            "n_domains": 3,
            "ci_source": "AUROC 95% CI (gaussian) where present; else point",
        })
    df = pd.DataFrame(rows)

    maha = boot["mahalanobis"]
    print("P(mean_spread_Maha < mean_spread_other) over 1000 AUROC-CI draws:")
    for other in ("msp", "energy", "logit_norm", "knn"):
        p = float(np.mean(maha < boot[other]))
        df.loc[df["method"] == DISPLAY[other], "p_maha_spread_lt"] = p
        print(f"  vs {DISPLAY[other]:12s}  P={p:.3f}")
    df.loc[df["method"] == "Mahalanobis", "p_maha_spread_lt"] = np.nan

    # protocol flags
    n_ci = sum(1 for c in cells.values() if c["protocol"] == "auroc_ci_gaussian")
    n_pt = sum(1 for c in cells.values() if c["protocol"] == "point_only")
    print(f"\nCells with AUROC CI: {n_ci}; point-only (ViM/ReAct/ELogitNorm fill): {n_pt}")
    print(
        "FLAG: Camelyon ViM/ReAct/ELogitNorm AUROCs were filled from features; "
        "ViM covariance used a 30k train subsample vs Maha/kNN/MSP fit on full "
        "302436 train. Those three methods are NOT protocol-matched to Maha/MSP "
        "on Camelyon. Skin ViM/ReAct used full train (CSV). MIDOG ViM used full "
        "train (small n)."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
