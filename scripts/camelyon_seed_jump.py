#!/usr/bin/env python3
"""Score seed-variance jump against decision_precommit_seed_variance.md.

Run after seeds 43–46 CSVs exist. Seed 42 is the frozen official cell.
Does not rewrite official W.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

JUMP_DELTA = 0.15
SEEDS = (42, 43, 44, 45, 46)
ANCHORS = ("resnet18", "resnet50", "densenet121", "effb3")
ROOT = Path("outputs/reports/camelyon17/frac1")
OUT = Path("outputs/reports/camelyon_seed_jump.txt")


def msp_of(seed: int, backbone: str) -> float:
    path = ROOT / f"seed{seed}" / f"{backbone}_score_comparison.csv"
    df = pd.read_csv(path)
    return float(df.loc[df["Method"] == "msp", "AUROC"].iloc[0])


def main() -> None:
    rows = []
    for seed in SEEDS:
        rec = {"seed": seed}
        for bb in ANCHORS:
            rec[bb] = msp_of(seed, bb)
        rec["resnet_mean"] = 0.5 * (rec["resnet18"] + rec["resnet50"])
        rec["jump_densenet"] = rec["densenet121"] - rec["resnet_mean"]
        rec["jump_effb3"] = rec["effb3"] - rec["resnet_mean"]
        rec["dense_hit"] = rec["jump_densenet"] >= JUMP_DELTA
        rec["effb3_hit"] = rec["jump_effb3"] >= JUMP_DELTA
        rows.append(rec)
    df = pd.DataFrame(rows)

    def verdict(col: str) -> str:
        mean = float(df[col].mean())
        std = float(df[col].std(ddof=1))
        n_hit = int((df[col] >= JUMP_DELTA).sum())
        if mean >= JUMP_DELTA and n_hit >= 4:
            branch = "not a seed-42 fluke"
        elif mean >= JUMP_DELTA:
            branch = "effect present, variance high; report, do not hide"
        else:
            branch = "seed-fragile; official A cannot stay unqualified"
        return f"{col}: {mean:.3f} ± {std:.3f} ({n_hit}/5 seeds ≥{JUMP_DELTA}) → {branch}"

    lines = [
        "Camelyon seed-variance. Official W stays seed 42.",
        verdict("jump_densenet"),
        verdict("jump_effb3"),
        "",
        df.to_csv(index=False),
    ]
    text = "\n".join(lines)
    OUT.write_text(text)
    print(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
