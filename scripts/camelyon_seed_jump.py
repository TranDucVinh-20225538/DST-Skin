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
# Seed-42-only in the official zoo; jump vs that seed's ResNet mean.
EXTRA = (
    "convnext_tiny",
    "mobilenet_v3_large",
    "regnet_y_3_2gf",
    "efficientnet_v2_s_224",
)
ROOT = Path("outputs/reports/camelyon17/frac1")
OUT = Path("outputs/reports/camelyon_seed_jump.txt")


def msp_of(seed: int, backbone: str) -> float:
    path = ROOT / f"seed{seed}" / f"{backbone}_score_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return float(df.loc[df["Method"] == "msp", "AUROC"].iloc[0])


def maybe_msp(seed: int, backbone: str) -> float | None:
    path = ROOT / f"seed{seed}" / f"{backbone}_score_comparison.csv"
    if not path.exists():
        return None
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

    extra_lines = ["", "== extra 4 (seed-42-only in official zoo; same jump vs that seed's ResNet mean) =="]
    extra_ready = True
    extra_rows = []
    for seed in SEEDS:
        rec = {"seed": seed, "resnet_mean": 0.5 * (msp_of(seed, "resnet18") + msp_of(seed, "resnet50"))}
        for bb in EXTRA:
            val = maybe_msp(seed, bb)
            rec[bb] = val
            if val is None:
                extra_ready = False
            else:
                rec[f"jump_{bb}"] = val - rec["resnet_mean"]
        extra_rows.append(rec)
    if extra_ready:
        edf = pd.DataFrame(extra_rows)
        for bb in EXTRA:
            jumps = np.array([r[f"jump_{bb}"] for r in extra_rows], dtype=float)
            mean = float(jumps.mean())
            std = float(jumps.std(ddof=1))
            n_hit = int((jumps >= JUMP_DELTA).sum())
            if mean >= JUMP_DELTA and n_hit >= 4:
                branch = "not a seed-42 fluke"
            elif mean >= JUMP_DELTA:
                branch = "effect present, variance high; report, do not hide"
            else:
                branch = "seed-fragile; do not read as confirmed jump"
            extra_lines.append(
                f"jump_{bb}: {mean:.3f} ± {std:.3f} ({n_hit}/5 seeds ≥{JUMP_DELTA}) → {branch}"
            )
        extra_lines.append("")
        extra_lines.append(edf.to_csv(index=False))
    else:
        extra_lines.append("pending: extra CSVs not complete. Official W unchanged.")

    lines = [
        "Camelyon seed-variance. Official W stays seed 42.",
        verdict("jump_densenet"),
        verdict("jump_effb3"),
        "",
        df.to_csv(index=False),
        *extra_lines,
    ]
    text = "\n".join(lines)
    OUT.write_text(text)
    print(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
