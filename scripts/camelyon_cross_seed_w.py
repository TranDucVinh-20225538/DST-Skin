#!/usr/bin/env python3
"""Cross-seed Kendall W at fixed architecture (decision_precommit_cross_seed_w.md).

Reads the existing 7-score CSVs for the four seed-variance anchors x seeds
42-46. CPU only. Official seed-42 8-CNN W=0.694 is not recomputed here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "rebuild_architecture_invariance", HERE / "rebuild_architecture_invariance.py"
)
_REBUILD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_REBUILD)

DISPLAY = _REBUILD.DISPLAY
METHODS_ORDER = _REBUILD.METHODS_ORDER
kendall_w = _REBUILD.kendall_w
ranks_rows = _REBUILD._ranks_high_is_1_rows

ROOT = Path("outputs/reports/camelyon17/frac1")
OUT_CSV = Path("outputs/reports/camelyon_cross_seed_w.csv")
OUT_TXT = Path("outputs/reports/camelyon_cross_seed_w.txt")
ANCHORS = ("resnet18", "resnet50", "densenet121", "effb3")
SEEDS = (42, 43, 44, 45, 46)
OFFICIAL_W = 0.694


OFFICIAL_RANKS = Path("outputs/reports/architecture_invariance_ranks.csv")


def auroc_vector(seed: int, backbone: str) -> np.ndarray:
    if seed == 42:
        # Seed-42 score CSVs lack some logit methods; the official W reads them from the frozen ranks file.
        df = pd.read_csv(OFFICIAL_RANKS)
        sub = df[(df.domain == "camelyon17") & (df.backbone == backbone)].set_index("method")["auroc"]
        return np.array([float(sub.loc[m]) for m in METHODS_ORDER])
    path = ROOT / f"seed{seed}" / f"{backbone}_score_comparison.csv"
    df = pd.read_csv(path)
    by = {DISPLAY[m]: float(a) for m, a in zip(df["Method"], df["AUROC"]) if m in DISPLAY}
    missing = [m for m in METHODS_ORDER if m not in by]
    if missing:
        raise RuntimeError(f"{path}: missing {missing}")
    return np.array([by[m] for m in METHODS_ORDER])


def main() -> None:
    rows = []
    per_anchor = {}
    for bb in ANCHORS:
        mat = np.stack([auroc_vector(s, bb) for s in SEEDS])
        ranks = ranks_rows(mat)
        w = kendall_w(ranks)
        per_anchor[bb] = w
        for s, r, a in zip(SEEDS, ranks, mat):
            for m, rk, au in zip(METHODS_ORDER, r, a):
                rows.append({"anchor": bb, "seed": s, "method": m, "auroc": au, "rank": rk})

    cross_arch = {}
    for s in SEEDS:
        mat = np.stack([auroc_vector(s, bb) for bb in ANCHORS])
        cross_arch[s] = kendall_w(ranks_rows(mat))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    vals = np.array(list(per_anchor.values()))
    ca = np.array(list(cross_arch.values()))
    lines = [
        "Cross-seed Kendall W, fixed architecture, Camelyon hospital-2, 7 scores.",
        "Ranking unit = seed (5 seeds: 42-46). Source: existing *_score_comparison.csv.",
        f"Official cross-architecture W (8 CNN, seed 42) = {OFFICIAL_W} (not recomputed).",
        "",
        "W_seed per anchor (k=5 seeds, n=7 methods):",
    ]
    lines += [f"  {bb:12s} {w:.3f}" for bb, w in per_anchor.items()]
    lines += [
        f"  mean +- sd   {vals.mean():.3f} +- {vals.std(ddof=1):.3f}  (min {vals.min():.3f}, max {vals.max():.3f})",
        "",
        "Supplementary, not in the read table: cross-architecture W over the same",
        "four anchors at each seed (k=4 backbones, n=7 methods):",
    ]
    lines += [f"  seed {s}       {w:.3f}" for s, w in cross_arch.items()]
    lines += [f"  mean +- sd   {ca.mean():.3f} +- {ca.std(ddof=1):.3f}", ""]
    lines.append("Per-anchor rank matrix (rows=seed, 1=best):")
    df = pd.DataFrame(rows)
    for bb in ANCHORS:
        piv = df[df.anchor == bb].pivot(index="seed", columns="method", values="rank")[METHODS_ORDER]
        lines += [f"-- {bb}", piv.to_string()]
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
