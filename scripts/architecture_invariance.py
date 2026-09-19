#!/usr/bin/env python3
"""Architecture-invariance of OOD methods across R18/R50/EffB3.

Uses saved score CSVs. Fills missing ViM/ReAct/ELogitNorm from existing
feature files (CPU only — those three were scored but not written for
Camelyon-full / MIDOG). No GPU, no retrain, no CIFAR-10-C.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.benchmark_metrics import calc_auroc
from src.utils.ood_vim_react import (
    fit_react_on_fc,
    fit_vim,
    react_energy_score,
    vim_score,
)

OUT = Path("outputs/reports/architecture_invariance.csv")
METHODS = (
    "msp",
    "energy",
    "logit_norm",
    "vim",
    "react_energy",
    "mahalanobis",
    "knn",
)
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
CSV_COL = {"resnet18": "auroc_r18", "resnet50": "auroc_r50", "effb3": "auroc_effb3"}

CELLS = {
    ("skin_isic_pad", "resnet18"): (
        Path("outputs/reports/resnet18_score_comparison.csv"),
        Path("outputs/features/resnet18_isic_pad_features.pt"),
    ),
    ("skin_isic_pad", "resnet50"): (
        Path("outputs/reports/resnet50_score_comparison.csv"),
        Path("outputs/features/resnet50_isic_pad_features.pt"),
    ),
    ("skin_isic_pad", "effb3"): (
        Path("outputs/reports/effb3_score_comparison.csv"),
        Path("outputs/features/effb3_isic_pad_features.pt"),
    ),
    ("camelyon17", "resnet18"): (
        Path("outputs/reports/camelyon17/frac1/seed42/resnet18_score_comparison.csv"),
        Path("outputs/features/camelyon17/frac1/seed42/resnet18_features.pt"),
    ),
    ("camelyon17", "resnet50"): (
        Path("outputs/reports/camelyon17/frac1/seed42/resnet50_score_comparison.csv"),
        Path("outputs/features/camelyon17/frac1/seed42/resnet50_features.pt"),
    ),
    ("camelyon17", "effb3"): (
        Path("outputs/reports/camelyon17/frac1/seed42/effb3_score_comparison.csv"),
        Path("outputs/features/camelyon17/frac1/seed42/effb3_features.pt"),
    ),
    ("midog", "resnet18"): (
        Path("outputs/reports/midog/seed42/resnet18_score_comparison.csv"),
        Path("outputs/features/midog/seed42/resnet18_features.pt"),
    ),
    ("midog", "resnet50"): (
        Path("outputs/reports/midog/seed42/resnet50_score_comparison.csv"),
        Path("outputs/features/midog/seed42/resnet50_features.pt"),
    ),
    ("midog", "effb3"): (
        Path("outputs/reports/midog/seed42/effb3_score_comparison.csv"),
        Path("outputs/features/midog/seed42/effb3_features.pt"),
    ),
}


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_csv_auroc(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        out[str(row["Method"])] = float(row["AUROC"])
    return out


def load_features(path: Path) -> dict:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    keep = {}
    for k in (
        "train_logits",
        "train_feats",
        "val_logits",
        "val_feats",
        "ood_logits",
        "ood_feats",
        "fc_weight",
        "fc_bias",
    ):
        if k in data:
            keep[k] = to_numpy(data[k])
    del data
    return keep


def fill_missing(have: dict[str, float], feats: dict) -> dict[str, float]:
    """Point AUROC for methods not in the CSV. Does not refit Maha/kNN."""
    need = [m for m in METHODS if m not in have]
    if not need:
        return have
    val_logits, ood_logits = feats["val_logits"], feats["ood_logits"]
    val_feats, ood_feats = feats["val_feats"], feats["ood_feats"]
    if "logit_norm" in need:
        have["logit_norm"] = calc_auroc(
            np.linalg.norm(val_logits, axis=1),
            np.linalg.norm(ood_logits, axis=1),
        )
        print(f"    filled logit_norm={have['logit_norm']:.4f}", flush=True)
    if "react_energy" in need:
        params = fit_react_on_fc(
            feats["train_feats"], feats["fc_weight"], feats["fc_bias"]
        )
        have["react_energy"] = calc_auroc(
            react_energy_score(val_feats, params),
            react_energy_score(ood_feats, params),
        )
        print(f"    filled react_energy={have['react_energy']:.4f}", flush=True)
    if "vim" in need:
        print("    fitting ViM (CPU)...", flush=True)
        train_feats = feats["train_feats"]
        train_logits = feats["train_logits"]
        if len(train_feats) > 30_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(train_feats), 30_000, replace=False)
            train_feats = train_feats[idx]
            train_logits = train_logits[idx]
            print(f"    ViM train subsample n={len(idx)}", flush=True)
        vparams = fit_vim(
            train_feats,
            train_logits,
            feats["fc_weight"],
            feats["fc_bias"],
        )
        # Match OODScorer.get_all_scores: scores["vim"] = -vim_score(...)
        have["vim"] = calc_auroc(
            -vim_score(val_feats, vparams),
            -vim_score(ood_feats, vparams),
        )
        print(f"    filled vim={have['vim']:.4f}", flush=True)
    return have


def main() -> None:
    grid: dict[tuple[str, str], dict[str, float]] = {}
    for (domain, backbone), (csv_path, feat_path) in CELLS.items():
        print(f"=== {domain}/{backbone} ===", flush=True)
        have = load_csv_auroc(csv_path)
        missing = [m for m in METHODS if m not in have]
        if missing:
            print(f"  missing {missing}; load {feat_path}", flush=True)
            feats = load_features(feat_path)
            have = fill_missing(have, feats)
            del feats
        else:
            print("  CSV complete", flush=True)
        grid[(domain, backbone)] = have

    rows = []
    domains = ("skin_isic_pad", "camelyon17", "midog")
    for method in METHODS:
        for domain in domains:
            vals = []
            rec = {"method": DISPLAY[method], "domain": domain}
            for bb in BACKBONES:
                a = float(grid[(domain, bb)][method])
                rec[CSV_COL[bb]] = a
                vals.append(a)
            rec["spread"] = float(np.std(vals, ddof=1))
            rec["mean_auroc"] = float(np.mean(vals))
            rows.append(rec)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print(df.to_string(index=False))

    summary = (
        df.groupby("method", sort=False)
        .agg(
            mean_auroc=("mean_auroc", "mean"),
            mean_spread=("spread", "mean"),
        )
        .reset_index()
    )
    summary["mean_minus_spread"] = summary["mean_auroc"] - summary["mean_spread"]
    summary = summary.sort_values("mean_minus_spread", ascending=False)
    print("\nArchitecture-invariance (mean AUROC and mean std across 3 domains):")
    print(summary.to_string(index=False))
    best = summary.iloc[0]
    print(
        f"\nBest high-mean / low-spread combo (by mean_auroc - mean_spread): "
        f"{best['method']}  mean={best['mean_auroc']:.3f}  spread={best['mean_spread']:.3f}"
    )


if __name__ == "__main__":
    main()
