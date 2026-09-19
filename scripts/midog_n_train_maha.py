#!/usr/bin/env python3
"""H4: is MIDOG Maha weak because ID n is too small for Ledoit-Wolf?

Subsample skin / Camelyon-full train features to n=1896 (MIDOG train size),
refit Ledoit-Wolf, recompute Maha AUROC on the same ID-val / OOD split.
20 random draws. No retrain. Does not touch zoo / CIFAR-10-C.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

from src.utils.benchmark_metrics import calc_auroc
from src.utils.scoring import OODScorer

OUT_SUMMARY = Path("outputs/reports/midog_n_train_maha_summary.csv")
OUT_DRAWS = Path("outputs/reports/midog_n_train_maha_draws.csv")
N_TARGET = 1896
N_DRAWS = 20
SEED = 42
MIDOG_BAND = (0.59, 0.73)

CELLS = [
    (
        "skin_isic_pad",
        "resnet18",
        Path("outputs/features/resnet18_isic_pad_features.pt"),
        Path("outputs/reports/resnet18_score_comparison.csv"),
    ),
    (
        "skin_isic_pad",
        "resnet50",
        Path("outputs/features/resnet50_isic_pad_features.pt"),
        Path("outputs/reports/resnet50_score_comparison.csv"),
    ),
    (
        "skin_isic_pad",
        "effb3",
        Path("outputs/features/effb3_isic_pad_features.pt"),
        Path("outputs/reports/effb3_score_comparison.csv"),
    ),
    (
        "camelyon17",
        "resnet18",
        Path("outputs/features/camelyon17/frac1/seed42/resnet18_features.pt"),
        Path("outputs/reports/camelyon17/frac1/seed42/resnet18_score_comparison.csv"),
    ),
    (
        "camelyon17",
        "resnet50",
        Path("outputs/features/camelyon17/frac1/seed42/resnet50_features.pt"),
        Path("outputs/reports/camelyon17/frac1/seed42/resnet50_score_comparison.csv"),
    ),
    (
        "camelyon17",
        "effb3",
        Path("outputs/features/camelyon17/frac1/seed42/effb3_features.pt"),
        Path("outputs/reports/camelyon17/frac1/seed42/effb3_score_comparison.csv"),
    ),
    (
        "midog",
        "resnet18",
        Path("outputs/features/midog/seed42/resnet18_features.pt"),
        Path("outputs/reports/midog/seed42/resnet18_score_comparison.csv"),
    ),
    (
        "midog",
        "resnet50",
        Path("outputs/features/midog/seed42/resnet50_features.pt"),
        Path("outputs/reports/midog/seed42/resnet50_score_comparison.csv"),
    ),
    (
        "midog",
        "effb3",
        Path("outputs/features/midog/seed42/effb3_features.pt"),
        Path("outputs/reports/midog/seed42/effb3_score_comparison.csv"),
    ),
]


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_csv_maha(path: Path) -> float:
    df = pd.read_csv(path)
    hit = df[df["Method"].astype(str) == "mahalanobis"]
    if hit.empty:
        raise ValueError(f"no mahalanobis row in {path}")
    return float(hit.iloc[0]["AUROC"])


def load_split_feats(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    train = np.ascontiguousarray(to_numpy(data["train_feats"]), dtype=np.float32)
    val = np.ascontiguousarray(to_numpy(data["val_feats"]), dtype=np.float32)
    ood = np.ascontiguousarray(to_numpy(data["ood_feats"]), dtype=np.float32)
    del data
    gc.collect()
    return train, val, ood


def fit_lw(train_feats: np.ndarray) -> LedoitWolf:
    feats_n = OODScorer.l2_normalize(np.asarray(train_feats, dtype=np.float32))
    return LedoitWolf().fit(feats_n)


def maha_scores(feats: np.ndarray, lw: LedoitWolf) -> np.ndarray:
    feats_n = OODScorer.l2_normalize(np.asarray(feats, dtype=np.float32))
    diff = np.asarray(feats_n, dtype=np.float64) - np.asarray(lw.location_, dtype=np.float64)
    left = diff @ np.asarray(lw.precision_, dtype=np.float64)
    d2 = np.sum(left * diff, axis=1)
    return -np.sqrt(np.maximum(d2, 0.0))


def precision_cond(lw: LedoitWolf) -> float:
    prec = np.asarray(lw.precision_, dtype=np.float64)
    return float(np.linalg.cond(prec))


def maha_auroc(val_feats: np.ndarray, ood_feats: np.ndarray, lw: LedoitWolf) -> float:
    return calc_auroc(maha_scores(val_feats, lw), maha_scores(ood_feats, lw))


def main() -> None:
    rng = np.random.default_rng(SEED)
    summary_rows: list[dict] = []
    draw_rows: list[dict] = []

    for domain, backbone, feat_path, csv_path in CELLS:
        print(f"\n=== {domain}/{backbone} ===", flush=True)
        auroc_full_csv = load_csv_maha(csv_path)
        train, val, ood = load_split_feats(feat_path)
        n_full, d = int(train.shape[0]), int(train.shape[1])
        print(
            f"  n_train={n_full} d={d} n/d={n_full / d:.3f} "
            f"n_val={len(val)} n_ood={len(ood)} csv_maha={auroc_full_csv:.4f}",
            flush=True,
        )

        print("  fitting Ledoit-Wolf on full train...", flush=True)
        lw_full = fit_lw(train)
        cond_full = precision_cond(lw_full)
        shrink_full = float(lw_full.shrinkage_)
        auroc_full_refit = maha_auroc(val, ood, lw_full)
        print(
            f"  full refit AUROC={auroc_full_refit:.4f} "
            f"cond(prec)={cond_full:.3e} shrinkage={shrink_full:.4f}",
            flush=True,
        )
        del lw_full
        gc.collect()

        row = {
            "domain": domain,
            "backbone": backbone,
            "n_full": n_full,
            "d": d,
            "n_over_d_full": n_full / d,
            "auroc_full_csv": auroc_full_csv,
            "auroc_full_refit": auroc_full_refit,
            "cond_prec_full": cond_full,
            "shrinkage_full": shrink_full,
            "n_target": N_TARGET,
            "n_draws": N_DRAWS,
        }

        if domain == "midog":
            row.update(
                {
                    "subsampled": False,
                    "n_sub": n_full,
                    "auroc_sub_mean": auroc_full_refit,
                    "auroc_sub_std": 0.0,
                    "auroc_sub_min": auroc_full_refit,
                    "auroc_sub_max": auroc_full_refit,
                    "cond_prec_sub_mean": cond_full,
                    "cond_prec_sub_std": 0.0,
                    "shrinkage_sub_mean": shrink_full,
                    "note": "native MIDOG train size; not subsampled",
                }
            )
            summary_rows.append(row)
            del train, val, ood
            gc.collect()
            continue

        if n_full <= N_TARGET:
            row.update(
                {
                    "subsampled": False,
                    "n_sub": n_full,
                    "auroc_sub_mean": auroc_full_refit,
                    "auroc_sub_std": 0.0,
                    "auroc_sub_min": auroc_full_refit,
                    "auroc_sub_max": auroc_full_refit,
                    "cond_prec_sub_mean": cond_full,
                    "cond_prec_sub_std": 0.0,
                    "shrinkage_sub_mean": shrink_full,
                    "note": f"n_full={n_full} <= {N_TARGET}; skip subsample",
                }
            )
            summary_rows.append(row)
            del train, val, ood
            gc.collect()
            continue

        aurocs = np.empty(N_DRAWS, dtype=np.float64)
        conds = np.empty(N_DRAWS, dtype=np.float64)
        shrinks = np.empty(N_DRAWS, dtype=np.float64)
        for i in range(N_DRAWS):
            idx = rng.choice(n_full, N_TARGET, replace=False)
            lw = fit_lw(train[idx])
            aurocs[i] = maha_auroc(val, ood, lw)
            conds[i] = precision_cond(lw)
            shrinks[i] = float(lw.shrinkage_)
            draw_rows.append(
                {
                    "domain": domain,
                    "backbone": backbone,
                    "draw": i,
                    "n_sub": N_TARGET,
                    "auroc": aurocs[i],
                    "cond_prec": conds[i],
                    "shrinkage": shrinks[i],
                }
            )
            print(
                f"  draw {i:02d}/{N_DRAWS - 1} AUROC={aurocs[i]:.4f} "
                f"cond={conds[i]:.3e} shrink={shrinks[i]:.4f}",
                flush=True,
            )
            del lw
            gc.collect()

        row.update(
            {
                "subsampled": True,
                "n_sub": N_TARGET,
                "auroc_sub_mean": float(aurocs.mean()),
                "auroc_sub_std": float(aurocs.std(ddof=1)),
                "auroc_sub_min": float(aurocs.min()),
                "auroc_sub_max": float(aurocs.max()),
                "cond_prec_sub_mean": float(conds.mean()),
                "cond_prec_sub_std": float(conds.std(ddof=1)),
                "shrinkage_sub_mean": float(shrinks.mean()),
                "note": "",
            }
        )
        lo, hi = MIDOG_BAND
        in_band = lo <= row["auroc_sub_mean"] <= hi
        print(
            f"  SUB n={N_TARGET} AUROC={row['auroc_sub_mean']:.4f} "
            f"+- {row['auroc_sub_std']:.4f}  "
            f"(full csv {auroc_full_csv:.4f}, drop "
            f"{auroc_full_csv - row['auroc_sub_mean']:+.4f})  "
            f"in_MIDOG_band={in_band}",
            flush=True,
        )
        summary_rows.append(row)
        del train, val, ood
        gc.collect()

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(draw_rows).to_csv(OUT_DRAWS, index=False)
    print(f"\nWrote {OUT_SUMMARY}", flush=True)
    print(df.to_string(index=False), flush=True)
    print(
        "\nH4 rule: if skin/Camelyon mean subsample AUROC falls into "
        f"MIDOG band {MIDOG_BAND}, sample size can explain MIDOG Maha. "
        "If it stays near full-n AUROC, stop the 'why MIDOG' line.",
        flush=True,
    )


if __name__ == "__main__":
    main()
