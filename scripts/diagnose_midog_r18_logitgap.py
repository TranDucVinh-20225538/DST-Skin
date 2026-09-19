#!/usr/bin/env python3
"""MIDOG LogitGap sign-flip diagnostic (R18 vs R50/EffB3). Diagnostic only.

Does not change locked trim=1% in benchmark_metrics.py.
Does not touch Camelyon / CIFAR-10-C jobs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import trim_mean

from src.utils.benchmark_metrics import aggregate_logit_gap, per_sample_logit_gap

SEED = 42
BACKBONES = ("resnet18", "resnet50", "effb3")
OUT_DIR = Path("outputs/reports/midog/seed42")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_logits(backbone: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(f"outputs/features/midog/seed{SEED}/{backbone}_features.pt")
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    return to_numpy(data["val_logits"]), to_numpy(data["ood_logits"])


def iqr_outlier_frac(x: np.ndarray, k: float = 3.0) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return float(np.mean((x < lo) | (x > hi)))


def summarize(backbone: str) -> dict:
    id_logits, ood_logits = load_logits(backbone)
    id_g = per_sample_logit_gap(id_logits)
    ood_g = per_sample_logit_gap(ood_logits)
    row = {
        "backbone": backbone,
        "n_id": int(len(id_g)),
        "n_ood": int(len(ood_g)),
        "id_mean": float(np.mean(id_g)),
        "ood_mean": float(np.mean(ood_g)),
        "id_median": float(np.median(id_g)),
        "ood_median": float(np.median(ood_g)),
        "id_p01": float(np.percentile(id_g, 1)),
        "id_p99": float(np.percentile(id_g, 99)),
        "ood_p01": float(np.percentile(ood_g, 1)),
        "ood_p99": float(np.percentile(ood_g, 99)),
        "id_max": float(np.max(id_g)),
        "ood_max": float(np.max(ood_g)),
        "ood_frac_beyond_3iqr": iqr_outlier_frac(ood_g, 3.0),
        "id_frac_beyond_3iqr": iqr_outlier_frac(id_g, 3.0),
        "gap_raw_mean": float(np.mean(ood_g) - np.mean(id_g)),
        "gap_median": float(np.median(ood_g) - np.median(id_g)),
    }
    for trim in (0.0, 0.01, 0.05, 0.10):
        if trim == 0.0:
            a, b = float(np.mean(id_g)), float(np.mean(ood_g))
        elif trim == 0.01:
            a, b = aggregate_logit_gap(id_g), aggregate_logit_gap(ood_g)
        else:
            a = float(trim_mean(id_g, proportiontocut=trim))
            b = float(trim_mean(ood_g, proportiontocut=trim))
        row[f"id_trim_{int(trim*100):02d}"] = a
        row[f"ood_trim_{int(trim*100):02d}"] = b
        row[f"gap_trim_{int(trim*100):02d}"] = b - a
    return row, id_g, ood_g


def plot_r18(id_g: np.ndarray, ood_g: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
    for ax, arr, title, color in (
        (axes[0], id_g, f"ID test 1a (n={len(id_g)})", "#1f77b4"),
        (axes[1], ood_g, f"OOD 1b+1c (n={len(ood_g)})", "#d62728"),
    ):
        ax.hist(arr, bins=40, color=color, alpha=0.75, density=True)
        ax.axvline(np.mean(arr), color="black", ls="--", lw=1, label=f"mean {np.mean(arr):.2f}")
        ax.axvline(np.median(arr), color="gray", ls=":", lw=1, label=f"median {np.median(arr):.2f}")
        t1 = aggregate_logit_gap(arr)
        t10 = float(trim_mean(arr, 0.10))
        ax.axvline(t1, color="green", ls="-.", lw=1, label=f"trim1% {t1:.2f}")
        ax.axvline(t10, color="orange", ls="-.", lw=1, label=f"trim10% {t10:.2f}")
        ax.set_title(title)
        ax.set_xlabel("per-sample LogitGap")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("MIDOG ResNet-18 LogitGap (max − mean of others)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    r18_id = r18_ood = None
    for bb in BACKBONES:
        row, id_g, ood_g = summarize(bb)
        rows.append(row)
        print(
            f"{bb:10s}  raw={row['gap_raw_mean']:+.4f}  "
            f"trim1={row['gap_trim_01']:+.4f}  trim5={row['gap_trim_05']:+.4f}  "
            f"trim10={row['gap_trim_10']:+.4f}  median={row['gap_median']:+.4f}  "
            f"OOD>3IQR={100*row['ood_frac_beyond_3iqr']:.2f}%"
        )
        if bb == "resnet18":
            r18_id, r18_ood = id_g, ood_g
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "logitgap_trim_diagnostic.csv"
    df.to_csv(csv_path, index=False)
    png_path = OUT_DIR / "r18_logitgap_hist.png"
    plot_r18(r18_id, r18_ood, png_path)

    r18 = df.loc[df.backbone == "resnet18"].iloc[0]
    r18_trims = [r18["gap_trim_01"], r18["gap_trim_05"], r18["gap_trim_10"], r18["gap_median"]]
    r18_signs = [np.sign(v) for v in r18_trims]
    r50_sign = np.sign(df.loc[df.backbone == "resnet50", "gap_trim_01"].iloc[0])
    r18_consistent = len(set(r18_signs)) == 1
    r18_matches_r50 = r18_signs[0] == r50_sign
    tail_frac = r18["ood_frac_beyond_3iqr"]
    if r18_consistent and (not r18_matches_r50) and tail_frac < 0.02:
        verdict = (
            "REAL backbone-dependent effect: R18 LogitGap stays positive in the bulk "
            f"(median {r18['gap_median']:+.3f}; trim 1/5/10% all same sign; "
            f"OOD beyond 3×IQR = {100*tail_frac:.2f}%). "
            f"Locked trim=1% is {r18['gap_trim_01']:+.4f}. "
            "Not a tail artifact — do not change benchmark_metrics.py; report R18 as "
            "backbone-dependent (ΔAUROC also largest at R18)."
        )
    elif (not r18_matches_r50) and np.sign(r18["gap_trim_05"]) == r50_sign:
        verdict = (
            "TAIL ARTIFACT: trim 1% flipped vs R50, but heavier trim recovers R50 sign. "
            "Keep locked trim=1%; flag R18 LogitGap as estimator-sensitive."
        )
    else:
        verdict = (
            "MIXED: R18 sign is not stable across mean/median/trim. "
            "Keep locked trim=1%; do not treat R18 LogitGap as a main-table number."
        )
    print("\nVERDICT:", verdict)
    (OUT_DIR / "r18_logitgap_verdict.txt").write_text(verdict + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
