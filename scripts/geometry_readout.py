#!/usr/bin/env python3
"""Geometry readout of covariate shift vs unsupervised OOD scores.

Nine existing cells (skin / Camelyon-full / MIDOG x R18/R50/EffB3).
No new training. Gaussian log-density is NOT computed (monotonic in
mean-to-mean Mahalanobis). Does not touch CIFAR-10-C or the backbone zoo job.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

from src.utils.scoring import OODScorer

OUT = Path("outputs/reports/geometry_readout_table.csv")
SCATTER = Path("outputs/reports/geometry_readout_scatter.png")
PROBE = Path("outputs/reports/domain_classifier_auc.csv")
TOP_K = 5

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


def load_auroc(path: Path, method: str) -> float:
    df = pd.read_csv(path)
    hit = df[df["Method"] == method]
    if hit.empty:
        raise RuntimeError(f"{path}: missing {method}")
    return float(hit.iloc[0]["AUROC"])


def geometry(train_feats: np.ndarray, ood_feats: np.ndarray) -> dict:
    train_n = OODScorer.l2_normalize(np.asarray(train_feats, dtype=np.float32))
    ood_n = OODScorer.l2_normalize(np.asarray(ood_feats, dtype=np.float32))
    print(
        f"    Ledoit-Wolf fit n_train={len(train_n)} d={train_n.shape[1]} "
        f"n_ood={len(ood_n)}",
        flush=True,
    )
    lw = LedoitWolf().fit(train_n)
    mu_id = np.asarray(lw.location_, dtype=np.float64)
    mu_ood = ood_n.mean(axis=0).astype(np.float64)
    delta = mu_ood - mu_id
    mean_shift = float(np.linalg.norm(delta))
    prec = np.asarray(lw.precision_, dtype=np.float64)
    d2 = float(delta @ prec @ delta)
    maha_mean_shift = float(np.sqrt(max(d2, 0.0)))

    cov = np.asarray(lw.covariance_, dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    k = min(TOP_K, evecs.shape[1])
    v5 = evecs[:, -k:]
    proj = v5 @ (v5.T @ delta)
    denom = float(np.dot(delta, delta))
    pct = 100.0 * float(np.dot(proj, proj)) / (denom + 1e-12)
    return {
        "mean_shift": mean_shift,
        "maha_mean_shift": maha_mean_shift,
        "pct_shift_in_top5_eigenspace": pct,
        "n_train": int(len(train_n)),
        "n_ood": int(len(ood_n)),
        "feat_dim": int(train_n.shape[1]),
    }


def main() -> None:
    probe = pd.read_csv(PROBE)
    rows = []
    for domain, backbone, feat_path, score_path in CELLS:
        print(f"=== {domain}/{backbone} ===", flush=True)
        prow = probe[(probe["domain"] == domain) & (probe["backbone"] == backbone)]
        if prow.empty:
            raise RuntimeError(f"No probe AUC for {domain}/{backbone}")
        probe_auroc = float(prow.iloc[0]["domain_classifier_auc_cv"])
        maha = load_auroc(score_path, "mahalanobis")
        msp = load_auroc(score_path, "msp")
        knn = load_auroc(score_path, "knn")

        try:
            data = torch.load(feat_path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(feat_path, map_location="cpu")
        train_feats = to_numpy(data["train_feats"])
        ood_feats = to_numpy(data["ood_feats"])
        del data
        geo = geometry(train_feats, ood_feats)
        del train_feats, ood_feats
        gc.collect()

        row = {
            "domain": domain,
            "backbone": backbone,
            "probe_auroc": probe_auroc,
            "maha_auroc": maha,
            "msp_auroc": msp,
            "knn_auroc": knn,
            "mean_shift": geo["mean_shift"],
            "maha_mean_shift": geo["maha_mean_shift"],
            "pct_shift_in_top5_eigenspace": geo["pct_shift_in_top5_eigenspace"],
        }
        rows.append(row)
        print(
            f"    probe={probe_auroc:.4f} maha={maha:.4f} msp={msp:.4f} knn={knn:.4f} "
            f"||dmu||={geo['mean_shift']:.4f} maha_dmu={geo['maha_mean_shift']:.4f} "
            f"top5={geo['pct_shift_in_top5_eigenspace']:.2f}%",
            flush=True,
        )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}", flush=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {
            "skin_isic_pad": "#1f77b4",
            "camelyon17": "#2ca02c",
            "midog": "#d62728",
        }
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        for domain, g in df.groupby("domain"):
            ax.scatter(
                g["probe_auroc"],
                g["maha_auroc"],
                c=colors[domain],
                s=70,
                label=domain,
                zorder=3,
            )
            for _, r in g.iterrows():
                ax.annotate(
                    r["backbone"],
                    (r["probe_auroc"], r["maha_auroc"]),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                )
        ax.plot([0.5, 1.0], [0.5, 1.0], ls="--", c="0.5", lw=1, label="y = x")
        ax.set_xlabel("supervised probe AUROC (ID vs OOD linear)")
        ax.set_ylabel("unsupervised Mahalanobis AUROC")
        ax.set_title("Feature probe vs Mahalanobis readout (9 cells)")
        ax.set_xlim(0.99, 1.001)
        ax.set_ylim(0.45, 1.02)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(SCATTER, dpi=160)
        plt.close(fig)
        print(f"Wrote {SCATTER}", flush=True)
    except Exception as exc:
        print(f"scatter skipped: {exc}", flush=True)

    print("\nDomain means of pct_shift_in_top5_eigenspace:")
    print(df.groupby("domain")["pct_shift_in_top5_eigenspace"].agg(["mean", "min", "max"]))
    print("\nDomain means of maha_mean_shift:")
    print(df.groupby("domain")["maha_mean_shift"].agg(["mean", "min", "max"]))
    print("\nDomain means of mean_shift (L2):")
    print(df.groupby("domain")["mean_shift"].agg(["mean", "min", "max"]))
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
