#!/usr/bin/env python3
"""Recompute locked dispersion metrics (LogitGap trim-1% primary; CVID appendix)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from torch.serialization import add_safe_globals
except ImportError:
    def add_safe_globals(_):
        pass

from src.utils.benchmark_metrics import build_dispersion_vs_separability_table

SEED = 42

JOBS = [
    # Skin (ISIC → PAD)
    ("skin", "resnet18", "outputs/features/resnet18_isic_pad_features.pt",
     "outputs/reports/resnet18_dispersion_vs_separability.csv"),
    ("skin", "resnet50", "outputs/features/resnet50_isic_pad_features.pt",
     "outputs/reports/resnet50_dispersion_vs_separability.csv"),
    ("skin", "effb3", "outputs/features/effb3_isic_pad_features.pt",
     "outputs/reports/effb3_dispersion_vs_separability.csv"),
    # CIFAR-10 → SVHN
    ("cifar_svhn", "resnet18", "outputs/features/cifar10_svhn/seed42/resnet18_features.pt",
     "outputs/reports/cifar10_svhn/seed42/resnet18_dispersion_vs_separability.csv"),
    ("cifar_svhn", "resnet50", "outputs/features/cifar10_svhn/seed42/resnet50_features.pt",
     "outputs/reports/cifar10_svhn/seed42/resnet50_dispersion_vs_separability.csv"),
    ("cifar_svhn", "effb3", "outputs/features/cifar10_svhn/seed42/effb3_features.pt",
     "outputs/reports/cifar10_svhn/seed42/effb3_dispersion_vs_separability.csv"),
    # Camelyon17
    ("camelyon17", "resnet18", "outputs/features/camelyon17/seed42/resnet18_features.pt",
     "outputs/reports/camelyon17/seed42/resnet18_dispersion_vs_separability.csv"),
    ("camelyon17", "resnet50", "outputs/features/camelyon17/seed42/resnet50_features.pt",
     "outputs/reports/camelyon17/seed42/resnet50_dispersion_vs_separability.csv"),
    ("camelyon17", "effb3", "outputs/features/camelyon17/seed42/effb3_features.pt",
     "outputs/reports/camelyon17/seed42/effb3_dispersion_vs_separability.csv"),
]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_features(path: str) -> dict:
    reconstruct = getattr(np.core.multiarray, "_reconstruct", None)
    if reconstruct is not None:
        add_safe_globals([reconstruct])
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ood_row(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["split"] == "OOD"].iloc[0]


def regenerate(domain: str, backbone: str, feat_path: str, out_csv: str, n_perm: int) -> dict:
    old = None
    if Path(out_csv).exists():
        old_df = pd.read_csv(out_csv)
        old = ood_row(old_df)

    data = load_features(feat_path)
    df = build_dispersion_vs_separability_table(
        to_numpy(data["val_logits"]),
        to_numpy(data["ood_logits"]),
        to_numpy(data["val_feats"]),
        to_numpy(data["ood_feats"]),
        n_silhouette_perm=n_perm,
        seed=SEED,
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    new = ood_row(df)

    row = {
        "domain": domain,
        "backbone": backbone,
        "cvid_gap_new": new["cvid_gap_vs_id"],
        "logit_gap_new": new["logit_gap_mean_gap_vs_id"],
        "cvid_gap_CI_low": new["cvid_gap_vs_id_CI_low"],
        "cvid_gap_CI_high": new["cvid_gap_vs_id_CI_high"],
        "logit_gap_CI_low": new["logit_gap_mean_gap_vs_id_CI_low"],
        "logit_gap_CI_high": new["logit_gap_mean_gap_vs_id_CI_high"],
        "silhouette": new["feature_silhouette_cosine"],
    }
    if old is not None:
        row["cvid_gap_old"] = old["cvid_gap_vs_id"]
        row["logit_gap_old"] = old["logit_gap_mean_gap_vs_id"]
        row["cvid_delta"] = row["cvid_gap_new"] - row["cvid_gap_old"]
        row["logit_gap_delta"] = row["logit_gap_new"] - row["logit_gap_old"]
    print(f"[{domain}/{backbone}] cvid {row.get('cvid_gap_old', 'NA')} -> {row['cvid_gap_new']:.4f} | "
          f"logit_gap {row.get('logit_gap_old', 'NA')} -> {row['logit_gap_new']:.4f}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--domain", choices=[j[0] for j in JOBS] + ["all"], default="all")
    args = parser.parse_args()

    rows = []
    for domain, backbone, feat, out in JOBS:
        if args.domain != "all" and domain != args.domain:
            continue
        if not Path(feat).exists():
            print(f"SKIP missing {feat}")
            continue
        rows.append(regenerate(domain, backbone, feat, out, args.n_perm))

    ablation = pd.DataFrame(rows)
    out_path = Path("outputs/reports/robust_dispersion_ablation.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(ablation.to_string(index=False))


if __name__ == "__main__":
    main()
