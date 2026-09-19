#!/usr/bin/env python3
"""Domain-classifier AUC as an external-ish severity proxy on real domains.

Train a linear probe to tell ID vs shifted features apart. Higher AUC = clearer shift.
Does not assign strong/moderate/mild by hand. CIFAR-SVHN is excluded (semantic shift).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

JOBS = [
    ("skin_isic_pad", "resnet18", "outputs/features/resnet18_isic_pad_features.pt"),
    ("skin_isic_pad", "resnet50", "outputs/features/resnet50_isic_pad_features.pt"),
    ("skin_isic_pad", "effb3", "outputs/features/effb3_isic_pad_features.pt"),
    ("camelyon17", "resnet18", "outputs/features/camelyon17/seed42/resnet18_features.pt"),
    ("camelyon17", "resnet50", "outputs/features/camelyon17/seed42/resnet50_features.pt"),
    ("camelyon17", "effb3", "outputs/features/camelyon17/seed42/effb3_features.pt"),
    ("midog", "resnet18", "outputs/features/midog/seed42/resnet18_features.pt"),
    ("midog", "resnet50", "outputs/features/midog/seed42/resnet50_features.pt"),
    ("midog", "effb3", "outputs/features/midog/seed42/effb3_features.pt"),
]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def domain_auc(id_feats: np.ndarray, ood_feats: np.ndarray, seed: int = 42) -> dict:
    x = np.concatenate([l2_normalize(id_feats), l2_normalize(ood_feats)], axis=0)
    y = np.concatenate([
        np.zeros(len(id_feats), dtype=np.int64),
        np.ones(len(ood_feats), dtype=np.int64),
    ])
    # Cap for speed on huge Camelyon matrices while keeping class balance.
    rng = np.random.default_rng(seed)
    max_per = 5000
    idx_id = rng.choice(len(id_feats), size=min(max_per, len(id_feats)), replace=False)
    idx_ood = rng.choice(len(ood_feats), size=min(max_per, len(ood_feats)), replace=False)
    x = np.concatenate([l2_normalize(id_feats[idx_id]), l2_normalize(ood_feats[idx_ood])])
    y = np.concatenate([
        np.zeros(len(idx_id), dtype=np.int64),
        np.ones(len(idx_ood), dtype=np.int64),
    ])
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(x, y):
        pipe.fit(x[tr], y[tr])
        proba = pipe.predict_proba(x[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
    pipe.fit(x, y)
    return {
        "domain_classifier_auc_cv": float(np.mean(aucs)),
        "domain_classifier_auc_cv_std": float(np.std(aucs)),
        "n_id_used": int(len(idx_id)),
        "n_ood_used": int(len(idx_ood)),
    }


def main() -> None:
    rows = []
    for domain, backbone, path in JOBS:
        p = Path(path)
        if not p.exists():
            print(f"SKIP missing {path}")
            continue
        try:
            data = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(p, map_location="cpu")
        id_feats = to_numpy(data["val_feats"])
        ood_feats = to_numpy(data["ood_feats"])
        stats = domain_auc(id_feats, ood_feats)
        row = {"domain": domain, "backbone": backbone, **stats}
        rows.append(row)
        print(
            f"{domain:16s} {backbone:10s} AUC={stats['domain_classifier_auc_cv']:.4f} "
            f"± {stats['domain_classifier_auc_cv_std']:.4f}"
        )
    out = Path("outputs/reports/domain_classifier_auc.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
