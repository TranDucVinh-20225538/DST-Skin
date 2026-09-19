#!/usr/bin/env python3
"""Hospital-1 extract + MSP coverage for a stain_cov backbone.

Does not train. Does not overwrite official features.pt (hospital 2 stays).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.camelyon_ood import (  # noqa: E402
    _WildsPathDataset,
    build_transform,
    get_wilds_dataset,
)
from src.models.cnn_family import get_cnn_backbone, recipe as cnn_recipe
from src.utils.benchmark_metrics import build_score_comparison_df
from src.utils.feature_extractor import extract_features_and_logits
from src.utils.scoring import OODScorer

OOD_METHODS = (
    "msp",
    "energy",
    "react_energy",
    "logit_norm",
    "mahalanobis",
    "knn",
    "vim",
)
TAG = "stain_cov"
SEED = 42


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def coverage_at_risk10(scores: np.ndarray, correct: np.ndarray, target: float = 0.10) -> float | None:
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    csort = np.asarray(correct, dtype=np.float64)[order]
    n = len(csort)
    best = None
    for k in range(1, n + 1):
        risk = 1.0 - float(csort[:k].mean())
        if risk <= target:
            best = k / n
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=("resnet18", "resnet50", "densenet121"))
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    rel = f"frac1/{TAG}/seed{SEED}"
    ckpt = Path(f"data/models/camelyon17/{rel}/{args.backbone}_best.pth")
    feat = Path(f"outputs/features/camelyon17/{rel}/{args.backbone}_features.pt")
    score_h2 = Path(f"outputs/reports/camelyon17/{rel}/{args.backbone}_score_comparison.csv")
    if not ckpt.exists() or not feat.exists() or not score_h2.exists():
        raise FileNotFoundError(f"need stain_cov ckpt+features+csv: {ckpt} {feat} {score_h2}")

    try:
        pack = torch.load(feat, map_location="cpu", weights_only=False)
    except TypeError:
        pack = torch.load(feat, map_location="cpu")

    h1_pt = feat.parent / f"{args.backbone}_hospital1_ood.pt"
    h1_csv = score_h2.parent / f"{args.backbone}_hospital1_score_comparison.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not h1_pt.exists():
        model = get_cnn_backbone(args.backbone, num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        model.eval()
        bs = int(cnn_recipe(args.backbone)["batch_size"])
        dataset = get_wilds_dataset(download=False)
        tf = build_transform(args.backbone, train=False, stain_cov=False)
        loader = DataLoader(
            _WildsPathDataset(dataset.get_subset("val", transform=tf), "camelyon17/val"),
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"=== extract hospital-1 n={len(loader.dataset)} {args.backbone} ===", flush=True)
        logits, feats = extract_features_and_logits(model, loader, device)
        torch.save({"ood_logits": logits, "ood_feats": feats, "ood_split": "val"}, h1_pt)
        print(f"saved {h1_pt}", flush=True)
    else:
        print(f"skip hospital-1 extract, {h1_pt}", flush=True)

    try:
        h1 = torch.load(h1_pt, map_location="cpu", weights_only=False)
    except TypeError:
        h1 = torch.load(h1_pt, map_location="cpu")
    scorer = OODScorer(
        k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None
    )
    scorer.fit(
        to_numpy(pack["train_feats"]),
        train_labels=None,
        train_logits=to_numpy(pack["train_logits"]),
        fc_weight=to_numpy(pack["fc_weight"]),
        fc_bias=to_numpy(pack["fc_bias"]),
    )
    scores_id = scorer.get_all_scores(to_numpy(pack["val_logits"]), to_numpy(pack["val_feats"]))
    scores_h1 = scorer.get_all_scores(to_numpy(h1["ood_logits"]), to_numpy(h1["ood_feats"]))
    methods = [m for m in OOD_METHODS if m in scores_id]
    df = build_score_comparison_df(scores_id, scores_h1, methods, seed=SEED)
    df.to_csv(h1_csv, index=False)
    msp_h1 = float(df.loc[df["Method"].astype(str).str.lower() == "msp", "AUROC"].iloc[0])
    print(f"hospital-1 MSP={msp_h1:.4f} -> {h1_csv}", flush=True)

    ood_z = to_numpy(pack["ood_logits"])
    ood_y = to_numpy(pack["ood_labels"])
    msp_ood = OODScorer.score_msp(ood_z)
    ood_ok = (ood_z.argmax(1) == ood_y).astype(np.float64)
    cov = coverage_at_risk10(msp_ood, ood_ok)
    df_h2 = pd.read_csv(score_h2)
    msp_h2 = float(df_h2.loc[df_h2["Method"].astype(str).str.lower() == "msp", "AUROC"].iloc[0])
    meta = {
        "backbone": args.backbone,
        "hospital2_msp": msp_h2,
        "hospital1_msp": msp_h1,
        "hospital2_msp_coverage_at_risk10": cov,
        "n_ood_h2": int(len(ood_y)),
    }
    cov_path = score_h2.parent / f"{args.backbone}_stain_cov_aux.json"
    cov_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
