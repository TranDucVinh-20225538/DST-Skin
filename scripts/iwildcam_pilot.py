#!/usr/bin/env python3
"""#5 iWildCam WILDS: R18, R50, DenseNet-121. JUMP_DELTA=0.15 vs this domain's ResNet mean.

Does not mix into Camelyon/skin Kendall W. Hard stop 2026-10-02.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.camelyon17_pilot import (
    NUM_EPOCHS,
    OOD_METHODS,
    evaluate,
    make_optimizer,
    set_seed,
    to_numpy,
)
from src.datasets.iwildcam_ood import (
    IWILDCAM_NUM_CLASSES,
    OOD_SPLIT,
    data_ready,
    get_dataloaders,
    get_wilds_dataset,
    split_summary,
)
from src.models.cnn_family import fc_params as cnn_fc_params
from src.models.cnn_family import get_cnn_backbone, recipe as cnn_recipe
from src.utils.benchmark_metrics import (
    build_dispersion_vs_separability_table,
    build_score_comparison_df,
)
from src.utils.feature_extractor import extract_features_and_logits
from src.utils.scoring import OODScorer

BACKBONES = ("resnet18", "resnet50", "densenet121")
JUMP_DELTA = 0.15
SEED = 42


def wait_for_data() -> None:
    marker = data_ready()
    if not marker.exists():
        raise SystemExit(
            f"iWildCam not ready ({marker}). Run --stage download first; "
            "do not hold a GPU in a wait loop."
        )
    print(f"iWildCam ready: {marker}", flush=True)


def paths(backbone: str) -> dict[str, Path]:
    rel = "seed42"
    model_dir = Path(f"data/models/iwildcam/{rel}")
    feat_dir = Path(f"outputs/features/iwildcam/{rel}")
    report_dir = Path(f"outputs/reports/iwildcam/{rel}")
    for d in (model_dir, feat_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "model_best": model_dir / f"{backbone}_best.pth",
        "features": feat_dir / f"{backbone}_features.pt",
        "score_csv": report_dir / f"{backbone}_score_comparison.csv",
        "dispersion_csv": report_dir / f"{backbone}_dispersion_vs_separability.csv",
        "stem": backbone,
    }


def train_one(backbone: str, num_workers: int, device: torch.device) -> None:
    p = paths(backbone)
    if p["model_best"].exists():
        print(f"  skip train {p['model_best']}", flush=True)
        return
    rec = cnn_recipe(backbone)
    bs = int(rec["batch_size"])
    loaders = get_dataloaders(
        backbone, batch_size=bs, num_workers=num_workers, include_ood=False
    )
    model = get_cnn_backbone(backbone, IWILDCAM_NUM_CLASSES, pretrained=True).to(device)
    opt = make_optimizer(model, backbone)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    ce = nn.CrossEntropyLoss()
    best = 0.0
    print(f"=== Train iWildCam {backbone} n_train={len(loaders['train'].dataset)} ===", flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running = 0.0
        n = 0
        for images, labels, _ in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(images)
            loss = ce(logits, labels)
            loss.backward()
            opt.step()
            running += loss.item() * labels.size(0)
            n += labels.size(0)
        sched.step()
        acc, _ = evaluate(model, loaders["id_val"], device)
        print(
            f"[Epoch {epoch:02d}/{NUM_EPOCHS}] loss={running / max(n, 1):.4f} "
            f"id_val_acc={acc:.4f}",
            flush=True,
        )
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), p["model_best"])
            print(f"  saved {p['model_best']} (acc={best:.4f})", flush=True)
    del model
    torch.cuda.empty_cache()


def extract_one(backbone: str, num_workers: int, device: torch.device) -> None:
    p = paths(backbone)
    if p["features"].exists():
        print(f"  skip extract {p['features']}", flush=True)
        return
    rec = cnn_recipe(backbone)
    bs = int(rec["batch_size"])
    loaders = get_dataloaders(
        backbone, batch_size=bs, num_workers=num_workers, include_ood=True
    )
    model = get_cnn_backbone(backbone, IWILDCAM_NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(p["model_best"], map_location=device))
    model.to(device)

    def loader_labels(loader):
        parts = []
        for _, labels, _ in loader:
            parts.append(labels.numpy())
        return np.concatenate(parts).astype(np.int64)

    splits = {}
    print(f"=== Extract iWildCam {backbone} ===", flush=True)
    for name, loader in loaders.items():
        print(f"--- {name} ({len(loader.dataset)}) ---", flush=True)
        logits, feats = extract_features_and_logits(model, loader, device)
        splits[name] = {
            "logits": logits,
            "feats": feats,
            "labels": loader_labels(loader),
        }
    fc_w, fc_b = cnn_fc_params(model)
    torch.save(
        {
            "train_logits": splits["train"]["logits"],
            "train_feats": splits["train"]["feats"],
            "train_labels": splits["train"]["labels"],
            "val_logits": splits["id_val"]["logits"],
            "val_feats": splits["id_val"]["feats"],
            "val_labels": splits["id_val"]["labels"],
            "ood_logits": splits["ood"]["logits"],
            "ood_feats": splits["ood"]["feats"],
            "ood_labels": splits["ood"]["labels"],
            "fc_weight": fc_w,
            "fc_bias": fc_b,
            "ood_split": OOD_SPLIT,
            "domain": "iwildcam",
        },
        p["features"],
    )
    print(f"Saved {p['features']}", flush=True)


def analyze_one(backbone: str) -> dict:
    p = paths(backbone)
    try:
        data = torch.load(p["features"], map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(p["features"], map_location="cpu")
    scorer = OODScorer(
        k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None
    )
    scorer.fit(
        to_numpy(data["train_feats"]),
        train_labels=None,
        train_logits=to_numpy(data["train_logits"]),
        fc_weight=to_numpy(data["fc_weight"]),
        fc_bias=to_numpy(data["fc_bias"]),
    )
    scores_id = scorer.get_all_scores(to_numpy(data["val_logits"]), to_numpy(data["val_feats"]))
    scores_ood = scorer.get_all_scores(to_numpy(data["ood_logits"]), to_numpy(data["ood_feats"]))
    methods = [m for m in OOD_METHODS if m in scores_id]
    df = build_score_comparison_df(scores_id, scores_ood, methods, seed=SEED)
    df.to_csv(p["score_csv"], index=False)
    build_dispersion_vs_separability_table(
        to_numpy(data["val_logits"]),
        to_numpy(data["ood_logits"]),
        to_numpy(data["val_feats"]),
        to_numpy(data["ood_feats"]),
        seed=SEED,
    ).to_csv(p["dispersion_csv"], index=False)
    msp = float(df.loc[df["Method"] == "msp", "AUROC"].iloc[0])
    maha = float(df.loc[df["Method"] == "mahalanobis", "AUROC"].iloc[0])
    print(f"{backbone}: MSP={msp:.4f} Maha={maha:.4f}")
    return {"backbone": backbone, "msp": msp, "maha": maha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("download", "train", "extract", "analyze", "all"), default="all")
    parser.add_argument("--backbone", default="all")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    names = list(BACKBONES) if args.backbone == "all" else [args.backbone]
    set_seed(SEED)
    if args.stage == "download":
        print("Downloading WILDS iWildCam (~12GB compressed)...", flush=True)
        get_wilds_dataset(download=True)
        print("Download complete.", flush=True)
        print(split_summary(download=False))
        return
    wait_for_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}", flush=True)
    rows = []
    if args.stage in ("train", "all"):
        for bb in names:
            train_one(bb, args.num_workers, device)
    if args.stage in ("extract", "all"):
        for bb in names:
            extract_one(bb, args.num_workers, device)
    if args.stage in ("analyze", "all"):
        for bb in names:
            rows.append(analyze_one(bb))
    if rows:
        df = pd.DataFrame(rows)
        r18 = float(df.loc[df.backbone == "resnet18", "msp"].iloc[0]) if "resnet18" in df.backbone.values else float("nan")
        r50 = float(df.loc[df.backbone == "resnet50", "msp"].iloc[0]) if "resnet50" in df.backbone.values else float("nan")
        den = float(df.loc[df.backbone == "densenet121", "msp"].iloc[0]) if "densenet121" in df.backbone.values else float("nan")
        mean = np.nanmean([r18, r50])
        jumped = (not np.isnan(den)) and (den - mean >= JUMP_DELTA)
        verdict = (
            f"iWildCam ResNet mean MSP={mean:.3f}. DenseNet={den:.3f}. "
            f"jump={den - mean:+.3f} (thr {JUMP_DELTA}). "
        )
        verdict += (
            "HIT — architecture-instability is not medical-only."
            if jumped
            else "MISS — bound: may be biomedical-specific. Not a failed paper."
        )
        out = Path("outputs/reports/iwildcam_jump.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(verdict + "\n" + df.to_csv(index=False))
        print(verdict)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
