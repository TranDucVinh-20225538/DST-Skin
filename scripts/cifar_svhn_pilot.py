#!/usr/bin/env python3
"""Pilot: CIFAR-10 (ID) vs SVHN (OOD), 1 seed × 3 backbones."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.serialization import add_safe_globals
except ImportError:
    def add_safe_globals(_):
        pass

from src.datasets.cifar_ood import CIFAR10_NUM_CLASSES, get_dataloaders
from src.models.efficientnet_b3 import get_efficientnet_b3
from src.models.resnet18 import get_resnet18
from src.models.resnet50 import get_resnet50
from src.utils.benchmark_metrics import (
    BOOTSTRAP_METHODS,
    build_dispersion_vs_separability_table,
    build_score_comparison_df,
)
from src.utils.feature_extractor import extract_features_and_logits
from src.utils.scoring import OODScorer

BACKBONES = ("resnet18", "resnet50", "effb3")
SEED_DEFAULT = 42
NUM_EPOCHS = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model(
    backbone: str,
    num_classes: int = CIFAR10_NUM_CLASSES,
    pretrained: bool = True,
):
    if backbone == "resnet18":
        return get_resnet18(num_classes=num_classes, pretrained=pretrained)
    if backbone == "resnet50":
        return get_resnet50(num_classes=num_classes, pretrained=pretrained)
    if backbone == "effb3":
        return get_efficientnet_b3(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown backbone: {backbone}")


def fc_params(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "fc"):
        return model.fc.weight.detach().cpu(), model.fc.bias.detach().cpu()
    return (
        model.classifier[1].weight.detach().cpu(),
        model.classifier[1].bias.detach().cpu(),
    )


def paths(seed: int, backbone: str) -> dict[str, Path]:
    model_dir = Path(f"data/models/cifar10_svhn/seed{seed}")
    feature_dir = Path(f"outputs/features/cifar10_svhn/seed{seed}")
    report_dir = Path(f"outputs/reports/cifar10_svhn/seed{seed}")
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return {
        "model_dir": model_dir,
        "model_best": model_dir / f"{backbone}_best.pth",
        "features": feature_dir / f"{backbone}_features.pt",
        "score_csv": report_dir / f"{backbone}_score_comparison.csv",
        "dispersion_csv": report_dir / f"{backbone}_dispersion_vs_separability.csv",
        "summary_csv": report_dir / "pilot_summary.csv",
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1), 0.0


def train_backbone(backbone: str, seed: int, batch_size: int, num_workers: int) -> None:
    set_seed(seed)
    device = get_device()
    p = paths(seed, backbone)
    loaders = get_dataloaders(
        backbone,
        batch_size=batch_size,
        num_workers=num_workers,
        include_ood=False,
    )

    model = get_model(backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0.0
    print(f"\n=== Train {backbone} on {device} (seed={seed}) ===")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels, _ in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        scheduler.step()
        train_loss = running_loss / len(loaders["train"].dataset)
        val_acc, _ = evaluate(model, loaders["id_val"], device)
        lr = scheduler.get_last_lr()[0]
        print(
            f"[Epoch {epoch:02d}/{NUM_EPOCHS}] LR={lr:.2e} "
            f"loss={train_loss:.4f} id_test_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), p["model_best"])
            print(f"  saved {p['model_best']} (acc={best_acc:.4f})")


def wait_for_svhn(path: Path = Path("data/raw/svhn/test_32x32.mat"), min_bytes: int = 64_000_000) -> None:
    import time

    print(f"Waiting for SVHN at {path} (need >={min_bytes} bytes)...", flush=True)
    while not path.exists() or path.stat().st_size < min_bytes:
        size = path.stat().st_size if path.exists() else 0
        print(f"  SVHN not ready ({size} bytes), sleep 60s...", flush=True)
        time.sleep(60)
    print(f"SVHN ready: {path.stat().st_size} bytes", flush=True)


def extract_backbone(backbone: str, seed: int, batch_size: int, num_workers: int) -> None:
    device = get_device()
    p = paths(seed, backbone)
    loaders = get_dataloaders(backbone, batch_size=batch_size, num_workers=num_workers)

    model = get_model(backbone, pretrained=False)
    state = torch.load(p["model_best"], map_location=device)
    model.load_state_dict(state)
    model.to(device)

    def loader_labels(loader) -> np.ndarray:
        parts = []
        for _, labels, _ in loader:
            parts.append(labels.numpy())
        return np.concatenate(parts).astype(np.int64)

    print(f"\n=== Extract {backbone} on {device} ===")
    splits = {}
    for name, loader in loaders.items():
        print(f"--- {name} ---")
        logits, feats = extract_features_and_logits(model, loader, device)
        splits[name] = {
            "logits": logits,
            "feats": feats,
            "labels": loader_labels(loader),
        }

    fc_weight, fc_bias = fc_params(model)
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
            "fc_weight": fc_weight,
            "fc_bias": fc_bias,
        },
        p["features"],
    )
    print(f"Saved {p['features']}")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def analyze_backbone(backbone: str, seed: int) -> dict:
    set_seed(seed)
    p = paths(seed, backbone)
    reconstruct = getattr(np.core.multiarray, "_reconstruct", None)
    if reconstruct is not None:
        add_safe_globals([reconstruct])

    try:
        data = torch.load(p["features"], map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(p["features"], map_location="cpu")

    train_logits = to_numpy(data["train_logits"])
    train_feats = to_numpy(data["train_feats"])
    val_logits = to_numpy(data["val_logits"])
    val_feats = to_numpy(data["val_feats"])
    val_labels = to_numpy(data["val_labels"])
    ood_logits = to_numpy(data["ood_logits"])
    ood_feats = to_numpy(data["ood_feats"])
    fc_weight = to_numpy(data["fc_weight"])
    fc_bias = to_numpy(data["fc_bias"])

    scorer = OODScorer(
        k_nearest=50,
        use_react=True,
        react_percentile=90.0,
        use_vim=True,
        vim_dim=None,
    )
    scorer.fit(
        train_feats,
        train_labels=None,
        train_logits=train_logits,
        fc_weight=fc_weight,
        fc_bias=fc_bias,
    )

    scores_id = scorer.get_all_scores(val_logits, val_feats)
    scores_ood = scorer.get_all_scores(ood_logits, ood_feats)
    methods = [m for m in ("msp", "energy", "mahalanobis", "knn") if m in scores_id]

    df_scores = build_score_comparison_df(scores_id, scores_ood, methods, seed=seed)
    df_scores.to_csv(p["score_csv"], index=False)

    df_disp = build_dispersion_vs_separability_table(
        val_logits, ood_logits, val_feats, ood_feats, seed=seed
    )
    df_disp.to_csv(p["dispersion_csv"], index=False)

    maha = df_scores.loc[df_scores["Method"] == "mahalanobis"].iloc[0]
    msp = df_scores.loc[df_scores["Method"] == "msp"].iloc[0]
    ood_row = df_disp.loc[df_disp["split"] == "OOD"].iloc[0]

    summary = {
        "backbone": backbone,
        "seed": seed,
        "mahalanobis_auroc": float(maha["AUROC"]),
        "msp_auroc": float(msp["AUROC"]),
        "auroc_delta_maha_minus_msp": float(maha["AUROC"] - msp["AUROC"]),
        "cvid_gap_vs_id": float(ood_row["cvid_gap_vs_id"]),
        "cvid_gap_vs_id_CI_low": float(ood_row["cvid_gap_vs_id_CI_low"]),
        "cvid_gap_vs_id_CI_high": float(ood_row["cvid_gap_vs_id_CI_high"]),
        "logit_gap_mean_gap_vs_id": float(ood_row["logit_gap_mean_gap_vs_id"]),
        "logit_gap_mean_gap_vs_id_CI_low": float(
            ood_row["logit_gap_mean_gap_vs_id_CI_low"]
        ),
        "logit_gap_mean_gap_vs_id_CI_high": float(
            ood_row["logit_gap_mean_gap_vs_id_CI_high"]
        ),
        "feature_silhouette_cosine": float(ood_row["feature_silhouette_cosine"]),
        "silhouette_perm_p_value": float(ood_row["silhouette_perm_p_value"]),
        "n_id": int(df_disp.loc[df_disp["split"] == "ID", "n_samples"].iloc[0]),
        "n_ood": int(ood_row["n_samples"]),
    }

    print(f"\n=== {backbone} pilot summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return summary


def run_stage(stage: str, backbone: str, seed: int, batch_size: int, num_workers: int):
    names = list(BACKBONES) if backbone == "all" else [backbone]
    summaries = []

    if stage in ("train", "all"):
        for name in names:
            train_backbone(name, seed, batch_size, num_workers)

    if stage in ("extract", "all"):
        wait_for_svhn()
        for name in names:
            extract_backbone(name, seed, batch_size, num_workers)

    if stage in ("analyze", "all"):
        for name in names:
            summaries.append(analyze_backbone(name, seed))

    if summaries:
        report_dir = Path(f"outputs/reports/cifar10_svhn/seed{seed}")
        report_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(report_dir / "pilot_summary.csv", index=False)
        print(f"\nWrote {report_dir / 'pilot_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 vs SVHN pilot benchmark")
    parser.add_argument(
        "--stage",
        choices=("train", "extract", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--backbone", choices=[*BACKBONES, "all"], default="all")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    run_stage(args.stage, args.backbone, args.seed, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
