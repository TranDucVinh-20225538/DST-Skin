#!/usr/bin/env python3
"""Skin ISIC→PAD: 5 new CNN backbones (seed=42, 10 epochs, per-arch recipes).

Does not retrain R18/R50/EffB3. Writes under outputs/{features,reports}/skin/
so original root CSVs stay untouched. Launch only after Camelyon n=8 is in.
"""

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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.isic_dataset import ISICDataset
from src.models.cnn_family import (
    NEW_CNN_BACKBONES,
    fc_params as cnn_fc_params,
    get_cnn_backbone,
    input_size as default_input_size,
    recipe as cnn_recipe,
)
from src.utils.benchmark_metrics import (
    build_dispersion_vs_separability_table,
    build_score_comparison_df,
)
from src.utils.feature_extractor import extract_features_and_logits
from src.utils.scoring import OODScorer

NUM_EPOCHS = 10
SEED_DEFAULT = 42
OOD_METHODS = (
    "msp",
    "energy",
    "react_energy",
    "logit_norm",
    "mahalanobis",
    "knn",
    "vim",
)

TRAIN_CSV = "data/processed/isic2018_binary/isic_2018_binary_train.csv"
TRAIN_ROOT = "data/raw/isic2018/ISIC2018_Task3_Training_Input"
VAL_CSV = "data/processed/isic2018_binary/isic_2018_binary_val.csv"
VAL_ROOT = "data/raw/isic2018/ISIC2018_Task3_Validation_Input"
OOD_CSV = "data/processed/pad_ufes20_binary/pad_ufes_binary.csv"
OOD_ROOT = "data/raw/pad_ufes20/images"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(model: nn.Module, backbone: str):
    rec = cnn_recipe(backbone)
    if rec["optim"] == "adam":
        return optim.Adam(model.parameters(), lr=rec["lr"], weight_decay=rec["wd"])
    if rec["optim"] == "adamw":
        return optim.AdamW(model.parameters(), lr=rec["lr"], weight_decay=rec["wd"])
    if rec["optim"] == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=rec["lr"],
            momentum=float(rec.get("momentum", 0.9)),
            weight_decay=rec["wd"],
        )
    raise ValueError(rec["optim"])


def artifact_stem(backbone: str, input_size: int | None) -> str:
    if input_size is None:
        return backbone
    native = default_input_size(backbone)
    if int(input_size) == int(native):
        return backbone
    return f"{backbone}_{int(input_size)}"


def size_used(backbone: str, input_size: int | None) -> int:
    return int(input_size) if input_size is not None else default_input_size(backbone)


def build_train_transform(backbone: str, input_size: int | None = None):
    size = size_used(backbone, input_size)
    resize = int(round(size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(backbone: str, input_size: int | None = None):
    size = size_used(backbone, input_size)
    resize = 256 if size == 224 else size
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def paths(backbone: str, input_size: int | None = None) -> dict[str, Path]:
    model_dir = Path("data/models/skin")
    feat_dir = Path("outputs/features/skin")
    report_dir = Path("outputs/reports/skin")
    for d in (model_dir, feat_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(backbone, input_size)
    return {
        "model_best": model_dir / f"{stem}_best.pth",
        "features": feat_dir / f"{stem}_isic_pad_features.pt",
        "score_csv": report_dir / f"{stem}_score_comparison.csv",
        "dispersion_csv": report_dir / f"{stem}_dispersion_vs_separability.csv",
        "stem": stem,
    }


def get_id_loaders(backbone: str, batch_size: int, num_workers: int, input_size: int | None = None):
    train_ds = ISICDataset(
        TRAIN_CSV, TRAIN_ROOT, transform=build_train_transform(backbone, input_size)
    )
    val_ds = ISICDataset(
        VAL_CSV, VAL_ROOT, transform=build_eval_transform(backbone, input_size)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def get_extract_loaders(backbone: str, batch_size: int, num_workers: int, input_size: int | None = None):
    tf = build_eval_transform(backbone, input_size)
    specs = {
        "train": (TRAIN_CSV, TRAIN_ROOT),
        "id_val": (VAL_CSV, VAL_ROOT),
        "ood": (OOD_CSV, OOD_ROOT),
    }
    out = {}
    for name, (csv_path, img_root) in specs.items():
        ds = ISICDataset(csv_path, img_root, transform=tf)
        out[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == labels).mean())
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = 0.0
    return acc, auc


def train_backbone(backbone: str, seed: int, num_workers: int, input_size: int | None = None) -> None:
    set_seed(seed)
    p = paths(backbone, input_size)
    if p["model_best"].exists():
        print(f"  skip train, checkpoint exists: {p['model_best']}", flush=True)
        return
    rec = cnn_recipe(backbone)
    bs = int(rec["batch_size"])
    device = get_device()
    train_loader, val_loader = get_id_loaders(backbone, bs, num_workers, input_size)
    model = get_cnn_backbone(backbone, num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, backbone)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    best_auc = 0.0
    sample, _, _ = next(iter(train_loader))
    print(
        f"\n=== Train {backbone} skin (seed={seed}, "
        f"input_size={size_used(backbone, input_size)}, batch_chw={tuple(sample.shape)}, "
        f"optim={rec['optim']} lr={rec['lr']} wd={rec['wd']} bs={bs}) ===",
        flush=True,
    )
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running = 0.0
        n = 0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            n += labels.size(0)
        scheduler.step()
        acc, auc = evaluate(model, val_loader, device)
        print(
            f"[Epoch {epoch:02d}/{NUM_EPOCHS}] loss={running / max(n, 1):.4f} "
            f"val_acc={acc:.4f} val_auc={auc:.4f}",
            flush=True,
        )
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), p["model_best"])
            print(f"  saved {p['model_best']} (auc={best_auc:.4f})", flush=True)


def extract_backbone(backbone: str, num_workers: int, input_size: int | None = None) -> None:
    p = paths(backbone, input_size)
    if p["features"].exists():
        print(f"  skip extract, features exist: {p['features']}", flush=True)
        return
    rec = cnn_recipe(backbone)
    bs = int(rec["batch_size"])
    device = get_device()
    loaders = get_extract_loaders(backbone, bs, num_workers, input_size)
    model = get_cnn_backbone(backbone, num_classes=2, pretrained=False)
    state = torch.load(p["model_best"], map_location=device)
    model.load_state_dict(state)
    model.to(device)

    def loader_labels(loader) -> np.ndarray:
        parts = []
        for _, labels, _ in loader:
            parts.append(labels.numpy())
        return np.concatenate(parts).astype(np.int64)

    splits = {}
    for name, loader in loaders.items():
        print(f"--- extract {backbone} {name} ({len(loader.dataset)}) ---", flush=True)
        logits, feats = extract_features_and_logits(model, loader, device)
        splits[name] = {
            "logits": logits,
            "feats": feats,
            "labels": loader_labels(loader),
        }
    fc_weight, fc_bias = cnn_fc_params(model)
    feat_dim = int(splits["train"]["feats"].shape[1])
    if int(fc_weight.shape[1]) != feat_dim:
        raise ValueError(
            f"{backbone}: fc in_features={fc_weight.shape[1]} != feat_dim={feat_dim}"
        )
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
    print(f"Saved {p['features']}", flush=True)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def analyze_backbone(backbone: str, seed: int, input_size: int | None = None) -> dict:
    set_seed(seed)
    p = paths(backbone, input_size)
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
    df_scores = build_score_comparison_df(scores_id, scores_ood, methods, seed=seed)
    df_scores.to_csv(p["score_csv"], index=False)
    df_disp = build_dispersion_vs_separability_table(
        to_numpy(data["val_logits"]),
        to_numpy(data["ood_logits"]),
        to_numpy(data["val_feats"]),
        to_numpy(data["ood_feats"]),
        seed=seed,
    )
    df_disp.to_csv(p["dispersion_csv"], index=False)
    maha = df_scores.loc[df_scores["Method"] == "mahalanobis"].iloc[0]
    msp = df_scores.loc[df_scores["Method"] == "msp"].iloc[0]
    return {
        "backbone": backbone,
        "mahalanobis_auroc": float(maha["AUROC"]),
        "msp_auroc": float(msp["AUROC"]),
        "auroc_delta_maha_minus_msp": float(maha["AUROC"] - msp["AUROC"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "extract", "analyze", "all"), default="all")
    parser.add_argument("--backbone", default="new")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Override native input size; tags artifacts (efficientnet_v2_s_224).",
    )
    args = parser.parse_args()
    if args.backbone == "new":
        names = list(NEW_CNN_BACKBONES)
    else:
        names = [args.backbone]
        if names[0] not in NEW_CNN_BACKBONES:
            raise ValueError(names[0])
    summaries = []
    if args.stage in ("train", "all"):
        for name in names:
            train_backbone(name, args.seed, args.num_workers, args.input_size)
    if args.stage in ("extract", "all"):
        for name in names:
            extract_backbone(name, args.num_workers, args.input_size)
    if args.stage in ("analyze", "all"):
        for name in names:
            summaries.append(analyze_backbone(name, args.seed, args.input_size))
        Path("outputs/reports/skin").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv("outputs/reports/skin/pilot_summary.csv", index=False)
        if args.input_size is None:
            import runpy

            runpy.run_path(
                str(Path(__file__).with_name("rebuild_architecture_invariance.py")),
                run_name="__main__",
            )
        else:
            print(
                f"skip invariance rebuild (input_size={args.input_size} override; "
                "do not mix with default-resolution zoo table)",
                flush=True,
            )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
