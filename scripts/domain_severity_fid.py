#!/usr/bin/env python3
"""Task-agnostic domain severity: FID on frozen ImageNet ResNet-50 avgpool.

Independent of the evaluated classifier. CPU-friendly with a sample cap.
Does not touch Camelyon 59482 or CIFAR-10-C jobs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import linalg
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

from src.datasets.isic_dataset import ISICDataset
from src.datasets.midog_ood import (
    DATA_ROOT as MIDOG_ROOT,
    ImglistTupleDataset,
    imglist_path,
)
from src.datasets.camelyon_ood import get_dataloaders as get_camelyon_loaders

SEED = 42
MAX_PER_SPLIT = 2000
BATCH = 32
NUM_WORKERS = 2
OUT = Path("outputs/reports/domain_severity_fid.csv")


def imagenet_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def frozen_resnet50(device: torch.device) -> nn.Module:
    # Prefer cached IMAGENET1K_V1 (resnet50-0676ba61.pth) — do not stall on V2 download.
    try:
        model = models.resnet50(pretrained=True)
    except Exception:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


def cap_loader(dataset, n: int, batch: int, seed: int = SEED) -> DataLoader:
    n = min(n, len(dataset))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=n, replace=False))
    return DataLoader(
        Subset(dataset, idx.tolist()),
        batch_size=batch,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )


@torch.no_grad()
def extract(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    chunks = []
    for batch in tqdm(loader, desc="FID extract", leave=False):
        images = batch[0].to(device)
        feat = model(images)
        if feat.dim() > 2:
            feat = feat.flatten(start_dim=1)
        chunks.append(feat.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def fid_score(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    mu1, mu2 = x.mean(axis=0), y.mean(axis=0)
    s1 = np.cov(x, rowvar=False)
    s2 = np.cov(y, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


def skin_loaders(tf):
    id_ds = ISICDataset(
        "data/processed/isic2018_binary/isic_2018_binary_val.csv",
        "data/raw/isic2018/ISIC2018_Task3_Validation_Input",
        transform=tf,
    )
    ood_ds = ISICDataset(
        "data/processed/pad_ufes20_binary/pad_ufes_binary.csv",
        "data/raw/pad_ufes20/images",
        transform=tf,
    )
    return cap_loader(id_ds, MAX_PER_SPLIT, BATCH), cap_loader(ood_ds, MAX_PER_SPLIT, BATCH)


def midog_loaders(tf):
    id_ds = ImglistTupleDataset(imglist_path("id_test"), MIDOG_ROOT, tf)
    from torch.utils.data import ConcatDataset

    ood_ds = ConcatDataset([
        ImglistTupleDataset(imglist_path("csid_1b"), MIDOG_ROOT, tf),
        ImglistTupleDataset(imglist_path("csid_1c"), MIDOG_ROOT, tf),
    ])
    return cap_loader(id_ds, MAX_PER_SPLIT, BATCH), cap_loader(ood_ds, MAX_PER_SPLIT, BATCH)


def camelyon_loaders():
    # ImageNet 224 eval transform already (same stats as frozen R50).
    loaders = get_camelyon_loaders(
        "resnet50",
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        include_ood=True,
        download=False,
        train_frac=0.05,
        seed=SEED,
    )
    id_ds = loaders["id_val"].dataset
    ood_ds = loaders["ood"].dataset
    return cap_loader(id_ds, MAX_PER_SPLIT, BATCH), cap_loader(ood_ds, MAX_PER_SPLIT, BATCH)


def main() -> None:
    device = torch.device("cpu")
    print(f"FID encoder: frozen ImageNet ResNet-50 avgpool on {device}", flush=True)
    model = frozen_resnet50(device)
    tf = imagenet_transform()

    jobs = [
        ("skin_isic_pad", lambda: skin_loaders(tf)),
        ("camelyon17", camelyon_loaders),
        ("midog", lambda: midog_loaders(tf)),
    ]
    rows = []
    for domain, factory in jobs:
        print(f"\n=== {domain} ===", flush=True)
        id_loader, ood_loader = factory()
        id_feat = extract(model, id_loader, device)
        ood_feat = extract(model, ood_loader, device)
        score = fid_score(id_feat, ood_feat)
        rows.append({
            "domain": domain,
            "fid_score": score,
            "encoder": "imagenet_resnet50_avgpool_v1",
            "n_id": int(len(id_feat)),
            "n_ood": int(len(ood_feat)),
        })
        print(f"  FID={score:.3f}  n_id={len(id_feat)} n_ood={len(ood_feat)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
