#!/usr/bin/env python3
"""Camelyon CE + auxiliary SupCon pilot (Job B). R18 and DenseNet121 only.

Does not overwrite CE zoo checkpoints. Stem = {backbone}_supcon.
Khosla 2020: τ=0.07. Auxiliary λ=0.1 (original paper is SupCon-only;
λ is the standard single add-on, not a sweep).
Kill: DenseNet−R18 MSP gap shrink ≥50% (0.31 → ≤0.15) else Direction #1 dies.
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from scripts.camelyon17_pilot import set_seed, to_numpy
from src.datasets.camelyon_ood import (
    CAMELYON_NUM_CLASSES,
    OOD_SPLIT,
    build_transform,
    get_dataloaders,
    get_wilds_dataset,
)
from src.models.cnn_family import fc_params as cnn_fc_params
from src.models.cnn_family import get_cnn_backbone, recipe as cnn_recipe
from src.utils.benchmark_metrics import (
    build_dispersion_vs_separability_table,
    build_score_comparison_df,
)
from src.utils.feature_extractor import extract_features_and_logits
from src.utils.scoring import OODScorer

# Khosla 2020 temperature. Auxiliary weight: one value, no sweep.
SUPCON_TAU = 0.07
SUPCON_LAMBDA = 0.1
NUM_EPOCHS = 10
SEED = 42
BACKBONES = ("resnet18", "densenet121")
CE_MSP = {"resnet18": 0.5743335363500264, "densenet121": 0.8826714444722252}
CE_VAL_ACC_FLOOR_DROP = 0.02

OOD_METHODS = (
    "msp",
    "energy",
    "react_energy",
    "logit_norm",
    "mahalanobis",
    "knn",
    "vim",
)


class TwoCrop(Dataset):
    def __init__(self, subset, tf):
        self.subset = subset
        self.tf = tf

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, label, _meta = self.subset[idx]
        return self.tf(image), self.tf(image), int(label)


def supcon_loss(z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    z = torch.cat([z1, z2], dim=0)
    y = torch.cat([labels, labels], dim=0)
    sim = z @ z.T / tau
    eye = torch.eye(len(y), device=z.device, dtype=torch.bool)
    pos = y.unsqueeze(1).eq(y.unsqueeze(0)) & ~eye
    logits = sim.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    n_pos = pos.sum(1).clamp(min=1)
    return -(log_prob * pos.float()).sum(1).div(n_pos).mean()


def embeddings_and_logits(model: nn.Module, x: torch.Tensor):
    buf = []

    def hook(_m, _i, o):
        feat = o
        if feat.dim() > 2:
            feat = feat.flatten(start_dim=1)
        buf.append(feat)

    h = model.avgpool.register_forward_hook(hook)
    logits = model(x)
    h.remove()
    z = F.normalize(buf[0], dim=1)
    return logits, z


def make_optimizer(model: nn.Module, backbone: str):
    rec = cnn_recipe(backbone)
    if rec["optim"] == "adam":
        return optim.Adam(model.parameters(), lr=rec["lr"], weight_decay=rec["wd"])
    if rec["optim"] == "adamw":
        return optim.AdamW(model.parameters(), lr=rec["lr"], weight_decay=rec["wd"])
    return optim.SGD(
        model.parameters(),
        lr=rec["lr"],
        momentum=float(rec.get("momentum", 0.9)),
        weight_decay=rec["wd"],
    )


@torch.no_grad()
def val_acc(model, loader, device) -> float:
    model.eval()
    correct = 0
    n = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images).argmax(1)
        correct += (pred == labels).sum().item()
        n += labels.size(0)
    return correct / max(n, 1)


def paths(backbone: str) -> dict[str, Path]:
    rel = "frac1/seed42"
    stem = f"{backbone}_supcon"
    model_dir = Path(f"data/models/camelyon17/{rel}")
    feat_dir = Path(f"outputs/features/camelyon17/{rel}")
    report_dir = Path(f"outputs/reports/camelyon17/{rel}")
    for d in (model_dir, feat_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "stem": stem,
        "model_best": model_dir / f"{stem}_best.pth",
        "ce_model": model_dir / f"{backbone}_best.pth",
        "features": feat_dir / f"{stem}_features.pt",
        "score_csv": report_dir / f"{stem}_score_comparison.csv",
        "dispersion_csv": report_dir / f"{stem}_dispersion_vs_separability.csv",
    }


def train_one(backbone: str, num_workers: int, device: torch.device) -> float:
    p = paths(backbone)
    rec = cnn_recipe(backbone)
    bs = int(rec["batch_size"])
    tf = build_transform(backbone, train=True)
    raw = get_wilds_dataset(download=False)
    train_subset = raw.get_subset("train", transform=None)
    train_loader = DataLoader(
        TwoCrop(train_subset, tf),
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = get_dataloaders(
        backbone, batch_size=bs, num_workers=num_workers, train_frac=1.0, seed=SEED
    )["id_val"]

    ce_acc = None
    if p["ce_model"].exists():
        ce_model = get_cnn_backbone(backbone, CAMELYON_NUM_CLASSES, pretrained=False).to(device)
        ce_model.load_state_dict(torch.load(p["ce_model"], map_location=device))
        ce_acc = val_acc(ce_model, val_loader, device)
        print(f"CE baseline {backbone} id_val_acc={ce_acc:.4f}", flush=True)
        del ce_model
        torch.cuda.empty_cache()

    if p["model_best"].exists():
        model = get_cnn_backbone(backbone, CAMELYON_NUM_CLASSES, pretrained=False).to(device)
        model.load_state_dict(torch.load(p["model_best"], map_location=device))
        best = val_acc(model, val_loader, device)
        print(f"  skip train, checkpoint exists: {p['model_best']} acc={best:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
        return best

    model = get_cnn_backbone(backbone, CAMELYON_NUM_CLASSES, pretrained=True).to(device)
    opt = make_optimizer(model, backbone)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    ce = nn.CrossEntropyLoss()
    best = 0.0
    print(
        f"=== Train {backbone} CE+SupCon λ={SUPCON_LAMBDA} τ={SUPCON_TAU} "
        f"bs={bs} ===",
        flush=True,
    )
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running = 0.0
        n = 0
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad()
            logits1, z1 = embeddings_and_logits(model, x1)
            _logits2, z2 = embeddings_and_logits(model, x2)
            loss = ce(logits1, y) + SUPCON_LAMBDA * supcon_loss(z1, z2, y, SUPCON_TAU)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()
        acc = val_acc(model, val_loader, device)
        print(
            f"[Epoch {epoch:02d}/{NUM_EPOCHS}] loss={running / max(n, 1):.4f} "
            f"id_val_acc={acc:.4f}",
            flush=True,
        )
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), p["model_best"])
            print(f"  saved {p['model_best']} (acc={best:.4f})", flush=True)
    if ce_acc is not None and (ce_acc - best) > CE_VAL_ACC_FLOOR_DROP:
        print(
            f"WARNING {backbone}: SupCon val acc {best:.4f} dropped "
            f">{CE_VAL_ACC_FLOOR_DROP:.0%} vs CE {ce_acc:.4f}. "
            "Do not interpret OOD until this is addressed.",
            flush=True,
        )
    del model
    torch.cuda.empty_cache()
    return best


def extract_one(backbone: str, num_workers: int, device: torch.device) -> None:
    p = paths(backbone)
    if p["features"].exists():
        print(f"  skip extract {p['features']}", flush=True)
        return
    rec = cnn_recipe(backbone)
    bs = int(rec["batch_size"])
    loaders = get_dataloaders(
        backbone, batch_size=bs, num_workers=num_workers, train_frac=1.0, seed=SEED
    )
    model = get_cnn_backbone(backbone, CAMELYON_NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(p["model_best"], map_location=device))
    model.to(device)

    def loader_labels(loader):
        parts = []
        for _, labels, _ in loader:
            parts.append(labels.numpy())
        return np.concatenate(parts).astype(np.int64)

    splits = {}
    print(f"=== Extract {p['stem']} ===", flush=True)
    for name, loader in loaders.items():
        print(f"--- {name} ({len(loader.dataset)}) ---", flush=True)
        logits, feats = extract_features_and_logits(model, loader, device)
        splits[name] = {
            "logits": logits,
            "feats": feats,
            "labels": loader_labels(loader),
        }
    fc_w, fc_b = cnn_fc_params(model)
    feat_dim = int(splits["train"]["feats"].shape[1])
    if int(fc_w.shape[1]) != feat_dim:
        raise ValueError(
            f"{backbone}: fc_weight in_features={fc_w.shape[1]} "
            f"!= avgpool feat_dim={feat_dim}"
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
            "fc_weight": fc_w,
            "fc_bias": fc_b,
            "ood_split": OOD_SPLIT,
            "recipe": "ce_supcon",
            "lambda": SUPCON_LAMBDA,
            "tau": SUPCON_TAU,
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
    print(f"{p['stem']}: MSP={msp:.4f} Maha={maha:.4f} (CE MSP={CE_MSP[backbone]:.4f})")
    return {"backbone": backbone, "msp_supcon": msp, "maha_supcon": maha, "msp_ce": CE_MSP[backbone]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"CUDA {device} {gpu} λ={SUPCON_LAMBDA} τ={SUPCON_TAU}", flush=True)
    rows = []
    for bb in BACKBONES:
        train_one(bb, args.num_workers, device)
        extract_one(bb, args.num_workers, device)
        rows.append(analyze_one(bb))
    df = pd.DataFrame(rows)
    r18 = float(df.loc[df.backbone == "resnet18", "msp_supcon"].iloc[0])
    den = float(df.loc[df.backbone == "densenet121", "msp_supcon"].iloc[0])
    gap_ce = CE_MSP["densenet121"] - CE_MSP["resnet18"]
    gap = den - r18
    target = 0.5 * gap_ce
    ok = gap <= target
    verdict = (
        f"CE gap Dense−R18={gap_ce:.3f}. SupCon gap={gap:.3f} "
        f"(shrink {100 * (1 - gap / gap_ce):.0f}%, target ≥50% i.e. gap≤{target:.3f}). "
    )
    if ok:
        verdict += "PROMISING — scale full 8."
    else:
        verdict += "Direction #1 DEAD. Write-only. No center-loss/triplet."
    out = Path("outputs/reports/camelyon_supcon_pilot.txt")
    out.write_text(verdict + "\n" + df.to_csv(index=False))
    print(verdict)
    print(f"Wrote {out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
