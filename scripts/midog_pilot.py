#!/usr/bin/env python3
"""MIDOG covariate-shift pilot.

ResNet-50: OpenMIBOOD checkpoint (no retrain).
ResNet-18 / EffB3: train on domain 1a via the same imglist adapter, then extract+gap.
ID eval = test 1a; OOD = cs-ID 1b+1c.
"""

from __future__ import annotations

import argparse
import random
import time
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

from src.datasets.midog_ood import (
    CKPT_PATH,
    DATA_ROOT,
    MIDOG_NUM_CLASSES,
    _input_size,
    data_ready,
    download_checkpoint,
    download_id_csid,
    get_dataloaders,
    split_summary,
)
from src.models.cnn_family import (
    NEW_CNN_BACKBONES,
    ORIGINAL_BACKBONES,
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

TRAIN_BACKBONES = ("resnet18", "resnet50", "effb3")
ALL_BACKBONES = (*ORIGINAL_BACKBONES, *NEW_CNN_BACKBONES)
OOD_METHODS = (
    "msp",
    "energy",
    "react_energy",
    "logit_norm",
    "mahalanobis",
    "knn",
    "vim",
)
SEED_DEFAULT = 42
NUM_EPOCHS = 10


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


def get_model(backbone: str, pretrained: bool = True) -> nn.Module:
    return get_cnn_backbone(backbone, num_classes=MIDOG_NUM_CLASSES, pretrained=pretrained)


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
    """Tag non-native CNN size. EffV2-S@224 → efficientnet_v2_s_224 (official W)."""
    size = int(input_size) if input_size is not None else _input_size(backbone)
    native = default_input_size(backbone)
    if int(size) == int(native):
        return backbone
    return f"{backbone}_{int(size)}"


def paths(seed: int, backbone: str, input_size: int | None = None) -> dict[str, Path]:
    stem = artifact_stem(backbone, input_size)
    model_dir = Path(f"data/models/midog/seed{seed}")
    feature_dir = Path(f"outputs/features/midog/seed{seed}")
    report_dir = Path(f"outputs/reports/midog/seed{seed}")
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return {
        "stem": stem,
        "openmibood_ckpt": Path(CKPT_PATH),
        "model_best": model_dir / f"{stem}_best.pth",
        "features": feature_dir / f"{stem}_features.pt",
        "score_csv": report_dir / f"{stem}_score_comparison.csv",
        "dispersion_csv": report_dir / f"{stem}_dispersion_vs_separability.csv",
        "summary_csv": report_dir / "pilot_summary.csv",
    }


def fc_params(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    return cnn_fc_params(model)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def wait_for_midog(marker: Path | None = None) -> None:
    marker = marker or data_ready()
    print(f"Waiting for MIDOG ID+cs-ID at {marker}...", flush=True)
    while not marker.exists():
        print("  dataset not ready, sleep 60s...", flush=True)
        time.sleep(60)
    print(f"MIDOG ready: {marker}", flush=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def loader_labels(loader) -> np.ndarray:
    parts = []
    for _, labels, _ in loader:
        parts.append(labels.numpy())
    return np.concatenate(parts).astype(np.int64)


def load_openmibood_resnet50(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = get_resnet50(num_classes=MIDOG_NUM_CLASSES, pretrained=False)
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def train_backbone(
    backbone: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    input_size: int | None = None,
) -> None:
    p = paths(seed, backbone, input_size=input_size)
    if p["model_best"].exists():
        print(f"  skip train, checkpoint exists: {p['model_best']}", flush=True)
        return
    set_seed(seed)
    device = get_device()
    rec = cnn_recipe(backbone)
    bs = int(rec["batch_size"]) if rec.get("batch_size") else batch_size
    loaders = get_dataloaders(
        batch_size=bs,
        num_workers=num_workers,
        include_ood=False,
        data_root=DATA_ROOT,
        backbone=backbone,
        input_size=input_size,
    )
    model = get_model(backbone, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, backbone)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    best_acc = 0.0
    n_train = len(loaders["train"].dataset)
    sample, _, _ = next(iter(loaders["train"]))
    size_used = _input_size(backbone, input_size=input_size)
    print(
        f"\n=== Train {backbone} on {device} "
        f"(seed={seed}, n_train={n_train}, num_classes={MIDOG_NUM_CLASSES}, "
        f"stem={p['stem']}, input_size={size_used}, batch_chw={tuple(sample.shape)}, "
        f"optim={rec['optim']} lr={rec['lr']} wd={rec['wd']} bs={bs}) ===",
        flush=True,
    )
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
        val_acc = evaluate(model, loaders["id_val"], device)
        lr = scheduler.get_last_lr()[0]
        train_loss = running_loss / max(n_train, 1)
        print(
            f"[Epoch {epoch:02d}/{NUM_EPOCHS}] LR={lr:.2e} "
            f"loss={train_loss:.4f} id_val_acc={val_acc:.4f}",
            flush=True,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), p["model_best"])
            print(f"  saved {p['model_best']} (acc={best_acc:.4f})", flush=True)


def extract_backbone(
    backbone: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    input_size: int | None = None,
) -> None:
    device = get_device()
    p = paths(seed, backbone, input_size=input_size)
    if p["features"].exists():
        print(f"  skip extract, features exist: {p['features']}", flush=True)
        return
    inhouse = p["model_best"].exists()
    rec = cnn_recipe(backbone)
    bs = int(rec["batch_size"]) if rec.get("batch_size") else batch_size
    dl_backbone = backbone if (backbone != "resnet50" or inhouse) else None
    loaders = get_dataloaders(
        batch_size=bs,
        num_workers=num_workers,
        include_ood=True,
        data_root=DATA_ROOT,
        backbone=dl_backbone,
        input_size=input_size,
    )
    if backbone == "resnet50" and not inhouse:
        model = load_openmibood_resnet50(p["openmibood_ckpt"], device)
        source = "openmibood_ckpt"
    else:
        model = get_model(backbone, pretrained=False)
        state = torch.load(p["model_best"], map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        source = "trained_imagenet_init"

    print(
        f"\n=== Extract {p['stem']} (num_classes={MIDOG_NUM_CLASSES}) on {device} ===",
        flush=True,
    )
    splits = {}
    for name, loader in loaders.items():
        print(f"--- {name} ({len(loader.dataset)} samples) ---", flush=True)
        logits, feats = extract_features_and_logits(model, loader, device)
        splits[name] = {
            "logits": logits,
            "feats": feats,
            "labels": loader_labels(loader),
        }

    weight, bias = cnn_fc_params(model)
    feat_dim = int(splits["train"]["feats"].shape[1])
    if int(weight.shape[1]) != feat_dim:
        raise ValueError(
            f"{backbone}: fc_weight in_features={weight.shape[1]} "
            f"!= avgpool feat_dim={feat_dim}"
        )
    torch.save(
        {
            "train_logits": splits["train"]["logits"],
            "train_feats": splits["train"]["feats"],
            "train_labels": splits["train"]["labels"],
            "val_logits": splits["id_test"]["logits"],
            "val_feats": splits["id_test"]["feats"],
            "val_labels": splits["id_test"]["labels"],
            "ood_logits": splits["ood"]["logits"],
            "ood_feats": splits["ood"]["feats"],
            "ood_labels": splits["ood"]["labels"],
            "fc_weight": weight,
            "fc_bias": bias,
            "ood_split": "csid_1b+1c",
            "id_split": "test_1a",
            "backbone": p["stem"],
            "arch": backbone,
            "source": source,
            "input_size": _input_size(backbone, input_size=input_size),
        },
        p["features"],
    )
    print(f"Saved {p['features']}", flush=True)


def analyze_backbone(
    backbone: str,
    seed: int,
    input_size: int | None = None,
) -> dict:
    set_seed(seed)
    p = paths(seed, backbone, input_size=input_size)
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
    methods = [m for m in OOD_METHODS if m in scores_id]

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
        "backbone": p["stem"],
        "arch": backbone,
        "seed": seed,
        "id_split": str(data.get("id_split", "test_1a")),
        "ood_split": str(data.get("ood_split", "csid_1b+1c")),
        "source": str(data.get("source", "openmibood_ckpt" if backbone == "resnet50" else "trained_imagenet_init")),
        "mahalanobis_auroc": float(maha["AUROC"]),
        "msp_auroc": float(msp["AUROC"]),
        "auroc_delta_maha_minus_msp": float(maha["AUROC"] - msp["AUROC"]),
        "cvid_gap_vs_id": float(ood_row["cvid_gap_vs_id"]),
        "cvid_gap_vs_id_CI_low": float(ood_row["cvid_gap_vs_id_CI_low"]),
        "cvid_gap_vs_id_CI_high": float(ood_row["cvid_gap_vs_id_CI_high"]),
        "logit_gap_mean_gap_vs_id": float(ood_row["logit_gap_mean_gap_vs_id"]),
        "logit_gap_mean_gap_vs_id_CI_low": float(ood_row["logit_gap_mean_gap_vs_id_CI_low"]),
        "logit_gap_mean_gap_vs_id_CI_high": float(ood_row["logit_gap_mean_gap_vs_id_CI_high"]),
        "feature_silhouette_cosine": float(ood_row["feature_silhouette_cosine"]),
        "silhouette_perm_p_value": float(ood_row["silhouette_perm_p_value"]),
        "n_id": int(df_disp.loc[df_disp["split"] == "ID", "n_samples"].iloc[0]),
        "n_ood": int(ood_row["n_samples"]),
        "logit_gap_estimator": "trim_1pct",
        "cvid_status": "appendix",
        "input_size": int(data.get("input_size", _input_size(backbone, input_size=input_size))),
    }
    print(f"\n=== {p['stem']} MIDOG gap summary (ID=test 1a, OOD=1b+1c) ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return summary


def merge_summaries(seed: int, rows: list[dict]) -> None:
    p = paths(seed, "resnet18")["summary_csv"]
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        keep = old[~old["backbone"].isin(new["backbone"])]
        out = pd.concat([keep, new], ignore_index=True)
    else:
        out = new
    order = {b: i for i, b in enumerate(ALL_BACKBONES)}
    out["_ord"] = out["backbone"].map(lambda b: order.get(b, 99))
    out = out.sort_values("_ord").drop(columns="_ord")
    out.to_csv(p, index=False)
    print(f"Wrote {p}")


def parse_backbones(spec: str) -> list[str]:
    if spec == "all":
        return list(ALL_BACKBONES)
    if spec == "train":
        return list(TRAIN_BACKBONES)
    if spec == "new":
        return list(NEW_CNN_BACKBONES)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    for n in names:
        if n not in ALL_BACKBONES:
            raise ValueError(f"Unknown backbone {n}")
    return names


def run_probe() -> None:
    summary = split_summary()
    print("\n=== MIDOG split summary (OpenMIBOOD imglists) ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    report_dir = Path("outputs/reports/midog")
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(report_dir / "split_summary.csv", index=False)
    print(f"Wrote {report_dir / 'split_summary.csv'}")


def rebuild_invariance() -> None:
    try:
        import runpy

        runpy.run_path(
            str(Path(__file__).with_name("rebuild_architecture_invariance.py")),
            run_name="__main__",
        )
    except Exception as exc:
        print(f"invariance rebuild skipped: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIDOG covariate-shift: in-house zoo. Do not mix OpenMIBOOD R50."
    )
    parser.add_argument(
        "--stage",
        choices=("probe", "download", "train", "extract", "analyze", "all"),
        default="all",
    )
    parser.add_argument(
        "--backbone",
        default="train",
        help="name, comma-list, train (R18/R50/EffB3), new (5 CNN), or all",
    )
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--keep-wsi",
        action="store_true",
        help="Keep raw ~20GB TIFFs after cropping (default: delete)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild Kendall-W table after this stage (only when official cells are ready).",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    names = parse_backbones(args.backbone)

    if args.stage == "probe":
        run_probe()
        return

    if args.stage == "download":
        download_checkpoint()
        print("Downloading MIDOG domains 1a/1b/1c only (not full 65GB)...", flush=True)
        download_id_csid(keep_wsi=args.keep_wsi)
        run_probe()
        return

    wait_for_midog()
    summaries = []
    for name in names:
        if args.stage in ("train", "all"):
            train_backbone(name, args.seed, args.batch_size, args.num_workers)
        if args.stage in ("extract", "all"):
            extract_backbone(name, args.seed, args.batch_size, args.num_workers)
        if args.stage in ("analyze", "all"):
            summaries.append(analyze_backbone(name, args.seed))
    if summaries:
        merge_summaries(args.seed, summaries)
    if args.rebuild:
        rebuild_invariance()


if __name__ == "__main__":
    main()
