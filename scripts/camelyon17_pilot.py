#!/usr/bin/env python3
"""Pilot: WILDS Camelyon17 covariate shift — train on ID hospitals, OOD = hospital 2."""

from __future__ import annotations

import argparse
import json
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

from src.datasets.camelyon_ood import (
    CAMELYON_NUM_CLASSES,
    OOD_SPLIT,
    data_ready,
    get_dataloaders,
    get_wilds_dataset,
    split_summary,
)
from src.models.cnn_family import (
    NEW_CNN_BACKBONES,
    ORIGINAL_BACKBONES,
    VIT_BACKBONES,
    fc_params as cnn_fc_params,
    get_cnn_backbone,
    input_size as default_input_size,
    recipe as cnn_recipe,
)
from src.utils.benchmark_metrics import (
    build_dispersion_vs_separability_table,
    build_score_comparison_df,
    calc_auroc,
)
from src.utils.feature_extractor import extract_features_and_logits
from src.utils.scoring import OODScorer

BACKBONES = ORIGINAL_BACKBONES
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


def get_model(
    backbone: str,
    num_classes: int = CAMELYON_NUM_CLASSES,
    pretrained: bool = True,
):
    return get_cnn_backbone(backbone, num_classes=num_classes, pretrained=pretrained)


def resolve_names(backbone: str) -> list[str]:
    if backbone == "all":
        return list(BACKBONES)
    if backbone == "new":
        return list(NEW_CNN_BACKBONES)
    return [backbone]


def resolve_recipe(backbone: str, override: dict | None = None) -> dict:
    rec = cnn_recipe(backbone)
    if override:
        rec = {**rec, **{k: v for k, v in override.items() if v is not None}}
    return rec


def make_optimizer(model: nn.Module, backbone: str, rec: dict | None = None):
    rec = rec if rec is not None else cnn_recipe(backbone)
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


def run_tag(seed: int, train_frac: float, artifact_tag: str | None = None) -> str:
    """Keep 5% pilot at seed{N}; full/other fracs go under fracX/seed{N} so they never clobber.

    `artifact_tag` inserts a directory (e.g. matched_adam) so recipe ablations
    cannot overwrite official seed-42 checkpoints or score CSVs.
    """
    tag = (artifact_tag or "").strip() or None
    if abs(float(train_frac) - 0.05) < 1e-12:
        return f"{tag}/seed{seed}" if tag else f"seed{seed}"
    frac_tag = f"{float(train_frac):g}".replace(".", "p")
    if tag:
        return f"frac{frac_tag}/{tag}/seed{seed}"
    return f"frac{frac_tag}/seed{seed}"


def artifact_stem(backbone: str, input_size: int | None) -> str:
    """Keep default-resolution files untagged (effb3@300 stays effb3_best.pth)."""
    if input_size is None:
        return backbone
    native = default_input_size(backbone)
    if int(input_size) == int(native):
        return backbone
    return f"{backbone}_{int(input_size)}"


def paths(
    seed: int,
    backbone: str,
    train_frac: float = 0.05,
    input_size: int | None = None,
    artifact_tag: str | None = None,
) -> dict[str, Path]:
    rel = run_tag(seed, train_frac, artifact_tag=artifact_tag)
    stem = artifact_stem(backbone, input_size)
    model_dir = Path(f"data/models/camelyon17/{rel}")
    feature_dir = Path(f"outputs/features/camelyon17/{rel}")
    report_dir = Path(f"outputs/reports/camelyon17/{rel}")
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return {
        "model_dir": model_dir,
        "model_best": model_dir / f"{stem}_best.pth",
        "features": feature_dir / f"{stem}_features.pt",
        "score_csv": report_dir / f"{stem}_score_comparison.csv",
        "dispersion_csv": report_dir / f"{stem}_dispersion_vs_separability.csv",
        "summary_csv": report_dir / "pilot_summary.csv",
        "epoch_msp_csv": report_dir / f"{stem}_epoch_msp.csv",
        "stem": stem,
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


@torch.no_grad()
def collect_logits(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    for images, _, _ in loader:
        parts.append(model(images.to(device)).detach().cpu().numpy())
    return np.concatenate(parts, axis=0)


def msp_auroc_id_vs_ood(model: nn.Module, id_loader, ood_loader, device: torch.device) -> float:
    s_id = OODScorer.score_msp(collect_logits(model, id_loader, device))
    s_ood = OODScorer.score_msp(collect_logits(model, ood_loader, device))
    return calc_auroc(s_id, s_ood)


def wait_for_camelyon(marker: Path | None = None) -> None:
    marker = marker or data_ready()
    print(f"Waiting for Camelyon17 at {marker}...", flush=True)
    while not marker.exists():
        print("  dataset not ready, sleep 120s...", flush=True)
        time.sleep(120)
    print(f"Camelyon17 ready: {marker}", flush=True)


def train_backbone(
    backbone: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    train_frac: float,
    download: bool,
    input_size: int | None = None,
    log_msp_epoch: bool = False,
    artifact_tag: str | None = None,
    recipe_override: dict | None = None,
    stain_cov: bool = False,
) -> None:
    set_seed(seed)
    device = get_device()
    p = paths(
        seed, backbone, train_frac, input_size=input_size, artifact_tag=artifact_tag
    )
    if p["model_best"].exists():
        print(f"  skip train, checkpoint exists: {p['model_best']}", flush=True)
        return

    rec = resolve_recipe(backbone, recipe_override)
    bs = int(rec["batch_size"])
    loaders = get_dataloaders(
        backbone,
        batch_size=bs,
        num_workers=num_workers,
        include_ood=log_msp_epoch,
        download=download,
        train_frac=train_frac,
        seed=seed,
        input_size=input_size,
        stain_cov=stain_cov,
    )
    model = get_model(backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, backbone, rec=rec)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0.0
    n_train = len(loaders["train"].dataset)
    sample, _, _ = next(iter(loaders["train"]))
    size_used = int(input_size) if input_size is not None else default_input_size(backbone)
    tag_s = f" tag={artifact_tag}" if artifact_tag else ""
    stain_s = " stain_cov=HED σ=0.20" if stain_cov else ""
    print(
        f"\n=== Train {backbone} on {device} "
        f"(seed={seed}, train_frac={train_frac}, n_train={n_train}, "
        f"input_size={size_used}, batch_chw={tuple(sample.shape)}, "
        f"optim={rec['optim']} lr={rec['lr']} wd={rec['wd']}{tag_s}{stain_s}) ===",
        flush=True,
    )
    epoch_rows: list[dict] = []

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
        train_loss = running_loss / max(n_train, 1)
        val_acc, _ = evaluate(model, loaders["id_val"], device)
        lr = scheduler.get_last_lr()[0]
        msp = float("nan")
        if log_msp_epoch:
            msp = msp_auroc_id_vs_ood(model, loaders["id_val"], loaders["ood"], device)
            epoch_rows.append(
                {
                    "backbone": backbone,
                    "seed": seed,
                    "epoch": epoch,
                    "id_val_acc": val_acc,
                    "msp_auroc": msp,
                }
            )
            p["epoch_msp_csv"].parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(epoch_rows).to_csv(p["epoch_msp_csv"], index=False)
        extra = f" msp_auroc={msp:.4f}" if log_msp_epoch else ""
        print(
            f"[Epoch {epoch:02d}/{NUM_EPOCHS}] LR={lr:.2e} "
            f"loss={train_loss:.4f} id_val_acc={val_acc:.4f}{extra}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), p["model_best"])
            print(f"  saved {p['model_best']} (acc={best_acc:.4f})")

    meta_path = p["model_dir"] / f"{p['stem']}_train_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "backbone": backbone,
                "stem": p["stem"],
                "seed": seed,
                "artifact_tag": artifact_tag,
                "best_id_val_acc": float(best_acc),
                "optim": rec["optim"],
                "lr": rec["lr"],
                "wd": rec["wd"],
                "batch_size": int(rec["batch_size"]),
                "stain_cov": bool(stain_cov),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {meta_path} best_id_val_acc={best_acc:.4f}", flush=True)


def extract_backbone(
    backbone: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    train_frac: float,
    input_size: int | None = None,
    artifact_tag: str | None = None,
    recipe_override: dict | None = None,
) -> None:
    device = get_device()
    p = paths(
        seed, backbone, train_frac, input_size=input_size, artifact_tag=artifact_tag
    )
    if p["features"].exists():
        print(f"  skip extract, features exist: {p['features']}", flush=True)
        return
    rec = resolve_recipe(backbone, recipe_override)
    bs = int(rec["batch_size"])
    loaders = get_dataloaders(
        backbone,
        batch_size=bs,
        num_workers=num_workers,
        download=False,
        train_frac=train_frac,
        seed=seed,
        input_size=input_size,
    )

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
        print(f"--- {name} ({len(loader.dataset)} samples) ---")
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
            f"{backbone}: fc_weight in_features={fc_weight.shape[1]} "
            f"!= avgpool feat_dim={feat_dim} (ViM/ReAct would be invalid)"
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
            "ood_split": OOD_SPLIT,
            "train_frac": train_frac,
            "input_size": int(input_size) if input_size is not None else default_input_size(backbone),
        },
        p["features"],
    )
    print(f"Saved {p['features']}")


def fc_params(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    return cnn_fc_params(model)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def analyze_backbone(
    backbone: str,
    seed: int,
    train_frac: float = 0.05,
    input_size: int | None = None,
    artifact_tag: str | None = None,
) -> dict:
    set_seed(seed)
    p = paths(
        seed, backbone, train_frac, input_size=input_size, artifact_tag=artifact_tag
    )
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
        "train_frac": float(data.get("train_frac", 1.0)),
        "ood_split": str(data.get("ood_split", OOD_SPLIT)),
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
        "logit_gap_estimator": "trim_1pct",
        "cvid_status": "appendix",
        "input_size": int(data.get("input_size", default_input_size(backbone))),
    }

    print(f"\n=== {backbone} Camelyon17 pilot summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return summary


def run_probe(download: bool) -> None:
    wait_for_camelyon()
    summary = split_summary(download=False)
    print("\n=== Camelyon17 split summary (WILDS official) ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    report_dir = Path("outputs/reports/camelyon17")
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(report_dir / "split_summary.csv", index=False)
    print(f"Wrote {report_dir / 'split_summary.csv'}")


def run_stage(
    stage: str,
    backbone: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    train_frac: float,
    download: bool,
    input_size: int | None = None,
    log_msp_epoch: bool = False,
    artifact_tag: str | None = None,
    recipe_override: dict | None = None,
    stain_cov: bool = False,
) -> None:
    names = resolve_names(backbone)
    summaries = []

    if stage == "probe":
        run_probe(download=download)
        return

    if stage == "download":
        print("Downloading WILDS Camelyon17 (~10GB compressed)...", flush=True)
        get_wilds_dataset(download=True)
        print("Download complete.", flush=True)
        return

    if stage in ("train", "all"):
        wait_for_camelyon()
        for name in names:
            train_backbone(
                name,
                seed,
                batch_size,
                num_workers,
                train_frac,
                download=False,
                input_size=input_size,
                log_msp_epoch=log_msp_epoch,
                artifact_tag=artifact_tag,
                recipe_override=recipe_override,
                stain_cov=stain_cov,
            )

    if stage in ("extract", "all"):
        wait_for_camelyon()
        for name in names:
            extract_backbone(
                name,
                seed,
                batch_size,
                num_workers,
                train_frac,
                input_size=input_size,
                artifact_tag=artifact_tag,
                recipe_override=recipe_override,
            )

    if stage in ("analyze", "all"):
        for name in names:
            summaries.append(
                analyze_backbone(
                    name,
                    seed,
                    train_frac,
                    input_size=input_size,
                    artifact_tag=artifact_tag,
                )
            )

    if summaries:
        report_dir = Path(
            f"outputs/reports/camelyon17/{run_tag(seed, train_frac, artifact_tag)}"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        existing = report_dir / "pilot_summary.csv"
        new = pd.DataFrame(summaries)
        if existing.exists():
            old = pd.read_csv(existing)
            drop = set(new["backbone"].tolist())
            if backbone == "new":
                drop |= set(NEW_CNN_BACKBONES)
            keep = old[~old["backbone"].isin(drop)]
            pd.concat([keep, new], ignore_index=True).to_csv(existing, index=False)
        else:
            new.to_csv(existing, index=False)
        print(f"\nWrote {existing}")
        if artifact_tag:
            print(
                f"skip invariance rebuild (artifact_tag={artifact_tag}; "
                "do not mix with official seed-42 W)",
                flush=True,
            )
        elif input_size is None:
            try:
                import runpy

                runpy.run_path(
                    str(Path(__file__).with_name("rebuild_architecture_invariance.py")),
                    run_name="__main__",
                )
            except Exception as exc:
                print(f"invariance rebuild skipped: {exc}", flush=True)
        else:
            print(
                f"skip invariance rebuild (input_size={input_size} override; "
                "do not mix with default-resolution zoo table)",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="WILDS Camelyon17 covariate-shift pilot")
    parser.add_argument(
        "--stage",
        choices=("probe", "download", "train", "extract", "analyze", "all"),
        default="all",
    )
    parser.add_argument(
        "--backbone",
        choices=[*ALL_BACKBONES, *VIT_BACKBONES, "all", "new"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.05,
        help="Fraction of WILDS train split (default 0.05 ≈ 15k patches for cheap pilot)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow WILDS auto-download when dataset missing",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Override spatial size (e.g. 224 for EffB3 resolution control). "
        "Writes tagged artifacts (effb3_224_*) so the default-size run is kept.",
    )
    parser.add_argument(
        "--log-msp-epoch",
        action="store_true",
        help="Each epoch, log MSP AUROC on ID-val vs OOD (hospital-2). Does not extract Maha/kNN.",
    )
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help="Subdirectory under fracX/ so this run cannot overwrite official seed-42.",
    )
    parser.add_argument("--optim", default=None, help="Override optimizer (adam/adamw/sgd).")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=None)
    parser.add_argument(
        "--recipe-batch-size",
        type=int,
        default=None,
        help="Override recipe batch size (cnn_family default is used if omitted).",
    )
    parser.add_argument(
        "--stain-cov",
        action="store_true",
        help="HED stain jitter on train only. Use with --artifact-tag stain_cov. "
        "Does not change eval/OOD transforms.",
    )
    args = parser.parse_args()
    recipe_override = None
    if any(v is not None for v in (args.optim, args.lr, args.wd, args.recipe_batch_size)):
        recipe_override = {
            "optim": args.optim,
            "lr": args.lr,
            "wd": args.wd,
            "batch_size": args.recipe_batch_size,
        }
    artifact_tag = args.artifact_tag
    if args.stain_cov and not artifact_tag:
        artifact_tag = "stain_cov"
    run_stage(
        args.stage,
        args.backbone,
        args.seed,
        args.batch_size,
        args.num_workers,
        args.train_frac,
        args.download,
        input_size=args.input_size,
        log_msp_epoch=args.log_msp_epoch,
        artifact_tag=artifact_tag,
        recipe_override=recipe_override,
        stain_cov=args.stain_cov,
    )


if __name__ == "__main__":
    main()
