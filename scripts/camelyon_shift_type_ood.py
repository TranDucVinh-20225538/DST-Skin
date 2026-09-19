#!/usr/bin/env python3
"""Extract arm T (MIDOG 1a) and arm F (CIFAR-10 test) as OOD for frozen Camelyon classifiers.

No train. Does not overwrite official features.pt / score CSVs / Kendall W.
Read table: decision_precommit_shift_type.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.camelyon_ood import build_transform as camelyon_eval_transform
from src.datasets.cifar_ood import DATA_ROOT as CIFAR_DATA_ROOT
from src.datasets.cifar_ood import _PathDataset
from src.datasets.midog_ood import ImglistTupleDataset, data_ready as midog_ready
from src.datasets.midog_ood import imglist_path
from src.models.cnn_family import (
    fc_params as cnn_fc_params,
    get_cnn_backbone,
    input_size as default_input_size,
    recipe as cnn_recipe,
)
from src.utils.benchmark_metrics import build_score_comparison_df
from src.utils.feature_extractor import extract_features_and_logits
from src.utils.scoring import OODScorer

try:
    from torch.serialization import add_safe_globals
except ImportError:
    def add_safe_globals(_):
        pass

OOD_METHODS = (
    "msp",
    "energy",
    "react_energy",
    "logit_norm",
    "mahalanobis",
    "knn",
    "vim",
)

# Official 8 + held-out ViT. Jump 6/6 ignores vit_b_16.
CELLS: tuple[tuple[str, int | None], ...] = (
    ("resnet18", None),
    ("resnet50", None),
    ("densenet121", None),
    ("convnext_tiny", None),
    ("mobilenet_v3_large", None),
    ("regnet_y_3_2gf", None),
    ("effb3", None),
    ("efficientnet_v2_s", 224),
    ("vit_b_16", None),
)
ARMS = ("T", "F")
SEED = 42
TRAIN_FRAC = 1.0
FEAT_DIR = Path("outputs/features/camelyon17/frac1/seed42")
CKPT_DIR = Path("data/models/camelyon17/frac1/seed42")
OUT_FEAT = FEAT_DIR / "shift_type"
OUT_REP = Path("outputs/reports/camelyon17/frac1/seed42/shift_type")


def stem_of(backbone: str, input_size: int | None) -> str:
    if input_size is None:
        return backbone
    if int(input_size) == int(default_input_size(backbone)):
        return backbone
    return f"{backbone}_{int(input_size)}"


def size_of(backbone: str, input_size: int | None) -> int:
    return int(input_size) if input_size is not None else int(default_input_size(backbone))


def load_id_pack(stem: str) -> dict:
    path = FEAT_DIR / f"{stem}_features.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing official features (do not retrain): {path}")
    reconstruct = getattr(np.core.multiarray, "_reconstruct", None)
    if reconstruct is not None:
        add_safe_globals([reconstruct])
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    return data


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ood_loader(arm: str, backbone: str, input_size: int | None, num_workers: int) -> DataLoader:
    """Camelyon ImageNet eval transform on both arms. Not MIDOG adapter stats."""
    size = size_of(backbone, input_size)
    tf = camelyon_eval_transform(backbone, train=False, input_size=size)
    bs = int(cnn_recipe(backbone)["batch_size"])
    if arm == "T":
        if not midog_ready().exists():
            raise FileNotFoundError(f"MIDOG crops missing: {midog_ready()}")
        ds = ConcatDataset(
            [
                ImglistTupleDataset(imglist_path("train"), "data/raw/midog", tf),
                ImglistTupleDataset(imglist_path("id_val"), "data/raw/midog", tf),
                ImglistTupleDataset(imglist_path("id_test"), "data/raw/midog", tf),
            ]
        )
        prefix = "midog_1a"
    elif arm == "F":
        cifar = datasets.CIFAR10(
            root=f"{CIFAR_DATA_ROOT}/cifar10",
            train=False,
            download=False,
            transform=tf,
        )
        ds = _PathDataset(cifar, "cifar10/test")
        prefix = "cifar10_test"
    else:
        raise ValueError(arm)
    print(f"  {arm} {prefix}: n={len(ds)} input={size} bs={bs}", flush=True)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def extract_arm(
    backbone: str,
    input_size: int | None,
    arm: str,
    num_workers: int,
    force: bool,
) -> Path:
    stem = stem_of(backbone, input_size)
    out_pt = OUT_FEAT / f"{stem}_{arm}_ood.pt"
    if out_pt.exists() and not force:
        print(f"  skip extract, exists: {out_pt}", flush=True)
        return out_pt
    ckpt = CKPT_DIR / f"{stem}_best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing official checkpoint: {ckpt}")

    device = get_device()
    model = get_cnn_backbone(backbone, num_classes=2, pretrained=False)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    loader = ood_loader(arm, backbone, input_size, num_workers)
    print(f"=== extract {stem} arm={arm} on {device} ===", flush=True)
    logits, feats = extract_features_and_logits(model, loader, device)
    fc_weight, fc_bias = cnn_fc_params(model)
    OUT_FEAT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ood_logits": logits,
            "ood_feats": feats,
            "arm": arm,
            "backbone": backbone,
            "stem": stem,
            "input_size": size_of(backbone, input_size),
            "fc_weight": fc_weight,
            "fc_bias": fc_bias,
        },
        out_pt,
    )
    print(f"  saved {out_pt}", flush=True)
    return out_pt


def analyze_arm(backbone: str, input_size: int | None, arm: str, force: bool) -> Path:
    stem = stem_of(backbone, input_size)
    out_csv = OUT_REP / f"{stem}_{arm}_score_comparison.csv"
    if out_csv.exists() and not force:
        print(f"  skip analyze, exists: {out_csv}", flush=True)
        return out_csv
    pack = load_id_pack(stem)
    ood = torch.load(OUT_FEAT / f"{stem}_{arm}_ood.pt", map_location="cpu", weights_only=False)

    train_logits = to_numpy(pack["train_logits"])
    train_feats = to_numpy(pack["train_feats"])
    val_logits = to_numpy(pack["val_logits"])
    val_feats = to_numpy(pack["val_feats"])
    ood_logits = to_numpy(ood["ood_logits"])
    ood_feats = to_numpy(ood["ood_feats"])
    fc_weight = to_numpy(pack["fc_weight"])
    fc_bias = to_numpy(pack["fc_bias"])

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
    df = build_score_comparison_df(scores_id, scores_ood, methods, seed=SEED)
    OUT_REP.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    msp = float(df.loc[df["Method"] == "msp", "AUROC"].iloc[0])
    print(f"  {stem} {arm} MSP={msp:.4f} -> {out_csv}", flush=True)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument(
        "--backbone",
        default="all",
        help="one official name, or all (includes vit_b_16 held-out)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("extract", "analyze", "all"),
        default="all",
    )
    args = parser.parse_args()
    arms = ARMS if args.arm == "all" else (args.arm,)
    cells = CELLS if args.backbone == "all" else tuple(
        c for c in CELLS if c[0] == args.backbone or stem_of(*c) == args.backbone
    )
    if not cells:
        raise SystemExit(f"unknown backbone {args.backbone}")

    print(
        f"shift-type extract-only. official W untouched. arms={arms} n_cells={len(cells)}",
        flush=True,
    )
    for backbone, input_size in cells:
        for arm in arms:
            if args.stage in ("extract", "all"):
                extract_arm(backbone, input_size, arm, args.num_workers, args.force)
            if args.stage in ("analyze", "all"):
                analyze_arm(backbone, input_size, arm, args.force)
    print("shift-type done. run scripts/camelyon_shift_type_read.py", flush=True)


if __name__ == "__main__":
    main()
