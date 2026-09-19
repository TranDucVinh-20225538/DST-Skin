"""Recompute dispersion_vs_separability CSVs from saved feature files (no retrain)."""

import argparse
import random

import numpy as np
import torch

try:
    from torch.serialization import add_safe_globals
except ImportError:
    def add_safe_globals(_):
        pass

from src.utils.benchmark_metrics import build_dispersion_vs_separability_table

SEED = 42

BACKBONES = {
    "resnet18": (
        "outputs/features/resnet18_isic_pad_features.pt",
        "outputs/reports/resnet18_dispersion_vs_separability.csv",
    ),
    "resnet50": (
        "outputs/features/resnet50_isic_pad_features.pt",
        "outputs/reports/resnet50_dispersion_vs_separability.csv",
    ),
    "effb3": (
        "outputs/features/effb3_isic_pad_features.pt",
        "outputs/reports/effb3_dispersion_vs_separability.csv",
    ),
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_features(path: str) -> dict:
    add_safe_globals([np.core.multiarray._reconstruct])
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def update_backbone(name: str, feature_path: str, out_csv: str, n_perm: int) -> None:
    data = load_features(feature_path)
    val_logits = to_numpy(data["val_logits"])
    ood_logits = to_numpy(data["ood_logits"])
    val_feats = to_numpy(data["val_feats"])
    ood_feats = to_numpy(data["ood_feats"])

    df = build_dispersion_vs_separability_table(
        val_logits,
        ood_logits,
        val_feats,
        ood_feats,
        n_silhouette_perm=n_perm,
        seed=SEED,
    )
    df.to_csv(out_csv, index=False)
    print(f"\n=== {name} ===")
    print(df.to_string(index=False))
    print(f"Saved: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-perm",
        type=int,
        default=1000,
        help="Label-shuffle permutations for silhouette null (default: 1000)",
    )
    parser.add_argument(
        "--backbone",
        choices=list(BACKBONES) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    set_seed(SEED)
    names = list(BACKBONES) if args.backbone == "all" else [args.backbone]
    for name in names:
        feature_path, out_csv = BACKBONES[name]
        update_backbone(name, feature_path, out_csv, args.n_perm)


if __name__ == "__main__":
    main()
