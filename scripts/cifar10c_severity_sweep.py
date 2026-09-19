#!/usr/bin/env python3
"""CIFAR-10-C severity sweep: LogitGap and ΔAUROC vs external severity 1–5.

Reuses frozen CIFAR-10 classifiers from the SVHN pilot. Does not retrain.
Does not touch scoring.py / benchmark_metrics formulas / Camelyon / MIDOG.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.datasets.cifar_ood import CIFAR10_NUM_CLASSES
from src.datasets.cifar10c_ood import (
    CORRUPTIONS,
    data_ready,
    get_cifar10c_loader,
)
from src.models.efficientnet_b3 import get_efficientnet_b3
from src.models.resnet18 import get_resnet18
from src.models.resnet50 import get_resnet50
from src.utils.benchmark_metrics import (
    N_BOOTSTRAP,
    aggregate_logit_gap,
    bootstrap_ci,
    bootstrap_dispersion_gap_ci,
    calc_auroc,
    per_sample_logit_gap,
)
from src.utils.feature_extractor import extract_features_and_logits
from src.utils.scoring import OODScorer

BACKBONES = ("resnet18", "resnet50", "effb3")
SEED = 42


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model(backbone: str):
    if backbone == "resnet18":
        return get_resnet18(num_classes=CIFAR10_NUM_CLASSES, pretrained=False)
    if backbone == "resnet50":
        return get_resnet50(num_classes=CIFAR10_NUM_CLASSES, pretrained=False)
    if backbone == "effb3":
        return get_efficientnet_b3(num_classes=CIFAR10_NUM_CLASSES, pretrained=False)
    raise ValueError(backbone)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_pilot_features(backbone: str, seed: int = SEED) -> dict:
    path = Path(f"outputs/features/cifar10_svhn/seed{seed}/{backbone}_features.pt")
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    keep = {}
    for k, v in data.items():
        if k.endswith("_logits") or k.endswith("_feats") or k.endswith("_labels"):
            keep[k] = to_numpy(v)
    return keep


def load_classifier(backbone: str, device: torch.device, seed: int = SEED) -> torch.nn.Module:
    ckpt = Path(f"data/models/cifar10_svhn/seed{seed}/{backbone}_best.pth")
    model = get_model(backbone)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def bootstrap_delta_auroc(
    id_maha: np.ndarray,
    ood_maha: np.ndarray,
    id_msp: np.ndarray,
    ood_msp: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, float]:
    """Joint percentile bootstrap of ΔAUROC = Maha − MSP (same ID/OOD draws)."""
    n_id, n_ood = len(id_maha), len(ood_maha)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        id_idx = rng.integers(0, n_id, n_id)
        ood_idx = rng.integers(0, n_ood, n_ood)
        d_maha = calc_auroc(id_maha[id_idx], ood_maha[ood_idx])
        d_msp = calc_auroc(id_msp[id_idx], ood_msp[ood_idx])
        deltas[i] = d_maha - d_msp
    point = calc_auroc(id_maha, ood_maha) - calc_auroc(id_msp, ood_msp)
    return {
        "delta_auroc": float(point),
        "delta_auroc_ci_lo": float(np.percentile(deltas, 2.5)),
        "delta_auroc_ci_hi": float(np.percentile(deltas, 97.5)),
    }


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = SEED):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = float(spearmanr(x[idx], y[idx])[0])
    return {
        "rho": float(rho),
        "p_value": float(p),
        "rho_ci_lo": float(np.percentile(boots, 2.5)),
        "rho_ci_hi": float(np.percentile(boots, 97.5)),
    }


def sweep_backbone(
    backbone: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    n_boot: int,
) -> pd.DataFrame:
    print(f"\n=== CIFAR-10-C sweep {backbone} on {device} ===", flush=True)
    feats = load_pilot_features(backbone)
    train_feats = feats["train_feats"]
    id_logits = feats["val_logits"]
    id_feats = feats["val_feats"]

    scorer = OODScorer(k_nearest=50, use_react=False, use_vim=False)
    scorer.fit(train_feats)
    scores_id = scorer.get_all_scores(id_logits, id_feats)
    model = load_classifier(backbone, device)

    rows = []
    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            t0 = time.time()
            loader = get_cifar10c_loader(
                backbone, corruption, severity,
                batch_size=batch_size, num_workers=num_workers,
            )
            ood_logits, ood_feats = extract_features_and_logits(model, loader, device)
            ood_logits = to_numpy(ood_logits)
            ood_feats = to_numpy(ood_feats)
            scores_ood = scorer.get_all_scores(ood_logits, ood_feats)

            delta = bootstrap_delta_auroc(
                scores_id["mahalanobis"], scores_ood["mahalanobis"],
                scores_id["msp"], scores_ood["msp"],
                n_boot=n_boot,
            )
            maha_ci = bootstrap_ci(
                scores_id["mahalanobis"], scores_ood["mahalanobis"],
                n_boot=n_boot, seed=SEED,
            )
            msp_ci = bootstrap_ci(
                scores_id["msp"], scores_ood["msp"],
                n_boot=n_boot, seed=SEED,
            )
            gap = bootstrap_dispersion_gap_ci(
                id_logits, ood_logits, n_boot=n_boot, seed=SEED,
            )
            elapsed = time.time() - t0
            row = {
                "backbone": backbone,
                "corruption_type": corruption,
                "severity_level": severity,
                "delta_auroc": delta["delta_auroc"],
                "delta_auroc_ci_lo": delta["delta_auroc_ci_lo"],
                "delta_auroc_ci_hi": delta["delta_auroc_ci_hi"],
                "mahalanobis_auroc": maha_ci["AUROC"],
                "msp_auroc": msp_ci["AUROC"],
                "logit_gap_trim1": gap["logit_gap_mean_gap_vs_id"],
                "logit_gap_ci_lo": gap["logit_gap_mean_gap_vs_id_CI_low"],
                "logit_gap_ci_hi": gap["logit_gap_mean_gap_vs_id_CI_high"],
                "seconds": elapsed,
            }
            rows.append(row)
            print(
                f"  {corruption:20s} s{severity}  ΔAUROC={row['delta_auroc']:+.4f}  "
                f"LogitGap={row['logit_gap_trim1']:+.4f}  ({elapsed:.1f}s)",
                flush=True,
            )
    return pd.DataFrame(rows)


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    analyses = []
    for backbone, sub in df.groupby("backbone"):
        pooled_gap = bootstrap_spearman(
            sub["severity_level"].values, np.abs(sub["logit_gap_trim1"].values)
        )
        pooled_d = bootstrap_spearman(
            sub["severity_level"].values, sub["delta_auroc"].values
        )
        analyses.append({
            "backbone": backbone,
            "scope": "pooled",
            "corruption_type": "ALL",
            "n": len(sub),
            "spearman_abs_logit_gap": pooled_gap["rho"],
            "spearman_abs_logit_gap_ci_lo": pooled_gap["rho_ci_lo"],
            "spearman_abs_logit_gap_ci_hi": pooled_gap["rho_ci_hi"],
            "spearman_abs_logit_gap_p": pooled_gap["p_value"],
            "spearman_delta_auroc": pooled_d["rho"],
            "spearman_delta_auroc_ci_lo": pooled_d["rho_ci_lo"],
            "spearman_delta_auroc_ci_hi": pooled_d["rho_ci_hi"],
            "spearman_delta_auroc_p": pooled_d["p_value"],
        })
        for corruption, csub in sub.groupby("corruption_type"):
            g = spearmanr(csub["severity_level"], np.abs(csub["logit_gap_trim1"]))
            d = spearmanr(csub["severity_level"], csub["delta_auroc"])
            analyses.append({
                "backbone": backbone,
                "scope": "per_corruption",
                "corruption_type": corruption,
                "n": len(csub),
                "spearman_abs_logit_gap": float(g[0]),
                "spearman_abs_logit_gap_ci_lo": np.nan,
                "spearman_abs_logit_gap_ci_hi": np.nan,
                "spearman_abs_logit_gap_p": float(g[1]),
                "spearman_delta_auroc": float(d[0]),
                "spearman_delta_auroc_ci_lo": np.nan,
                "spearman_delta_auroc_ci_hi": np.nan,
                "spearman_delta_auroc_p": float(d[1]),
            })
    return pd.DataFrame(analyses)


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10-C severity sweep")
    parser.add_argument("--backbone", default="all",
                        choices=[*BACKBONES, "all"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    marker = data_ready()
    if not marker.exists():
        raise FileNotFoundError(
            f"CIFAR-10-C not found at {marker}. Download Hendrycks tar to data/raw/cifar10c/"
        )

    names = list(BACKBONES) if args.backbone == "all" else [args.backbone]
    device = get_device()
    t0 = time.time()
    frames = [
        sweep_backbone(bb, device, args.batch_size, args.num_workers, args.n_boot)
        for bb in names
    ]
    sweep = pd.concat(frames, ignore_index=True)
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / "cifar10c_severity_sweep.csv"
    # Specified columns first; extras kept at the end.
    preferred = [
        "corruption_type", "severity_level", "delta_auroc",
        "delta_auroc_ci_lo", "delta_auroc_ci_hi",
        "logit_gap_trim1", "logit_gap_ci_lo", "logit_gap_ci_hi",
    ]
    cols = ["backbone"] + preferred + [c for c in sweep.columns if c not in ["backbone", *preferred]]
    sweep = sweep[cols]
    sweep.to_csv(sweep_path, index=False)

    corr = analyze_correlations(sweep)
    corr_path = out_dir / "cifar10c_severity_spearman.csv"
    corr.to_csv(corr_path, index=False)
    elapsed = time.time() - t0
    print(f"\nWrote {sweep_path}")
    print(f"Wrote {corr_path}")
    print(f"Total runtime {elapsed/60:.1f} min")
    print("\n=== Pooled Spearman (severity 1–5 vs metric) ===")
    print(corr.loc[corr.scope == "pooled"].to_string(index=False))


if __name__ == "__main__":
    main()
