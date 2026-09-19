#!/usr/bin/env python3
"""Single-hypothesis check: ID-set ECE as a confound for LogitGap vs ΔAUROC.

Does not retrain. Does not touch CIFAR-10-C or locked benchmark_metrics formulas.
MIDOG stored val_* is ID-test 1a (the same split used for LogitGap/ΔAUROC).
Skin/Camelyon val_* are the ID validation splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from src.utils.scoring import OODScorer

OUT = Path("outputs/reports/ece_confound_check.csv")
GAP = Path("outputs/reports/fid_vs_gap_covariate.csv")
N_BINS = 15

CELLS = [
    (
        "skin_isic_pad",
        "resnet18",
        Path("outputs/features/resnet18_isic_pad_features.pt"),
        "id_val",
    ),
    (
        "skin_isic_pad",
        "resnet50",
        Path("outputs/features/resnet50_isic_pad_features.pt"),
        "id_val",
    ),
    (
        "skin_isic_pad",
        "effb3",
        Path("outputs/features/effb3_isic_pad_features.pt"),
        "id_val",
    ),
    (
        "camelyon17",
        "resnet18",
        Path("outputs/features/camelyon17/frac1/seed42/resnet18_features.pt"),
        "id_val",
    ),
    (
        "camelyon17",
        "resnet50",
        Path("outputs/features/camelyon17/frac1/seed42/resnet50_features.pt"),
        "id_val",
    ),
    (
        "camelyon17",
        "effb3",
        Path("outputs/features/camelyon17/frac1/seed42/effb3_features.pt"),
        "id_val",
    ),
    (
        "midog",
        "resnet18",
        Path("outputs/features/midog/seed42/resnet18_features.pt"),
        "id_test_1a",
    ),
    (
        "midog",
        "resnet50",
        Path("outputs/features/midog/seed42/resnet50_features.pt"),
        "id_test_1a",
    ),
    (
        "midog",
        "effb3",
        Path("outputs/features/midog/seed42/effb3_features.pt"),
        "id_test_1a",
    ),
]


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_id_logits_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    logits = to_numpy(data["val_logits"])
    labels = to_numpy(data["val_labels"]).reshape(-1)
    del data
    return logits, labels


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    z = torch.as_tensor(np.asarray(logits, dtype=np.float64))
    return F.softmax(z, dim=1).numpy()


def ece_maxprob(logits: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """Guo et al. 2017 equal-width ECE: confidence = max softmax.

    For binary logits this matches OODScorer.compute_ece / calibration.py
    (conf = max(p, 1-p), pred = p>0.5). For C>2 (MIDOG) the binary helpers
    would be wrong, so this is the one definition used for all 9 cells.
    """
    probs = softmax_probs(logits)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf > lo) & (conf <= hi)
        prop = float(np.mean(in_bin))
        if prop > 0:
            acc = float(np.mean(pred[in_bin] == labels[in_bin]))
            ece += abs(float(np.mean(conf[in_bin])) - acc) * prop
    return float(ece)


def binary_ece_existing(logits: np.ndarray, labels: np.ndarray) -> float | None:
    if logits.shape[1] != 2:
        return None
    probs = softmax_probs(logits)
    return float(OODScorer.compute_ece(probs, labels, n_bins=N_BINS))


def main() -> None:
    gap = pd.read_csv(GAP)
    rows = []
    for domain, backbone, path, id_split in CELLS:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"ECE {domain}/{backbone} <- {path}", flush=True)
        logits, labels = load_id_logits_labels(path)
        ece = ece_maxprob(logits, labels)
        ece_bin = binary_ece_existing(logits, labels)
        if ece_bin is not None and abs(ece_bin - ece) > 1e-8:
            print(
                f"  warn: binary compute_ece={ece_bin:.6f} vs maxprob={ece:.6f}",
                flush=True,
            )
        acc = float((logits.argmax(axis=1) == labels).mean())
        g = gap[(gap["domain"] == domain) & (gap["backbone"] == backbone)]
        if len(g) != 1:
            raise RuntimeError(f"expected 1 gap row for {domain}/{backbone}, got {len(g)}")
        rows.append({
            "domain": domain,
            "backbone": backbone,
            "ece_id": ece,
            "logit_gap_trim1": float(g["logit_gap_mean_gap_vs_id"].iloc[0]),
            "delta_auroc": float(g["auroc_delta_maha_minus_msp"].iloc[0]),
            "id_split": id_split,
            "n_id": int(len(labels)),
            "id_acc": acc,
        })
        print(
            f"  ECE={ece:.4f}  acc={acc:.4f}  n={len(labels)}  "
            f"LogitGap={rows[-1]['logit_gap_trim1']:+.3f}  "
            f"ΔAUROC={rows[-1]['delta_auroc']:.3f}",
            flush=True,
        )
        del logits, labels

    df = pd.DataFrame(rows)
    rho_gap, p_gap = spearmanr(df["ece_id"], df["logit_gap_trim1"])
    rho_dlt, p_dlt = spearmanr(df["ece_id"], df["delta_auroc"])
    rho_abs, p_abs = spearmanr(df["ece_id"], df["logit_gap_trim1"].abs())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Requested schema first; metadata after.
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print(df.to_string(index=False))
    print("\nPooled Spearman (n=9):")
    print(f"  ECE vs LogitGap (signed): ρ={rho_gap:.3f}  p={p_gap:.3f}")
    print(f"  ECE vs |LogitGap|:        ρ={rho_abs:.3f}  p={p_abs:.3f}")
    print(f"  ECE vs ΔAUROC:            ρ={rho_dlt:.3f}  p={p_dlt:.3f}")

    cam = df[df["domain"] == "camelyon17"].set_index("backbone")
    print("\nCamelyon full ranking check (hypothesis: high ECE → |LogitGap| high, Δ low):")
    for bb in ("effb3", "resnet50", "resnet18"):
        r = cam.loc[bb]
        print(
            f"  {bb:8s}  ECE={r['ece_id']:.4f}  |LogitGap|={abs(r['logit_gap_trim1']):.3f}  "
            f"ΔAUROC={r['delta_auroc']:.3f}"
        )


if __name__ == "__main__":
    main()
