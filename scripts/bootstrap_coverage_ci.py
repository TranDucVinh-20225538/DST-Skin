#!/usr/bin/env python3
"""Paired bootstrap CIs for OOD-triage coverage@risk 10% (CPU, no GPU).

Does not recompute Kendall W. Loads logits only; drops features after extract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.utils.scoring import OODScorer

ROOT = Path(__file__).resolve().parents[1]
CAMELYON = ROOT / "outputs/features/camelyon17/frac1/seed42"
MIDOG = ROOT / "outputs/features/midog/seed42"
OUT = ROOT / "outputs/reports/pathology_risk_coverage"
B = 1000
SEED = 42
TARGET = 0.10
OFFICIAL = [
    "resnet18",
    "resnet50",
    "densenet121",
    "convnext_tiny",
    "mobilenet_v3_large",
    "regnet_y_3_2gf",
    "effb3",
    "efficientnet_v2_s_224",
]


def load_pt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def coverage_at_risk(scores: np.ndarray, correct: np.ndarray, target: float = TARGET) -> float:
    order = np.argsort(-scores, kind="mergesort")
    csort = correct[order]
    n = csort.size
    prefix = np.cumsum(csort)
    k = np.arange(1, n + 1, dtype=np.float64)
    ok = (1.0 - prefix / k) <= target + 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.nonzero(ok)[0][-1] + 1) / n


def extract_ood_msp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_pt(path)
    z = to_np(data["ood_logits"])
    y = to_np(data["ood_labels"])
    del data
    scores = OODScorer.score_msp(z).astype(np.float64)
    correct = (z.argmax(1) == y).astype(np.float64)
    return scores, correct


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def boot_one(scores: np.ndarray, correct: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = scores.size
    out = np.empty(B, dtype=np.float64)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        out[i] = coverage_at_risk(scores[idx], correct[idx])
    return out


def boot_paired(
    s_a: np.ndarray,
    c_a: np.ndarray,
    s_b: np.ndarray,
    c_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = s_a.size
    a = np.empty(B, dtype=np.float64)
    b = np.empty(B, dtype=np.float64)
    d = np.empty(B, dtype=np.float64)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        a[i] = coverage_at_risk(s_a[idx], c_a[idx])
        b[i] = coverage_at_risk(s_b[idx], c_b[idx])
        d[i] = b[i] - a[i]
    return a, b, d


def fmt(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "OOD-triage coverage@risk 10% MSP. Percentile bootstrap, B=1000, seed=42.",
        "Not a new official W. Not mixed with OpenMIBOOD public R50.",
        "",
    ]
    cam = {}
    for name in OFFICIAL:
        path = CAMELYON / f"{name}_features.pt"
        print(f"load camelyon {name}", flush=True)
        cam[name] = extract_ood_msp(path)

    rng = np.random.default_rng(SEED)
    lines.append("=== Camelyon hospital-2, unpaired per backbone ===")
    points = {}
    for name in OFFICIAL:
        s, c = cam[name]
        point = coverage_at_risk(s, c)
        points[name] = point
        samples = boot_one(s, c, rng)
        lo, hi = percentile_ci(samples)
        lines.append(f"{name:24s} {fmt(point, lo, hi)}  n={s.size}")
        print(lines[-1], flush=True)

    print("paired DenseNet vs MobileNet", flush=True)
    a, b, d = boot_paired(*cam["densenet121"], *cam["mobilenet_v3_large"], rng)
    d_point = points["mobilenet_v3_large"] - points["densenet121"]
    lo_a, hi_a = percentile_ci(a)
    lo_b, hi_b = percentile_ci(b)
    lo_d, hi_d = percentile_ci(d)
    lines += [
        "",
        "=== Camelyon paired bootstrap (same OOD indices) ===",
        f"densenet121              {fmt(points['densenet121'], lo_a, hi_a)}",
        f"mobilenet_v3_large       {fmt(points['mobilenet_v3_large'], lo_b, hi_b)}",
        f"MobileNet - DenseNet     {fmt(d_point, lo_d, hi_d)}",
        "",
    ]
    print("\n".join(lines[-5:]), flush=True)

    mid_names = ["resnet18", "resnet50", "effb3"]
    mid = {}
    for name in mid_names:
        path = MIDOG / f"{name}_features.pt"
        print(f"load midog {name}", flush=True)
        mid[name] = extract_ood_msp(path)

    lines.append("=== MIDOG 1a vs 1b+1c, unpaired ===")
    for name in mid_names:
        s, c = mid[name]
        point = coverage_at_risk(s, c)
        samples = boot_one(s, c, np.random.default_rng(SEED + 7))
        lo, hi = percentile_ci(samples)
        lines.append(f"{name:24s} {fmt(point, lo, hi)}  n={s.size}")
        print(lines[-1], flush=True)

    out = OUT / "coverage_bootstrap_ci.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
