#!/usr/bin/env python3
"""Coverage@risk validation-split redo (decision_precommit_coverage_val_split.md).

CPU only, no re-extraction. Hospital-2 OOD patches are split 50/50 by patient
(stratified on tumor label, seed 42), the split is frozen to disk before any
model is scored, then:
  Pipeline 1: pick max MSP AUROC on val-half, report its test-half coverage@risk 10%.
  Pipeline 2: pick max coverage@risk 10% on val-half, report its test-half coverage.
  Oracle: full-test-set numbers already in the paper (0.078 / 0.904), contrast only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.benchmark_metrics import calc_auroc  # noqa: E402
from src.utils.scoring import OODScorer  # noqa: E402

FEATS = ROOT / "outputs/features/camelyon17/frac1/seed42"
META = ROOT / "data/raw/wilds/camelyon17_v1.0/metadata.csv"
OUT = ROOT / "outputs/reports"
SPLIT_CSV = OUT / "camelyon_coverage_val_split_patients.csv"
RES_CSV = OUT / "camelyon_coverage_val_split.csv"
RES_TXT = OUT / "camelyon_coverage_val_split.txt"
SPLIT_SEED = 42
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


def coverage_at_risk(scores: np.ndarray, correct: np.ndarray, target: float = TARGET) -> float:
    order = np.argsort(-scores, kind="mergesort")
    csort = correct[order]
    k = np.arange(1, csort.size + 1, dtype=np.float64)
    ok = (1.0 - np.cumsum(csort) / k) <= target + 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.nonzero(ok)[0][-1] + 1) / csort.size


def freeze_split(meta: pd.DataFrame) -> np.ndarray:
    """Return boolean mask (True = val-half) over hospital-2 rows in metadata order."""
    if SPLIT_CSV.exists():
        frozen = pd.read_csv(SPLIT_CSV, dtype={"patient": str})
        val_patients = set(frozen.loc[frozen.half == "val", "patient"])
        return meta["patient"].isin(val_patients).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SPLIT_SEED)
    val_idx, _ = next(sgkf.split(np.zeros(len(meta)), meta["tumor"], groups=meta["patient"]))
    is_val = np.zeros(len(meta), dtype=bool)
    is_val[val_idx] = True
    rows = []
    for p, g in meta.groupby("patient"):
        half = "val" if is_val[g.index.map(meta.index.get_loc)].all() else "test"
        rows.append({
            "patient": p,
            "slides": " ".join(str(s) for s in sorted(g["slide"].unique())),
            "n_patches": len(g),
            "tumor_frac": round(float(g["tumor"].mean()), 4),
            "half": half,
        })
    pd.DataFrame(rows).to_csv(SPLIT_CSV, index=False)
    return is_val


def main() -> None:
    meta = pd.read_csv(META, index_col=0, dtype={"patient": str})
    meta = meta[meta.center == 2].reset_index(drop=True)
    is_val = freeze_split(meta)
    halves = {"val": is_val, "test": ~is_val}
    print(f"split frozen: val n={is_val.sum()} test n={(~is_val).sum()}", flush=True)

    rows = []
    for bb in OFFICIAL:
        print(f"load {bb}", flush=True)
        d = torch.load(FEATS / f"{bb}_features.pt", map_location="cpu", weights_only=False)
        id_z = np.asarray(d["val_logits"], dtype=np.float64)
        ood_z = np.asarray(d["ood_logits"], dtype=np.float64)
        ood_y = np.asarray(d["ood_labels"])
        del d
        if not np.array_equal(ood_y, meta["tumor"].to_numpy()):
            raise RuntimeError(f"{bb}: OOD order does not match WILDS metadata")
        id_s = OODScorer.score_msp(id_z)
        ood_s = OODScorer.score_msp(ood_z)
        correct = (ood_z.argmax(1) == ood_y).astype(np.float64)
        rec = {
            "backbone": bb,
            "auroc_full": calc_auroc(id_s, ood_s),
            "cov10_full": coverage_at_risk(ood_s, correct),
        }
        for h, mask in halves.items():
            rec[f"auroc_{h}"] = calc_auroc(id_s, ood_s[mask])
            rec[f"cov10_{h}"] = coverage_at_risk(ood_s[mask], correct[mask])
            rec[f"n_{h}"] = int(mask.sum())
        rows.append(rec)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES_CSV, index=False)
    p1 = df.loc[df.auroc_val.idxmax()]
    p2 = df.loc[df.cov10_val.idxmax()]
    o_auc = df.loc[df.auroc_full.idxmax()]
    o_cov = df.loc[df.cov10_full.idxmax()]
    best_test = df.cov10_test.max()
    split = pd.read_csv(SPLIT_CSV, dtype={"patient": str})
    lines = [
        "Camelyon hospital-2 coverage@risk 10% (MSP), validation-split redo. Seed-42 official 8 CNNs.",
        f"Split: StratifiedGroupKFold(2, shuffle, random_state={SPLIT_SEED}) on tumor, grouped by patient. Frozen in {SPLIT_CSV.name}.",
        split.to_string(index=False),
        f"val-half n={int(is_val.sum())}, test-half n={int((~is_val).sum())}",
        "",
        df.round(4).to_string(index=False),
        "",
        f"Pipeline 1 (AUROC-select on val):    {p1.backbone}  val AUROC {p1.auroc_val:.3f} -> test cov@10 {p1.cov10_test:.3f}",
        f"Pipeline 2 (coverage-select on val): {p2.backbone}  val cov@10 {p2.cov10_val:.3f} -> test cov@10 {p2.cov10_test:.3f}",
        f"Pipeline 2 - Pipeline 1 (test-half coverage): {p2.cov10_test - p1.cov10_test:+.3f}",
        f"Best achievable test-half coverage (any backbone): {best_test:.3f} ({df.loc[df.cov10_test.idxmax()].backbone})",
        f"Oracle, full test set (contrast only): AUROC-best {o_auc.backbone} cov {o_auc.cov10_full:.3f}; "
        f"coverage-best {o_cov.backbone} cov {o_cov.cov10_full:.3f}",
    ]
    RES_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
