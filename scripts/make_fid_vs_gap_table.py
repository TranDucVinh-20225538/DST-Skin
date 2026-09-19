#!/usr/bin/env python3
"""Covariate-shift FID vs ΔAUROC/LogitGap — no mixed train protocols.

Camelyon MUST be full-scale (frac1/seed42). The 5% pilot is a different
training regime and is not comparable to skin/MIDOG on the severity axis.
CIFAR-10/SVHN stays out (semantic/far-OOD). CIFAR-10-C is a separate sweep.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("outputs/reports")
FID_CSV = ROOT / "domain_severity_fid.csv"
CAMELYON_FULL = ROOT / "camelyon17/frac1/seed42/pilot_summary.csv"
CAMELYON_PILOT = ROOT / "camelyon17/seed42/pilot_summary.csv"
MIDOG = ROOT / "midog/seed42/pilot_summary.csv"
SKIN_PRIMARY = ROOT / "pilot_primary_delta_logitgap.csv"
OUT = ROOT / "fid_vs_gap_covariate.csv"
OUT_CORR = ROOT / "pilot_correlation_covariate.csv"

METRIC_COLS = [
    "mahalanobis_auroc",
    "msp_auroc",
    "auroc_delta_maha_minus_msp",
    "logit_gap_mean_gap_vs_id",
    "logit_gap_mean_gap_vs_id_CI_low",
    "logit_gap_mean_gap_vs_id_CI_high",
]


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def skin_rows(fid: float) -> pd.DataFrame:
    src = _require(SKIN_PRIMARY)
    df = src[src["domain"] == "skin_isic_pad"].copy()
    if df.empty:
        raise RuntimeError("No skin_isic_pad rows in pilot_primary_delta_logitgap.csv")
    df["train_protocol"] = "full_isic_train"
    df["fid_score"] = fid
    df["logit_gap_estimator"] = "trim_1pct"
    return df


def camelyon_full_rows(fid: float) -> pd.DataFrame:
    if not CAMELYON_FULL.exists():
        raise FileNotFoundError(
            f"Camelyon full-scale summary missing: {CAMELYON_FULL}. "
            "Do not fall back to the 5% pilot."
        )
    df = pd.read_csv(CAMELYON_FULL)
    n = df["backbone"].nunique()
    if n < 3:
        raise RuntimeError(
            f"{CAMELYON_FULL} has {n} backbone(s); need resnet18/resnet50/effb3"
        )
    fracs = set(df["train_frac"].astype(float).tolist()) if "train_frac" in df.columns else set()
    if fracs and any(abs(f - 1.0) > 1e-9 for f in fracs):
        raise RuntimeError(f"Camelyon summary is not train_frac=1.0: {fracs}")
    if CAMELYON_PILOT.exists():
        # Explicitly unused — keep the 5% file on disk for the subsample
        # ablation, but never join it into this table.
        pass
    out = pd.DataFrame({
        "domain": "camelyon17",
        "backbone": df["backbone"],
        "seed": df["seed"],
        **{c: df[c] for c in METRIC_COLS},
        "logit_gap_estimator": "trim_1pct",
        "train_protocol": "full_train_frac_1.0",
        "fid_score": fid,
    })
    return out


def midog_rows(fid: float) -> pd.DataFrame:
    df = _require(MIDOG)
    protocol = []
    for _, row in df.iterrows():
        src = str(row.get("source", "") or "")
        if row["backbone"] == "resnet50" and "trained" not in src:
            protocol.append("frozen_openmibood")
        else:
            protocol.append("trained_on_1a")
    out = pd.DataFrame({
        "domain": "midog",
        "backbone": df["backbone"],
        "seed": df["seed"],
        **{c: df[c] for c in METRIC_COLS},
        "logit_gap_estimator": "trim_1pct",
        "train_protocol": protocol,
        "fid_score": fid,
    })
    return out


def main() -> None:
    fid = _require(FID_CSV).set_index("domain")["fid_score"]
    parts = [
        skin_rows(float(fid["skin_isic_pad"])),
        camelyon_full_rows(float(fid["camelyon17"])),
        midog_rows(float(fid["midog"])),
    ]
    cols = [
        "domain",
        "backbone",
        "seed",
        "train_protocol",
        "fid_score",
        *METRIC_COLS,
        "logit_gap_estimator",
    ]
    out = pd.concat(parts, ignore_index=True)[cols]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    # Covariate correlation table: same rows, no semantic CIFAR.
    out.to_csv(OUT_CORR, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(f"Wrote {OUT_CORR} (replaced 5% Camelyon mix)")
    print(out.to_string(index=False))
    print("\nΔAUROC by domain (min–max across backbones):")
    for domain, g in out.groupby("domain"):
        d = g["auroc_delta_maha_minus_msp"]
        print(
            f"  {domain:16s}  FID={g['fid_score'].iloc[0]:6.1f}  "
            f"ΔAUROC {d.min():.3f}–{d.max():.3f}  "
            f"LogitGap {g['logit_gap_mean_gap_vs_id'].min():+.3f}–"
            f"{g['logit_gap_mean_gap_vs_id'].max():+.3f}"
        )


if __name__ == "__main__":
    main()
