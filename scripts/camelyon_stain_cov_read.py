#!/usr/bin/env python3
"""Apply decision_precommit_stain_cov.md. Official W stays no-stain seed 42."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

JUMP_LINE = 0.6947
GAP_BAR = 0.154
DENSE_FLOOR = 0.80
ID_FLOOR = 0.97
R18_FLOOR = 0.574
R50_FLOOR = 0.515
COV_FLOOR = 0.135
OFF = {"resnet18": 0.574, "resnet50": 0.515, "densenet121": 0.883}
REP = Path("outputs/reports/camelyon17/frac1/stain_cov/seed42")
MOD = Path("data/models/camelyon17/frac1/stain_cov/seed42")
OUT = Path("outputs/reports/camelyon_stain_cov.txt")
BBS = ("resnet18", "resnet50", "densenet121")


def msp_csv(path: Path) -> float:
    df = pd.read_csv(path)
    col = "Method" if "Method" in df.columns else "method"
    return float(df.loc[df[col].astype(str).str.lower() == "msp", "AUROC"].iloc[0])


def id_acc(bb: str) -> float | None:
    path = MOD / f"{bb}_train_meta.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["best_id_val_acc"])


def aux(bb: str) -> dict:
    path = REP / f"{bb}_stain_cov_aux.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def hits_bar(msp: float, dense_msp: float, floor: float) -> bool:
    return msp >= JUMP_LINE or ((dense_msp - msp) <= GAP_BAR and msp >= floor)


def main() -> None:
    missing = [
        p
        for bb in BBS
        for p in (
            REP / f"{bb}_score_comparison.csv",
            REP / f"{bb}_hospital1_score_comparison.csv",
        )
        if not p.exists()
    ]
    if missing:
        print("stain-cov readout waiting for:")
        for p in missing:
            print(f"  {p}")
        return
    h2 = {bb: msp_csv(REP / f"{bb}_score_comparison.csv") for bb in BBS}
    h1 = {bb: msp_csv(REP / f"{bb}_hospital1_score_comparison.csv") for bb in BBS}
    acc = {bb: id_acc(bb) for bb in BBS}
    cov = {bb: aux(bb).get("hospital2_msp_coverage_at_risk10") for bb in BBS}
    dirty = any(a is not None and a < ID_FLOOR for a in acc.values()) or h2["densenet121"] < DENSE_FLOOR
    r18_h2 = hits_bar(h2["resnet18"], h2["densenet121"], R18_FLOOR)
    r50_h2 = hits_bar(h2["resnet50"], h2["densenet121"], R50_FLOOR)
    r18_h1 = hits_bar(h1["resnet18"], h1["densenet121"], R18_FLOOR)
    cov_ok = cov["resnet18"] is not None and float(cov["resnet18"]) >= COV_FLOOR
    gap = h2["densenet121"] - h2["resnet18"]

    if dirty:
        branch = "DIRTY — Job-B-class fail. Appendix kill. No Viên 2. CVPR still on the table."
    elif not r18_h2 or not cov_ok:
        branch = "MISS — hospital-2 R18 bar or coverage fail. One more method viên later, then stop. CVPR still on the table."
    elif not r50_h2:
        branch = "WEAK HIT — R18 hospital-2 hits, R50 does not. Method paragraph, no 8-CNN, no Viên 2. CVPR still on the table."
    elif not r18_h1:
        branch = "WEAK HIT — hospital-2 both ResNets hit, hospital-1 R18 misses. Do not claim stain-cov generalizes. No Viên 2."
    else:
        branch = "STRONG HIT — 3/3 + hospital-1. Only now consider 8-CNN stain-cov and whether to open Viên 2 / skip CVPR."

    lines = [
        "Viên 1 stain-covariate readout. Official W stays no-stain seed 42.",
        "decision_precommit_stain_cov.md",
        f"dirty={dirty}  Dense-R18 gap(h2)={gap:.3f}  R18 cov@10={cov['resnet18']}",
        "",
        "hospital-2 (test):",
    ]
    for bb in BBS:
        acc_s = "NA" if acc[bb] is None else f"{acc[bb]:.4f}"
        lines.append(
            f"  {bb:14s} MSP={h2[bb]:.3f} (off {OFF[bb]:.3f}) id_val={acc_s} cov={cov[bb]}"
        )
    lines += ["", "hospital-1 (val, held-out):"]
    for bb in BBS:
        lines.append(f"  {bb:14s} MSP={h1[bb]:.3f}")
    lines += [
        "",
        f"R18 h2 bar={r18_h2}  R50 h2 bar={r50_h2}  R18 h1 bar={r18_h1}  cov_ok={cov_ok}",
        f"→ {branch}",
        "",
        "Do not mix stain_cov CSVs into rebuild_architecture_invariance.py.",
        "Do not skip CVPR today; decide after this readout.",
    ]
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
