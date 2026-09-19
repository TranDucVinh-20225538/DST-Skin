#!/usr/bin/env python3
"""Apply decision_precommit_matched_recipe.md. Official W stays standard-recipe seed 42."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

JUMP_DELTA = 0.15
RESNET_MEAN = 0.544654
JUMP_LINE = RESNET_MEAN + JUMP_DELTA
ID_FLOOR = 0.97
M1 = (
    ("convnext_tiny", None, "convnext_tiny"),
    ("mobilenet_v3_large", None, "mobilenet_v3_large"),
    ("regnet_y_3_2gf", None, "regnet_y_3_2gf"),
    ("efficientnet_v2_s", 224, "efficientnet_v2_s_224"),
)
M2 = (("resnet18", None, "resnet18"), ("resnet50", None, "resnet50"))
OFF_DIR = Path("outputs/reports/camelyon17/frac1/seed42")
M1_REP = Path("outputs/reports/camelyon17/frac1/matched_adam/seed42")
M1_MOD = Path("data/models/camelyon17/frac1/matched_adam/seed42")
M2_REP = Path("outputs/reports/camelyon17/frac1/matched_adamw/seed42")
M2_MOD = Path("data/models/camelyon17/frac1/matched_adamw/seed42")
OUT = Path("outputs/reports/camelyon_matched_recipe.txt")


def msp_of(report_dir: Path, stem: str) -> float:
    df = pd.read_csv(report_dir / f"{stem}_score_comparison.csv")
    col = "Method" if "Method" in df.columns else "method"
    return float(df.loc[df[col].astype(str).str.lower() == "msp", "AUROC"].iloc[0])


def id_acc(model_dir: Path, stem: str) -> float | None:
    path = model_dir / f"{stem}_train_meta.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["best_id_val_acc"])


def main() -> None:
    lines = [
        "Matched-recipe readout. Official W stays standard-recipe seed 42.",
        f"Frozen ResNet mean={RESNET_MEAN:.6f}  jump line={JUMP_LINE:.4f}  ID floor={ID_FLOOR}",
        "",
        "== M1 jumpers → ResNet Adam ==",
    ]
    clean_hits = []
    for _bb, _sz, stem in M1:
        msp = msp_of(M1_REP, stem)
        acc = id_acc(M1_MOD, stem)
        jump = msp - RESNET_MEAN
        dirty = acc is not None and acc < ID_FLOOR
        hit = jump >= JUMP_DELTA
        flag = "DIRTY" if dirty else ("JUMP" if hit else "no")
        acc_s = "NA" if acc is None else f"{acc:.4f}"
        lines.append(
            f"  {stem:24s} MSP={msp:.3f} jump={jump:+.3f} id_val={acc_s} {flag}"
        )
        if not dirty:
            clean_hits.append(hit)
    n_clean = len(clean_hits)
    n_hit = int(sum(clean_hits))
    if n_clean == 0:
        m1 = "all dirty — ablation uninterpretable; #6 stays; official A unchanged"
    elif n_hit == 0:
        m1 = "0/4 jump — 6/6 was recipe-inflated; drop '4 families beyond EfficientNet'; A lives on DenseNet"
    elif n_hit >= 3:
        m1 = "≥3/4 still jump — recipe does not explain 6/6; keep 4-family sentence"
    else:
        m1 = f"{n_hit}/{n_clean} jump — mixed; §4.1 is DenseNet + survivors, not 6/6"
    lines += [f"clean hits={n_hit}/{n_clean}", f"→ {m1}", "", "== M2 ResNet → ConvNeXt AdamW =="]
    m2_jump = []
    for _bb, _sz, stem in M2:
        msp = msp_of(M2_REP, stem)
        acc = id_acc(M2_MOD, stem)
        dirty = acc is not None and acc < ID_FLOOR
        hit = msp >= JUMP_LINE
        flag = "DIRTY" if dirty else ("JUMP-BAND" if hit else "no-jump-band")
        acc_s = "NA" if acc is None else f"{acc:.4f}"
        lines.append(f"  {stem:24s} MSP={msp:.3f} id_val={acc_s} {flag}")
        if not dirty:
            m2_jump.append(hit)
    if not m2_jump:
        m2 = "both dirty — do not interpret"
    elif any(m2_jump):
        m2 = "ResNet can jump under AdamW — cannot write pure architecture split"
    else:
        m2 = "ResNet no-jump survives ConvNeXt recipe"
    lines += [f"→ {m2}", "", "Do not mix matched CSVs into rebuild_architecture_invariance.py."]
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
