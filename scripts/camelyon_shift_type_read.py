#!/usr/bin/env python3
"""Apply decision_precommit_shift_type.md read table. Do not rewrite official W."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

JUMP_DELTA = 0.15
SAT_MEDIAN = 0.95
SAT_MIN = 0.90
OFFICIAL = (
    "resnet18",
    "resnet50",
    "densenet121",
    "convnext_tiny",
    "mobilenet_v3_large",
    "regnet_y_3_2gf",
    "effb3",
    "efficientnet_v2_s_224",
)
NON_RESNET = [b for b in OFFICIAL if b not in ("resnet18", "resnet50")]
C_DIR = Path("outputs/reports/camelyon17/frac1/seed42")
NEW_DIR = C_DIR / "shift_type"
OUT = Path("outputs/reports/camelyon_shift_type.txt")


def msp_csv(path: Path) -> float:
    df = pd.read_csv(path)
    col = "Method" if "Method" in df.columns else "method"
    return float(df.loc[df[col].astype(str).str.lower() == "msp", "AUROC"].iloc[0])


def arm_msp(arm: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for stem in OFFICIAL:
        if arm == "C":
            path = C_DIR / f"{stem}_score_comparison.csv"
        else:
            path = NEW_DIR / f"{stem}_{arm}_score_comparison.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        out[stem] = msp_csv(path)
    return out


def classify(msp: dict[str, float]) -> dict:
    vals = np.array([msp[s] for s in OFFICIAL], dtype=float)
    saturated = bool(np.median(vals) >= SAT_MEDIAN or np.min(vals) >= SAT_MIN)
    rmean = 0.5 * (msp["resnet18"] + msp["resnet50"])
    jumps = {s: msp[s] - rmean for s in OFFICIAL}
    n_hit = int(sum(jumps[s] >= JUMP_DELTA for s in NON_RESNET))
    mean_jump = float(np.mean([jumps[s] for s in NON_RESNET]))
    if saturated:
        branch = "saturated; arm dropped"
    elif n_hit == 6:
        branch = "6/6 jump — not covariate-specific; do not claim shift-conditional law"
    elif n_hit == 0 and mean_jump < JUMP_DELTA:
        branch = "0/6 jump — HIT vs hospital-2 if this arm is T or F"
    else:
        branch = f"{n_hit}/6 jump — mixed; report, not a law"
    return {
        "saturated": saturated,
        "resnet_mean": rmean,
        "n_hit": n_hit,
        "mean_jump_6": mean_jump,
        "median_msp": float(np.median(vals)),
        "min_msp": float(np.min(vals)),
        "jumps": jumps,
        "msp": msp,
        "branch": branch,
    }


def joint(t: dict, f: dict) -> str:
    def bucket(d: dict) -> str:
        if d["saturated"]:
            return "sat"
        if d["n_hit"] == 6:
            return "6/6"
        if d["n_hit"] == 0 and d["mean_jump_6"] < JUMP_DELTA:
            return "0/6"
        return "mixed"

    key = (bucket(t), bucket(f))
    table = {
        ("0/6", "0/6"): "MSP-jump is hospital-shift specific on this zoo.",
        ("0/6", "sat"): "MSP-jump is hospital-shift specific; far-OOD saturated (expected).",
        ("6/6", "6/6"): "classifiers, not shift type. Do not sell covariate vs semantic.",
        ("0/6", "6/6"): "odd split (task no / far yes). Report; no slogan.",
        ("6/6", "0/6"): "jump survives other H&E; far-OOD does not. Not the v1.5 counter.",
        ("6/6", "sat"): "jump survives other H&E; far-OOD uninformative. Not the v1.5 counter.",
        ("sat", "sat"): "both arms saturated. Experiment failed to raise the ceiling. Stop.",
    }
    return table.get(key, f"mixed or leftover ({key[0]} T, {key[1]} F). No law. Official A unchanged.")


def fmt_arm(name: str, d: dict) -> list[str]:
    lines = [
        f"== arm {name} ==",
        f"median MSP={d['median_msp']:.3f} min={d['min_msp']:.3f} ResNet mean={d['resnet_mean']:.3f}",
        f"non-ResNet mean jump={d['mean_jump_6']:.3f}  hits={d['n_hit']}/6",
        f"→ {d['branch']}",
    ]
    for s in OFFICIAL:
        flag = "JUMP" if d["jumps"][s] >= JUMP_DELTA else "no"
        lines.append(f"  {s:24s} MSP={d['msp'][s]:.3f}  jump={d['jumps'][s]:+.3f}  {flag}")
    return lines


def main() -> None:
    c = classify(arm_msp("C"))
    t = classify(arm_msp("T"))
    f = classify(arm_msp("F"))
    lines = [
        "Camelyon shift-type readout. Official W stays hospital-2 seed 42.",
        "decision_precommit_shift_type.md",
        "",
        *fmt_arm("C covariate (frozen)", c),
        "",
        *fmt_arm("T task-shift MIDOG 1a", t),
        "",
        *fmt_arm("F far-OOD CIFAR-10 test", f),
        "",
        "Joint:",
        joint(t, f),
        "",
        "Do not mix these CSVs into rebuild_architecture_invariance.py.",
    ]
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
