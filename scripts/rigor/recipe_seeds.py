#!/usr/bin/env python
"""H15: do the matched-recipe results (M1, M2) survive other seeds? (reads score CSVs only)

The paper's M1 (four non-ResNet nets trained with ResNet-Adam) and M2 (ResNets trained with
ConvNeXt-AdamW) are seed 42 only. `run_rigor_pack.sh --recipe-seeds` trains seeds 43-46
(same commands as scripts/submit_camelyon_matched_recipe.sh). This script reads:
  outputs/reports/camelyon17/frac1/matched_adam/seed{S}/{stem}_score_comparison.csv   (M1)
  outputs/reports/camelyon17/frac1/matched_adamw/seed{S}/{stem}_score_comparison.csv  (M2)
Jump of a run = its MSP minus the OFFICIAL ResNet mean of the same seed (as the paper's M1
uses the frozen official seed-42 ResNet mean 0.545). Bar 0.15. Also reported: MSP >= 0.6947.
Output: outputs/reports/rigor_pack/recipe_seeds/{recipe_seeds.csv, recipe_seeds_summary.txt}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rigor import common as C  # noqa: E402

M1 = ("convnext_tiny", "mobilenet_v3_large", "regnet_y_3_2gf", "efficientnet_v2_s")
M2 = ("resnet18", "resnet50")
TAGS = {"M1": ("matched_adam", M1), "M2": ("matched_adamw", M2)}
# seed-42 MSP published in the paper (Sec. "Optimizer challenges ..."): reproduction check
PUBLISHED42 = {("M1", "convnext_tiny"): 0.820, ("M1", "mobilenet_v3_large"): 0.737, ("M1", "regnet_y_3_2gf"): 0.805,
               ("M1", "efficientnet_v2_s"): 0.873, ("M2", "resnet18"): 0.773, ("M2", "resnet50"): 0.581}


def msp_from_csv(path: Path) -> float:
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path)
    r = df[df["Method"] == "msp"]
    return float(r["AUROC"].iloc[0]) if len(r) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--reports", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out = C.ensure_dir(Path(args.reports) if args.reports else C.default_reports(root) / "recipe_seeds")
    mi = C.METHODS_ORDER.index("MSP")
    rmean = {s: float(np.nanmean([C.metric_vector(root, "camelyon17", r, s)[mi] for r in C.RESNETS])) for s in C.SEEDS}
    rows = []
    for exp, (tag, archs) in TAGS.items():
        for a in archs:
            stem = C.file_stem(a, "camelyon17")
            for s in C.SEEDS:
                p = root / "outputs/reports/camelyon17/frac1" / tag / ("seed%d" % s) / ("%s_score_comparison.csv" % stem)
                m = msp_from_csv(p)
                rows.append({"exp": exp, "arch": a, "seed": s, "msp": m, "official_resnet_mean": rmean[s],
                             "jump": m - rmean[s], "jump_ge_0.15": bool(np.isfinite(m) and m - rmean[s] >= C.JUMP_DELTA),
                             "msp_ge_0.6947": bool(np.isfinite(m) and m >= 0.6947)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "recipe_seeds.csv", index=False)
    lines = ["H15 matched-recipe seeds (jump = MSP - official ResNet mean of the same seed; bar 0.15)"]
    for (exp, a), g in df.groupby(["exp", "arch"], sort=False):
        n = int(g.msp.notna().sum())
        hits = int(g["jump_ge_0.15"].sum())
        if n < 5:
            v = "INCOMPLETE (%d/5 seeds) - run `bash run_rigor_pack.sh --recipe-seeds`" % n
        elif hits >= 4:
            v = "robust (>=4/5)"
        elif hits == 3:
            v = "seed-dependent (3/5)"
        else:
            v = "seed-42 result not robust (<=2/5)"
        lines.append("  %s %-20s seeds %d/5  jump>=0.15: %d  -> %s" % (exp, a, n, hits, v))
    ok_all = True
    for (exp, a), pub in PUBLISHED42.items():
        r = df[(df.exp == exp) & (df.arch == a) & (df.seed == 42)]
        v = float(r.msp.iloc[0]) if len(r) else float("nan")
        ok = bool(np.isfinite(v) and abs(v - pub) <= 0.0015)
        ok_all &= ok
        lines.append("  REPRO %s %s %s seed42 MSP  published %.3f recomputed %.4f" % ("PASS" if ok else "FAIL", exp, a, pub, v))
    C.write_text(out / "recipe_seeds_summary.txt", lines)
    print("\n".join(lines))
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
