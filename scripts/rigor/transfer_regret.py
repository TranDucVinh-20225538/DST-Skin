#!/usr/bin/env python
"""H16: transfer regret of a copied score choice, over ALL pairs (CSV only, CPU, seconds).

The paper quotes one pair: ResNet-50's best logit score (ReAct) copied onto ConvNeXt costs
0.185 AUROC (seed 42). A reviewer will ask whether that pair was picked because it is the
worst. This script gives the full distribution.

regret(src -> tgt, family F) = max_{m in F} AUROC(tgt, m) - AUROC(tgt, argmax_{m in F} AUROC(src, m))
  kind cross_arch : src, tgt = two different archs at the same seed (all ordered pairs, every seed)
  kind cross_seed : src, tgt = two different seeds of the same arch (all ordered pairs, every arch)
  families        : nonfeature = MSP, Energy, ELogitNorm, ViM, ReAct (what the paper calls "logit")
                    feature    = Mahalanobis, kNN
                    all        = all 7

Outputs: outputs/reports/rigor_pack/transfer_regret/{regret_pairs.csv, regret_summary.csv, regret_log.txt}
REPRO  : ResNet-50 -> ConvNeXt, seed 42, nonfeature = 0.185 (tolerance 0.002).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rigor import common as C  # noqa: E402

FAMILIES = {
    "nonfeature": ("MSP", "Energy", "ELogitNorm", "ViM", "ReAct"),
    "feature": ("Mahalanobis", "kNN"),
    "all": tuple(C.METHODS_ORDER),
}


def regret(src: np.ndarray, tgt: np.ndarray, idx) -> float:
    s, t = src[list(idx)], tgt[list(idx)]
    if np.isnan(s).any() or np.isnan(t).any():
        return float("nan")
    return float(np.max(t) - t[int(np.argmax(s))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--reports", default=None)
    ap.add_argument("--domains", nargs="+", default=list(C.DOMAINS))
    args = ap.parse_args()
    root = Path(args.root)
    out = C.ensure_dir(Path(args.reports) if args.reports else C.default_reports(root) / "transfer_regret")
    archs, seeds = list(C.ARCHS), list(C.SEEDS)
    rows, log = [], ["transfer_regret  W source: %s" % C.W_SOURCE]
    repro_ok = True
    for domain in args.domains:
        cube = C.load_cube(root, domain, archs, seeds)  # (A, S, 7)
        for fam, ms in FAMILIES.items():
            idx = [C.METHODS_ORDER.index(m) for m in ms]
            for si, s in enumerate(seeds):
                for a in range(len(archs)):
                    for b in range(len(archs)):
                        if a != b:
                            rows.append({"domain": domain, "family": fam, "kind": "cross_arch", "seed": s,
                                         "src": archs[a], "tgt": archs[b],
                                         "regret": regret(cube[a, si], cube[b, si], idx)})
            for a in range(len(archs)):
                for i in range(len(seeds)):
                    for j in range(len(seeds)):
                        if i != j:
                            rows.append({"domain": domain, "family": fam, "kind": "cross_seed", "seed": seeds[j],
                                         "src": "%s@%d" % (archs[a], seeds[i]), "tgt": "%s@%d" % (archs[a], seeds[j]),
                                         "regret": regret(cube[a, i], cube[a, j], idx)})
        if domain == "camelyon17":
            r = [x for x in rows if x["domain"] == domain and x["family"] == "nonfeature" and x["kind"] == "cross_arch"
                 and x["seed"] == 42 and x["src"] == "resnet50" and x["tgt"] == "convnext_tiny"]
            v = r[0]["regret"] if r else float("nan")
            ok = bool(np.isfinite(v) and abs(v - 0.185) <= 0.002)
            repro_ok &= ok
            log.append("  REPRO %s ResNet-50 -> ConvNeXt seed42 nonfeature regret  published 0.185 recomputed %.4f"
                       % ("PASS" if ok else "FAIL", v))
    df = pd.DataFrame(rows)
    df.to_csv(out / "regret_pairs.csv", index=False)
    summ = []
    for (d, fam, kind), g in df.groupby(["domain", "family", "kind"]):
        x = g["regret"].dropna().values
        if not len(x):
            continue
        row = {"domain": d, "family": fam, "kind": kind, "n_pairs": len(x), "median": float(np.median(x)),
               "q90": float(np.quantile(x, 0.9)), "max": float(np.max(x)), "frac_gt_0.05": float(np.mean(x > 0.05)),
               "frac_zero": float(np.mean(x <= 1e-12))}
        if d == "camelyon17" and fam == "nonfeature" and kind == "cross_arch":
            x42 = g[g.seed == 42]["regret"].dropna().values
            row["pct_rank_of_0.185_seed42"] = float(np.mean(x42 <= 0.185)) if len(x42) else float("nan")
        summ.append(row)
    pd.DataFrame(summ).to_csv(out / "regret_summary.csv", index=False)
    log.append("Wrote regret_pairs.csv, regret_summary.csv. Read only against H16 in the precommit.")
    C.write_text(out / "regret_log.txt", log)
    print("\n".join(log))
    return 0 if repro_ok else 2


if __name__ == "__main__":
    sys.exit(main())
