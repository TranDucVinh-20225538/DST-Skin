#!/usr/bin/env python3
"""R3 item 8 / P2-c: Section 4.2 two-channel table rebuilt from the 18 balanced-fold retrains (CPU, no new runs).

Source: isbi_patch2 v2 retrains (resnet50 / convnext_tiny / densenet121 x seeds 42-44 x folds 0, 1; balanced
5 + 5 slides per training hospital, so each fold's backbone and scorer are fitted on 15 slides), scores from
scripts/rigor/isbi_patch2_scores.py (DST OODScorer = Track A definitions, float64). Per (arch, fold): median over
seeds of seen-slide AUROC, unseen-slide AUROC, Delta_bb = seen - unseen, and ID accuracy (seen / unseen slides).
Logit scores: Delta_bb = backbone channel. Feature scores: backbone + scorer-fit channels.
Old column: the earlier single run (logit_retrain_slide_disjoint, seed 42, fold 0; MSP / Energy only).

Writes results/r3/8/p2c/two_channel_balanced_{per_run,table}.csv and two_channel_balanced.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "outputs/reports/rigor_pack/leakage/logit_retrain_slide_disjoint_v2"
V1 = REPO / "outputs/reports/rigor_pack/leakage/logit_retrain_slide_disjoint"
OUT = REPO / "results/r3/8/p2c"
ARCHS, SEEDS, FOLDS = ("resnet50", "convnext_tiny", "densenet121"), (42, 43, 44), (0, 1)
SCORES = ("MSP", "Energy", "ReAct", "Mahalanobis", "kNN", "ViM")
EXTRA = ("ELogitNorm",)
D = {"resnet50": 2048, "convnext_tiny": 768, "densenet121": 1024}
NEAR = (0.45, 0.55)


def main() -> None:
    rows = []
    for a in ARCHS:
        for s in SEEDS:
            for f in FOLDS:
                sc = json.loads((V2 / ("scores_%s_s%d_f%d.json" % (a, s, f))).read_text())
                tr = json.loads((V2 / ("%s_s%d_f%d.json" % (a, s, f))).read_text())
                r = {"arch": a, "seed": s, "fold": f, "n_fit_patches": sc["n_fit"], "acc_seen": tr["acc_id_seen"],
                     "acc_unseen": tr["acc_id_unseen"], "n_id_seen": tr["n_id_seen"], "n_id_unseen": tr["n_id_unseen"]}
                for m in SCORES + EXTRA:
                    r["seen_" + m], r["unseen_" + m] = sc["auroc_%s_seen" % m], sc["auroc_%s_unseen" % m]
                    r["dbb_" + m] = r["seen_" + m] - r["unseen_" + m]
                rows.append(r)
    P = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    P.to_csv(OUT / "two_channel_balanced_per_run.csv", index=False)

    old = {}
    for a in ARCHS:
        p = V1 / ("%s_fold0.json" % a)
        if p.exists():
            o = json.loads(p.read_text())
            old[a] = {"acc_unseen": o["id_val_acc_disjoint"],
                      "MSP": o["auroc_msp_same_slides"] - o["auroc_msp_slide_disjoint"],
                      "Energy": o["auroc_energy_same_slides"] - o["auroc_energy_slide_disjoint"]}

    agg = []
    for a in ARCHS:
        for f in FOLDS:
            g = P[(P.arch == a) & (P.fold == f)]
            r = {"arch": a, "fold": f, "n_seeds": len(g), "n_groups_fit": 15, "K": 2, "d": D[a],
                 "acc_seen": g.acc_seen.median(), "acc_unseen": g.acc_unseen.median(),
                 "acc_unseen_min": g.acc_unseen.min()}
            for m in SCORES + EXTRA:
                r["seen_" + m] = g["seen_" + m].median()
                r["unseen_" + m] = g["unseen_" + m].median()
                r["dbb_" + m] = g["dbb_" + m].median()
                r["dbb_min_" + m], r["dbb_max_" + m] = g["dbb_" + m].min(), g["dbb_" + m].max()
                r["near_chance_" + m] = bool(all(NEAR[0] <= x <= NEAR[1] for x in (r["seen_" + m], r["unseen_" + m])))
                r["below_chance_" + m] = bool(max(r["seen_" + m], r["unseen_" + m]) < 0.5)
            agg.append(r)
    A = pd.DataFrame(agg)
    A.to_csv(OUT / "two_channel_balanced_table.csv", index=False)

    def f3(x):
        return "%.3f" % x

    def fs(x):
        return "%+.3f" % x

    L = ["# P2-c: two-channel table from the 18 balanced-fold retrains (R3 item 8, CPU, no new runs)", "",
         "Camelyon isbi_patch2 v2 retrains, 3 CNNs x seeds 42-44 x 2 balanced slide folds. Each fold's backbone and"
         " scorer are fitted on the same 15 training slides (n_groups_fit = 15, K = 2). Every entry is the median over"
         " the 3 seeds. Delta_bb = AUROC(seen-slide id_val vs hospital 2) - AUROC(unseen-slide id_val vs hospital 2),"
         " within one model. For the logit scores (MSP, Energy, ReAct) Delta_bb is the backbone channel; for the"
         " feature scores (Mahalanobis, kNN, ViM) it is backbone + scorer-fit channels. Scorers: DST OODScorer"
         " (Track A definitions, float64). ViM: not numerically reproducible across BLAS thread counts (R3 item 1)."
         " [min, max] over seeds in the second table.", "",
         "## Delta_bb (median over seeds), with ID accuracy", "",
         "| arch | fold | d | acc seen | acc unseen | " + " | ".join(SCORES) + " |",
         "|" + "---|" * (5 + len(SCORES))]
    for _, r in A.iterrows():
        cells = []
        for m in SCORES:
            t = fs(r["dbb_" + m])
            if r["near_chance_" + m]:
                t += " (nc)"
            if r["below_chance_" + m]:
                t += " (bc)"
            cells.append(t)
        L.append("| %s | %d | %d | %s | %s | %s |" % (r.arch, r.fold, r.d, f3(r.acc_seen), f3(r.acc_unseen),
                                                     " | ".join(cells)))
    L += ["", "(nc) near chance: seen and unseen AUROC both in [0.45, 0.55]. (bc) below chance: both < 0.5.", "",
          "## AUROC seen / unseen (median over seeds) and Delta_bb [min, max] over seeds", "",
          "| arch | fold | score | AUROC seen | AUROC unseen | Delta_bb | [min, max] |", "|---|---|---|---|---|---|---|"]
    for _, r in A.iterrows():
        for m in SCORES:
            L.append("| %s | %d | %s | %s | %s | %s | [%s, %s] |" % (
                r.arch, r.fold, m, f3(r["seen_" + m]), f3(r["unseen_" + m]), fs(r["dbb_" + m]),
                fs(r["dbb_min_" + m]), fs(r["dbb_max_" + m])))
    L += ["", "## Old single run (logit_retrain_slide_disjoint, seed 42, fold 0) beside the new fold-0 median", "",
          "| arch | score | old Delta_bb | new fold-0 median | old acc unseen | new fold-0 acc unseen |",
          "|---|---|---|---|---|---|"]
    for a in ARCHS:
        n0 = A[(A.arch == a) & (A.fold == 0)].iloc[0]
        for m in ("MSP", "Energy"):
            if a in old:
                L.append("| %s | %s | %s | %s | %s | %s |" % (a, m, fs(old[a][m]), fs(n0["dbb_" + m]),
                                                             f3(old[a]["acc_unseen"]), f3(n0.acc_unseen)))
            else:
                L.append("| %s | %s | not run | %s | - | %s |" % (a, m, fs(n0["dbb_" + m]), f3(n0.acc_unseen)))
    L += ["", "The old run scored MSP / Energy only; the other four scores have no old value. Old fold 0 is a"
          " different slide partition (unbalanced; 97,099 training patches vs 172,943 in balanced fold 0), so the"
          " old / new rows compare protocols, not the same slides. ELogitNorm is in two_channel_balanced_table.csv"
          " (not one of the six table scores)."]
    (OUT / "two_channel_balanced.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
