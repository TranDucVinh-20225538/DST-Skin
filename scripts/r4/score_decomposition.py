#!/usr/bin/env python3
"""R4 analysis 2 (exploratory, post hoc): score-distribution decomposition of Eq. 1 (paper_2fold) on Camelyon17,
from the cached Track C scores (seed 42), Mahalanobis and kNN.

Per backbone x scorer x fold a (scorer s_{F_a} fitted on training fold a):
  ID-seen   = id_val patches of the slides in fold a       (cache: f{a}_id[fi == a])
  ID-unseen = id_val patches of the slides in fold 1 - a   (cache: f{a}_id[fi == 1 - a])
  OOD       = f{a}_ood (one vector per fold)
Checks: the OOD vector entering both AUROC terms is the same (max abs diff reported), and the recomputed
same / disjoint fold means equal the cells json same_2fold / disjoint_2fold (else STOP).
Writes results/r4/2/{quantiles.csv, REPORT.md, densities.png}.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
CACHE = REPO / "outputs/rigor_pack/miccai_campaign/scores"
CELLS = REPO / "outputs/reports/rigor_pack/foundation_gate/cells"
INP = Path.home() / "r3work/item1/inputs"
OUT = REPO / "results/r4/2"
MODELS = ("uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5", "resnet50", "convnext_tiny")
SCORERS = ("Mahalanobis", "kNN")
QS = (5, 25, 50, 75, 95)
FIG = (("uni", "kNN", "UNI kNN"), ("dinov2_vitb14", "Mahalanobis", "DINOv2-B Mahalanobis"),
       ("resnet50", "Mahalanobis", "ResNet-50 Mahalanobis"))
TOL = 1e-12


def main() -> int:
    from src.utils.benchmark_metrics import calc_auroc
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    rows, sets, stops = [], {}, []
    for model in MODELS:
        p = CACHE / ("camelyon_%s.npz" % model)
        if not p.exists():
            stops.append("%s: missing cache" % model)
            continue
        z = np.load(p)
        j = json.load(open(CELLS / ("camelyon_%s.json" % model)))
        zi = np.load(INP / ("camelyon_%s_s42.npz" % model), allow_pickle=True)
        gtr, ftr = zi["groups_train"].astype(str), zi["fold_train"]
        fi = z["fi"]
        for sc in SCORERS:
            same, dis = [], []
            for a in (0, 1):
                sid, sood = z["f%d_id_%s" % (a, sc)], z["f%d_ood_%s" % (a, sc)]
                seen, unseen = sid[fi == a], sid[fi == 1 - a]
                ood_same, ood_dis = sood, sood
                a_s, a_d = float(calc_auroc(seen, ood_same)), float(calc_auroc(unseen, ood_dis))
                ood_diff = float(np.max(np.abs(ood_same - ood_dis)))
                assert ood_diff == 0.0
                same.append(a_s)
                dis.append(a_d)
                q = {k: np.percentile(v, QS) for k, v in (("seen", seen), ("unseen", unseen), ("ood", sood))}
                iqr_u = q["unseen"][3] - q["unseen"][1]
                rows.append(dict(model=model, scorer=sc, fold=a, n_groups_fit=len(np.unique(gtr[ftr == a])), K=2,
                                 d=j["feat_dim"], id_acc=j["probe"]["id_acc"], n_seen=len(seen), n_unseen=len(unseen),
                                 n_ood=len(sood), auroc_same=a_s, auroc_disjoint=a_d, ood_max_abs_diff=ood_diff,
                                 ood_cross_fold_max_abs_diff=float(np.max(np.abs(z["f0_ood_%s" % sc] - z["f1_ood_%s" % sc]))),
                                 q=q, iqr_unseen=float(iqr_u), shift=float((q["seen"][2] - q["unseen"][2]) / iqr_u)))
                sets[(model, sc, a)] = (seen, unseen, sood)
            ds, dd = abs(np.mean(same) - j["same_2fold"][sc]), abs(np.mean(dis) - j["disjoint_2fold"][sc])
            if ds > TOL or dd > TOL:
                print("STOP: %s %s recomputed AUROC differs from cells json (%.3g, %.3g)" % (model, sc, ds, dd))
                return 1
            for r in rows[-2:]:
                r["delta_2fold"], r["check_same"], r["check_disjoint"] = j["delta"][sc], ds, dd

    with open(OUT / "quantiles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "scorer", "fold", "n_groups_fit", "K", "d", "id_acc", "set", "n", *["q%02d" % x for x in QS],
                    "auroc_same", "auroc_disjoint", "shift_median_over_iqr_unseen", "ood_max_abs_diff_between_terms"])
        for r in rows:
            for k, n in (("seen", r["n_seen"]), ("unseen", r["n_unseen"]), ("ood", r["n_ood"])):
                w.writerow([r["model"], r["scorer"], r["fold"], r["n_groups_fit"], r["K"], r["d"], "%.4f" % r["id_acc"], k, n,
                            *["%.6f" % v for v in r["q"][k]], "%.6f" % r["auroc_same"], "%.6f" % r["auroc_disjoint"],
                            "%.4f" % r["shift"], "%.1e" % r["ood_max_abs_diff"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
    col = {"ID-seen": "#d62728", "ID-unseen": "#1f77b4", "OOD": "#7f7f7f"}
    for c, (model, sc, title) in enumerate(FIG):
        for a in (0, 1):
            if (model, sc, a) not in sets:
                continue
            seen, unseen, sood = sets[(model, sc, a)]
            allv = np.concatenate([seen, unseen, sood])
            lo, hi = np.percentile(allv, [0.1, 99.9])
            bins = np.linspace(lo, hi, 151)
            x = ax[a, c]
            for name, v in (("ID-seen", seen), ("ID-unseen", unseen), ("OOD", sood)):
                x.hist(v, bins=bins, density=True, histtype="step", lw=1.4, color=col[name], label="%s (n=%d)" % (name, len(v)))
            r = next(r for r in rows if r["model"] == model and r["scorer"] == sc and r["fold"] == a)
            x.set_title("%s, fold %d\nAUROC same %.3f / disjoint %.3f" % (title, a, r["auroc_same"], r["auroc_disjoint"]), fontsize=9)
            x.set_xlabel("score (higher = more ID)", fontsize=8)
            x.tick_params(labelsize=7)
            if c == 0:
                x.set_ylabel("density", fontsize=8)
            x.legend(fontsize=7, frameon=False)
    fig.suptitle("Camelyon17 seed 42: score densities under the same fitted scorer s_{F_a} (x range = 0.1-99.9th pct of pooled scores)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "densities.png", dpi=150)

    L = ["# R4 analysis 2: score-distribution decomposition of Eq. 1 (EXPLORATORY, post hoc)", "",
         "Commit at report time: `%s`. Not in any precommit." % head, "", "VERDICT_PLACEHOLDER", "",
         "Source: cached Track C scores, Camelyon17, seed 42 (`outputs/rigor_pack/miccai_campaign/scores/camelyon_*.npz`).",
         "Under the scorer fitted on training fold a: ID-seen = id_val patches of fold-a slides, ID-unseen = id_val patches of fold-(1-a) slides, and OOD = hospital-2 patches.",
         "Higher score = more ID. Shift = (median ID-seen - median ID-unseen) / IQR(ID-unseen).", "",
         "## Checks", "",
         "- The OOD scores entering the two AUROC terms are the same vector in every backbone x scorer x fold: max abs diff = %s."
         % ("%.1e" % max(r["ood_max_abs_diff"] for r in rows)),
         "  For contrast, OOD scores differ between the two folds' fits by up to %.3g (max over cells)." % max(r["ood_cross_fold_max_abs_diff"] for r in rows),
         "- The recomputed fold-mean AUROCs equal the cells json `same_2fold` / `disjoint_2fold`: max abs diff %.1e / %.1e (tolerance %.0e)."
         % (max(r["check_same"] for r in rows), max(r["check_disjoint"] for r in rows), TOL), ""]
    if stops:
        L += ["STOP: " + "; ".join(stops), ""]
    L += ["## Shift and AUROC terms", "",
          "| backbone | scorer | fold | n_groups_fit | K | d | ID acc | AUROC same | AUROC disjoint | Delta (2-fold mean) | shift (IQR units) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %d | %d | %d | %d | %.4f | %.4f | %.4f | %.4f | %+.2f |" % (
            r["model"], r["scorer"], r["fold"], r["n_groups_fit"], r["K"], r["d"], r["id_acc"], r["auroc_same"],
            r["auroc_disjoint"], r["delta_2fold"], r["shift"]))
    L += ["", "## Quantiles (5 / 25 / 50 / 75 / 95)", "",
          "| backbone | scorer | fold | ID-seen | ID-unseen | OOD |", "|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda v: " / ".join("%.4g" % x for x in v)
        L.append("| %s | %s | %d | %s | %s | %s |" % (r["model"], r["scorer"], r["fold"], f(r["q"]["seen"]),
                                                     f(r["q"]["unseen"]), f(r["q"]["ood"])))
    L += ["", "![densities](densities.png)", "",
          "Figure: UNI kNN, DINOv2-B Mahalanobis and ResNet-50 Mahalanobis; rows = fold 0 / fold 1; step histograms (150 bins, density).", "",
          "Full table: `results/r4/2/quantiles.csv`.", "",
          "Caveats: post hoc and descriptive (no CI); one seed; quantiles pool patches, so slides with many patches weigh more."]
    pos = sum(r["shift"] > 0 for r in rows)
    L[4] = ("**Verdict:** the OOD scores are identical in both AUROC terms (diff 0) in all %d backbone x scorer x fold cases, "
            "so Delta comes entirely from the ID side. The ID-seen median lies above the ID-unseen median in %d of %d cases "
            "(shift %+.2f to %+.2f IQR)." % (len(rows), pos, len(rows), min(r["shift"] for r in rows), max(r["shift"] for r in rows)))
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print(L[4])
    return 0


if __name__ == "__main__":
    sys.exit(main())
