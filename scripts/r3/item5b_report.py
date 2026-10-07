#!/usr/bin/env python3
"""R3 item 5b REPORT.md from results/r3/5b/stepA.csv and results/r3/5b/stepB/*.json (criteria: PRECOMMIT.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
D = REPO / "results/r3/5b"
FMS = ("uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5")
FAIL = ("dinov2_vitb14", "dinov2_vitl14", "conch_v1_5")
SC = {"mahalanobis_l2": "Mahalanobis", "knn_mean_cosine": "kNN"}


def stepA(L):
    A = pd.read_csv(D / "stepA.csv")
    s42 = A[(A.seed == 42) & (A.scorer == "Mahalanobis") & A.fm.isin(FAIL)].set_index("fm")
    ok = {fm: bool(abs(s42.loc[fm, "F2_matched_minus_F1"]) <= 0.02 and abs(s42.loc[fm, "F2_naive_minus_F1"]) > 0.02)
          for fm in FAIL}
    explains = all(ok.values())
    v = ("naive pooling explains the Mahalanobis gap" if explains else
         "the Mahalanobis gap persists under matched pooling in %d of 3 failing cells (%s); matched pooling reduces it"
         % (sum(not x for x in ok.values()), ", ".join(f for f, x in ok.items() if not x)))
    L += ["## Step A: pooling check (cached Track A scores, no refits)", "", "Verdict: " + v + ".", "",
          "Rule (precommit): naive pooling explains the gap if, in each failing Mahalanobis cell at seed 42"
          " (DINOv2-B, DINOv2-L, CONCH), |F2_matched - F1| <= 0.02 and |F2_naive - F1| > 0.02.", "",
          "| FM | seed | scorer | leaky | F1 | F2_naive | F2_matched | F2_naive - F1 | F2_matched - F1 | n_id fold 0 / 1 |"
          " s.d. OOD fit 0 / fit 1 / averaged | corr(OOD fit 0, fit 1) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in A.iterrows():
        L.append("| %s | %d | %s | %.4f | %.4f | %.4f | %.4f | %+.4f | %+.4f | %d / %d | %.4g / %.4g / %.4g | %.3f |" % (
            r.fm, r.seed, r.scorer, r.auroc_leaky, r.F1, r.F2_naive, r.F2_matched, r.F2_naive_minus_F1,
            r.F2_matched_minus_F1, r.n_id_fold0, r.n_id_fold1, r.sd_ood_fit0, r.sd_ood_fit1, r.sd_ood_avg,
            r.corr_ood_fit0_fit1))
    L += ["", "F2_matched uses the same two arms as F1 (ID of fold k vs OOD, both scored by the fit that did not see"
          " fold k); it differs from F1 only in the arm weights (n_k vs 1/2), so F2_matched - F1 ="
          " sum_k (n_k / N - 1/2) AUROC_k. It is non-zero where the fold sizes are unequal and the two arm AUROCs"
          " differ (arm AUROCs in stepA.csv). n_groups_fit = 15 slides per fit, K = 2; d and probe ID accuracy as in"
          " the Step B table.", ""]
    return explains


def stepB(L):
    rows = []
    for fm in FMS:
        for sc in SC:
            p = D / "stepB" / ("%s_%s.json" % (fm, sc))
            pj = D / "stepB" / ("%s_%s_jk.json" % (fm, sc))
            if not p.exists():
                rows.append({"fm": fm, "scorer": SC[sc], "missing": True})
                continue
            j = json.loads(p.read_text())
            jk = json.loads(pj.read_text()) if pj.exists() else {}
            r = {"fm": fm, "scorer": SC[sc], "missing": False, "d": j["d"], "id_acc": j["probe_id_acc"]}
            for K in (2, 5, 10):
                k = j.get("K%d" % K)
                r["X%d" % K] = k["X_K"] if k else float("nan")
                r["leaky"] = k["auroc_leaky"] if k else r.get("leaky", float("nan"))
                r["fit%d" % K] = k["fit_patches_min_mean_max"] if k else None
                r["ng%d" % K] = k["n_groups_fit_min_max"] if k else None
                kj = jk.get("K%d" % K)
                r["ci%d" % K] = kj["X_K_ci"] if kj else None
            rows.append(r)
    B = pd.DataFrame(rows)
    ok = B[~B.missing]
    stab = {}
    for s in SC.values():
        g = ok[ok.scorer == s]
        n = int(((g.X5 - g.X10).abs() <= 0.01).sum()) if len(g) else 0
        stab[s] = (n, len(g))
    complete = all(v[1] == 5 for v in stab.values())
    stable = complete and all(v[0] >= 4 for v in stab.values())
    v = ("incomplete (%s)" % ", ".join("%s %d/5 cells" % (s, v[1]) for s, v in stab.items())) if not complete else \
        ("stable: |X_5 - X_10| <= 0.01 on %s; K = 5 becomes the package default" % ", ".join(
            "%s %d/5" % (s, v[0]) for s, v in stab.items()) if stable else
         "not stable: |X_5 - X_10| <= 0.01 on %s; recommend the largest affordable K and report the drift" % ", ".join(
             "%s %d/5" % (s, v[0]) for s, v in stab.items()))
    L += ["## Step B: fold-count stability (K = 2, 5, 10; folds stratified within hospital)", "", "Verdict: " + v + ".",
          "", "X_K = matched-pooled cross-fit AUROC (crossfit_ood, protocol default, fold_ids = slide fold of each id_val"
          " patch). Leaky = scorer fitted on all 30 slides. Criterion: |X_5 - X_10| <= 0.01 on >= 4/5 FMs for both"
          " scorers; |X_2 - X_5| is reported only.", "",
          "| FM | scorer | d | probe ID acc | leaky | X_2 | X_5 | X_10 | abs(X_5 - X_10) | abs(X_2 - X_5) |"
          " n_groups_fit K=2/5/10 | fit patches (min-max) K=2/5/10 | 95% CI X_5 / X_10 (jackknife) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in B.iterrows():
        if r.missing:
            L.append("| %s | %s | not run |" % (r.fm, r.scorer))
            continue

        def ng(x):
            return "%d-%d" % tuple(x) if x else "-"

        def fp(x):
            return "%d-%d" % (x[0], x[2]) if x else "-"

        def ci(x):
            return "[%.4f, %.4f]" % tuple(x) if x else "point only"
        L.append("| %s | %s | %d | %.3f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %s / %s / %s | %s / %s / %s | %s / %s |" % (
            r.fm, r.scorer, r.d, r.id_acc, r.leaky, r.X2, r.X5, r.X10, abs(r.X5 - r.X10), abs(r.X2 - r.X5),
            ng(r.ng2), ng(r.ng5), ng(r.ng10), fp(r.fit2), fp(r.fit5), fp(r.fit10), ci(r.ci5), ci(r.ci10)))
    L.append("")
    return stable, complete


def main() -> int:
    commit = sys.argv[1] if len(sys.argv) > 1 else "<commit>"
    L = ["# R3 item 5b: pooling check and fold-count stability", "", "commit: " + commit, "",
         "Post hoc relative to the paper's original precommit; criteria fixed in PRECOMMIT.json (fe65a1e) before"
         " Step B ran. Camelyon17, 30 training slides (3 hospitals x 10), 5 FMs, Track A scorers.", ""]
    explains = stepA(L)
    stable, complete = stepB(L)
    L += ["## Decision (precommit table)", "",
          "- Step A: " + ("paper warns that naive pooling inflates AUROC (up to the observed Mahalanobis gap); package"
                          " uses matched pooling." if explains else
                          "gap reported as unexplained; recommend group-disjoint evaluation for Mahalanobis."),
          "- Step B: " + ("pending (incomplete)." if not complete else
                          "K = 5 becomes the package default." if stable else
                          "recommend the largest affordable K and report the drift."), ""]
    (D / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
