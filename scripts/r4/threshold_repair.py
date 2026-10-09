#!/usr/bin/env python3
"""R4 analysis 1 (exploratory, precommit results/r4/1/PRECOMMIT.json): threshold repair by group-disjoint
calibration on Camelyon17, with slide-level false alarms.

  run MODEL   refit mahalanobis_l2 / knn_mean_cosine on F for 20 hospital-stratified F / C / T slide partitions,
              write results/r4/1/raw/MODEL.json (thresholds, coverage, per-slide false-alarm rates)
  report      aggregate raw/*.json into results/r4/1/{summary.csv, REPORT.md}
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
sys.path.insert(0, str(REPO))
W = Path.home() / "r3work/item1"
OUT = REPO / "results/r4/1"
MODELS = ("uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5", "resnet50", "convnext_tiny")
SCORERS = ("mahalanobis_l2", "knn_mean_cosine")
NP = 20
Q = 5.0


def partition(p, slides_by_hosp):
    rng = np.random.default_rng([4, p])
    roles = rng.permutation(3)
    sets = [[], [], []]
    for k, h in enumerate(sorted(slides_by_hosp)):
        s = np.array(sorted(slides_by_hosp[h]))
        if len(s) != 10:
            raise SystemExit("STOP: hospital %s has %d training slides, expected 10" % (h, len(s)))
        rng.shuffle(s)
        order = [int(roles[k])] + [j for j in range(3) if j != roles[k]]
        for j, part in zip(order, (s[:4], s[4:7], s[7:])):
            sets[j] += [str(x) for x in part]
    return sets


def slide_fa(score, gid, slides, t):
    return {s: float(np.mean(score[gid == s] < t)) for s in slides}


def run(model):
    import common as C
    from crossfit_ood import get_scorer
    f = W / "inputs" / ("camelyon_%s_s42.npz" % model)
    if not f.exists():
        raise SystemExit("STOP: missing %s" % f.name)
    meta = C.load_camelyon_metadata(REPO)
    hosp = {str(s): int(h) for s, h in zip(meta.slide, meta.center)}
    z = np.load(f, allow_pickle=True)
    xtr, gtr, xid, gid = (z[k] for k in ("features_train", "groups_train", "features_id_eval", "groups_id_eval"))
    gtr, gid = gtr.astype(str), gid.astype(str)
    if set(np.unique(gtr)) != set(np.unique(gid)):
        raise SystemExit("STOP: train / id_val slide sets differ")
    by_h = {}
    for s in np.unique(gtr):
        by_h.setdefault(hosp[s], []).append(s)
    parts = []
    for p in range(NP):
        F, Cs, T = partition(p, by_h)
        assert not (set(F) & set(Cs)) and not (set(F) & set(T)) and not (set(Cs) & set(T))
        fit = np.isin(gtr, F)
        mF, mC, mT = np.isin(gid, F), np.isin(gid, Cs), np.isin(gid, T)
        rec = dict(p=p, F=F, C=Cs, T=T, n_fit=int(fit.sum()), n_groups_fit=len(np.unique(gtr[fit])),
                   n_idval=dict(F=int(mF.sum()), C=int(mC.sum()), T=int(mT.sum())), scorers={})
        for sc in SCORERS:
            m = get_scorer(sc).fit(xtr[fit])
            s = np.asarray(m.score(xid), dtype=np.float64)
            if not np.all(np.isfinite(s)):
                raise SystemExit("STOP: non-finite scores %s %s p=%d" % (model, sc, p))
            r = {}
            for name, mask in (("leaky", mF), ("disjoint", mC)):
                t = float(np.percentile(s[mask], Q))
                r[name] = dict(t=t, cov_T=float(np.mean(s[mT] >= t)), cov_own=float(np.mean(s[mask] >= t)),
                               fa_T=slide_fa(s, gid, T, t))
            rec["scorers"][sc] = r
            print(model, p, sc, {k: round(v["cov_T"], 4) for k, v in r.items()}, flush=True)
        parts.append(rec)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    json.dump(dict(model=model, d=int(xtr.shape[1]), q=Q, partitions=parts),
              open(OUT / "raw" / ("%s.json" % model), "w"), indent=1)


def mr(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(x)), float(x.min()), float(x.max())


def fmt(t, k=3):
    return "%.*f [%.*f, %.*f]" % (k, t[0], k, t[1], k, t[2])


def report():
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    pre = subprocess.run(["git", "log", "-1", "--format=%h", "--", "results/r4/1/PRECOMMIT.json"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    rows, stops = [], []
    for model in MODELS:
        f = OUT / "raw" / ("%s.json" % model)
        if not f.exists():
            stops.append(model)
            continue
        R = json.load(open(f))
        cell = json.load(open(REPO / ("outputs/reports/rigor_pack/foundation_gate/cells/camelyon_%s.json" % model)))
        ng = sorted({p["n_groups_fit"] for p in R["partitions"]})
        for sc in SCORERS:
            row = dict(model=model, scorer=sc, n_groups_fit=ng[0] if len(ng) == 1 else ng, K="3-way x %d" % NP,
                       d=R["d"], id_acc=round(cell["probe"]["id_acc"], 4))
            for th in ("leaky", "disjoint"):
                per = [p["scorers"][sc][th] for p in R["partitions"]]
                fas = [np.array(list(x["fa_T"].values())) for x in per]
                row[th] = dict(cov_T=mr([x["cov_T"] for x in per]), cov_own=mr([x["cov_own"] for x in per]),
                               fa_med=mr([np.median(a) for a in fas]), gt10=mr([np.mean(a > 0.10) for a in fas]),
                               gt20=mr([np.mean(a > 0.20) for a in fas]), worst=mr([a.max() for a in fas]),
                               pooled=[float(v) for v in np.percentile(np.concatenate(fas), [5, 25, 50, 75, 95])])
            row["D"] = 0.93 <= row["disjoint"]["cov_T"][0] <= 0.97
            row["L"] = row["leaky"]["cov_T"][0] < 0.93
            rows.append(row)
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "scorer", "threshold", "n_groups_fit", "K", "d", "id_acc",
                    "cov_T_med", "cov_T_min", "cov_T_max", "cov_calib_med",
                    "slide_fa_median_med", "slide_fa_median_min", "slide_fa_median_max",
                    "frac_gt10_med", "frac_gt10_min", "frac_gt10_max", "frac_gt20_med", "frac_gt20_min", "frac_gt20_max",
                    "worst_med", "worst_min", "worst_max", "pooled_q05", "pooled_q25", "pooled_q50", "pooled_q75",
                    "pooled_q95", "pred_D", "pred_L"])
        for r in rows:
            for th in ("leaky", "disjoint"):
                x = r[th]
                w.writerow([r["model"], r["scorer"], th, r["n_groups_fit"], r["K"], r["d"], r["id_acc"],
                            *["%.6f" % v for v in x["cov_T"]], "%.6f" % x["cov_own"][0],
                            *["%.6f" % v for k in ("fa_med", "gt10", "gt20", "worst") for v in x[k]],
                            *["%.6f" % v for v in x["pooled"]], r["D"], r["L"]])
    nD, nL = sum(r["D"] for r in rows), sum(r["L"] for r in rows)
    nB = sum(r["D"] and r["L"] for r in rows)
    n = len(rows)
    verdict = ("SUPPORTED" if nB == n == 14 else "NOT SUPPORTED" if nB == 0 else "PARTLY SUPPORTED")
    L = ["# R4 analysis 1: threshold repair by group-disjoint calibration (EXPLORATORY)", "",
         "Commit at report time: `%s`. Precommit: `results/r4/1/PRECOMMIT.json` (committed `%s`, before any run)." % (head, pre), "",
         "**Verdict: prediction %s** (both parts hold in %d of %d cells; disjoint median coverage in [0.93, 0.97]: %d of %d; leaky median coverage < 0.93: %d of %d)."
         % (verdict, nB, n, nD, n, nL, n), ""]
    if stops:
        L += ["STOP (missing raw output): " + ", ".join(stops), ""]
    L += ["Camelyon17, seed 42, 30 training slides, 20 hospital-stratified slide partitions F / C / T (10 / 10 / 10 slides, 4 + 3 + 3 per hospital).",
          "The scorer is refit on F's training patches (cached frozen features; the cached Track C scores come from 2-fold fits and cannot be reused).",
          "Leaky threshold = 5th percentile of scores on F's id_val patches. Disjoint threshold = 5th percentile on C's id_val patches.",
          "Both are applied to T's id_val patches. Values are the median [min, max] over the 20 partitions.", "",
          "## (a) Patch-level realised coverage on T (target 0.95)", "",
          "| backbone | scorer | n_groups_fit | K | d | ID acc | leaky | disjoint | D | L |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %s | %s | %d | %.4f | %s | %s | %s | %s |" % (
            r["model"], r["scorer"], r["n_groups_fit"], r["K"], r["d"], r["id_acc"], fmt(r["leaky"]["cov_T"]),
            fmt(r["disjoint"]["cov_T"]), "yes" if r["D"] else "no", "yes" if r["L"] else "no"))
    L += ["", "D: 0.93 <= disjoint median <= 0.97. L: leaky median < 0.93. Coverage on the calibration slides themselves is 0.95 by construction "
          "(median %s to %s across cells)." % (
              "%.4f" % min(r[t]["cov_own"][0] for r in rows for t in ("leaky", "disjoint")),
              "%.4f" % max(r[t]["cov_own"][0] for r in rows for t in ("leaky", "disjoint"))), "",
          "## (b) Slide-level false alarms on T (10 T slides per partition)", "",
          "Per partition: median slide false-alarm rate, fraction of T slides with rate > 10% and > 20%, and the worst slide's rate.",
          "Each entry is the median [min, max] over 20 partitions. Pooled = 5/50/95th percentiles of all 200 slide rates.", "",
          "| backbone | scorer | threshold | median slide FA | frac > 10% | frac > 20% | worst slide | pooled q05 / q50 / q95 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for th in ("leaky", "disjoint"):
            x = r[th]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %.3f / %.3f / %.3f |" % (
                r["model"], r["scorer"], th, fmt(x["fa_med"]), fmt(x["gt10"], 2), fmt(x["gt20"], 2), fmt(x["worst"]),
                x["pooled"][0], x["pooled"][2], x["pooled"][4]))
    L += ["", "Per-partition values (slide lists, thresholds, per-slide rates): `results/r4/1/raw/*.json`; table: `results/r4/1/summary.csv`.", "",
          "Caveats: exploratory, one seed, one dataset; the 20 partitions share slides, so the ranges are not independent replicates."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:6]))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        report()
