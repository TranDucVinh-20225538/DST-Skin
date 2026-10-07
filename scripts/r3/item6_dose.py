#!/usr/bin/env python3
"""R3 item 6: dose-response of the scorer-fit inflation on Camelyon (post-hoc).

Slides: the 30 training slides, per hospital shuffled (default_rng(0)) and cut 3 / 3 / 4 into three sets; the set
with the most training patches is the filler set C, the other two are A and B. Two folds (dose, unseen) = (A, B)
and (B, A). Per fold the fit-set size is fixed at N = all training patches of the dose set; at dose f a fraction f
of EACH dose slide's training patches enters the fit set and the remaining N - (dose part) patches are drawn
uniformly from C (default_rng([1, fold, 1000 f])). Delta(f) = AUROC(id_val of dose slides vs OOD) - AUROC(id_val
of unseen slides vs OOD), fold mean; OOD = hospital 2. Scorers = Track A definitions as in the item-1 rerun
(mahalanobis_l2: Ledoit-Wolf on L2-normalised features; knn_mean_cosine: mean cosine distance to k = 50 neighbours,
on the validated GPU implementation). ViM dropped (Track A ViM is not numerically reproducible).
95% CI: cluster bootstrap (B = 2000, default_rng(2)) over id_val slides and OOD images,
same draws for every f and scorer. Sanity: Delta(0) CI must contain 0 (else STOP for that model x scorer);
Delta(1) is listed beside Track A Delta_fit and the item-1 Delta for context only (different split, not a check).

Writes results/r3/6/{dose.csv, REPORT.md, dose.png}.
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
sys.path.insert(0, str(REPO / "scripts/r3"))
sys.path.insert(0, str(REPO))
W = Path.home() / "r3work/item1"
OUT = REPO / "results/r3/6"
MODELS = ("resnet50", "convnext_tiny", "uni", "dinov2_vitb14")
FS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
SCORERS = ("mahalanobis_l2", "knn_mean_cosine")
KEY = {"mahalanobis_l2": "Mahalanobis", "knn_mean_cosine": "kNN"}
NB = 2000


def slide_sets(slides, hosp_of, n_patches):
    rng = np.random.default_rng(0)
    sets = [[], [], []]
    for h in sorted(set(hosp_of.values())):
        s = np.array(sorted(x for x in slides if hosp_of[x] == h))
        rng.shuffle(s)
        for j, part in enumerate((s[:3], s[3:6], s[6:])):
            sets[j] += list(part)
    tot = [sum(n_patches[x] for x in st) for st in sets]
    c = int(np.argmax(tot))
    a, b = [sets[j] for j in range(3) if j != c]
    return a, b, sets[c]


def main() -> int:
    import common as C
    from crossfit_ood import get_scorer
    from medbench_scores import cluster_resample, wauroc
    from recompute_gpu_knn import GPUKNNMeanCosineScorer
    meta = C.load_camelyon_metadata(REPO)
    hosp = {str(s): int(h) for s, h in zip(meta.slide, meta.center)}
    cells = {r["cell"]: r for r in csv.DictReader(open(W / "cells_camelyon.csv"))}
    si = {r["model"]: r for r in csv.DictReader(open(REPO / "outputs/reports/rigor_pack/miccai_campaign/trackA_foundation/slide_identity.csv"))}
    rows = []
    for model in MODELS:
        cell = "camelyon_%s_s42" % model
        z = np.load(W / "inputs" / (cell + ".npz"), allow_pickle=True)
        xtr, gtr, xid, gid, xood = (z[k] for k in ("features_train", "groups_train", "features_id_eval",
                                                   "groups_id_eval", "features_ood"))
        gtr, gid = gtr.astype(str), gid.astype(str)
        npat = {s: int((gtr == s).sum()) for s in np.unique(gtr)}
        A, B, Cf = slide_sets(list(npat), hosp, npat)
        if sum(npat[s] for s in Cf) < max(sum(npat[s] for s in A), sum(npat[s] for s in B)):
            raise SystemExit("STOP: filler set smaller than a dose set")
        folds = ((0, A, B), (1, B, A))
        S = {}
        for fold, dose, unseen in folds:
            N = sum(npat[s] for s in dose)
            pool_c = np.flatnonzero(np.isin(gtr, Cf))
            for f in FS:
                rng = np.random.default_rng([1, fold, int(round(f * 1000))])
                part = np.concatenate([rng.permutation(np.flatnonzero(gtr == s))[:int(round(f * npat[s]))] for s in dose])
                fill = rng.choice(pool_c, N - len(part), replace=False)
                fit = np.concatenate([part, fill])
                for sc in SCORERS:
                    m = GPUKNNMeanCosineScorer(k=50) if sc == "knn_mean_cosine" else get_scorer(sc)
                    m.fit(xtr[fit])
                    S[(fold, f, sc)] = (m.score(xid), m.score(xood), len(part), len(fill),
                                        len(np.unique(gtr[fit])))
                    print(model, fold, f, sc, flush=True)
        seen_mask = {fold: np.isin(gid, dose) for fold, dose, _ in folds}
        unseen_mask = {fold: np.isin(gid, unseen) for fold, _, unseen in folds}
        id_mask = np.isin(gid, A + B)
        brng = np.random.default_rng(2)
        nood = len(xood)
        wb = [(cluster_resample(gid[id_mask], brng), np.bincount(brng.integers(0, nood, nood), minlength=nood).astype(float))
              for _ in range(NB)]
        for f in FS:
            for sc in SCORERS:
                pts, boots = [], np.zeros(NB)
                for fold, _, _ in folds:
                    si_, so_, n_part, n_fill, n_g = S[(fold, f, sc)]
                    o = np.argsort(so_, kind="mergesort")
                    ss = so_[o]
                    one = np.ones(nood)
                    cum1 = np.cumsum(one[o])
                    a_s = wauroc(si_[seen_mask[fold]], np.ones(seen_mask[fold].sum()), ss, cum1, cum1[-1])
                    a_u = wauroc(si_[unseen_mask[fold]], np.ones(unseen_mask[fold].sum()), ss, cum1, cum1[-1])
                    pts.append((a_s, a_u, n_part, n_fill, n_g))
                    sid = si_[id_mask]
                    sm, um = seen_mask[fold][id_mask], unseen_mask[fold][id_mask]
                    for bi, (wg, wo) in enumerate(wb):
                        cum = np.cumsum(wo[o])
                        boots[bi] += (wauroc(sid[sm], wg[sm], ss, cum, cum[-1]) -
                                      wauroc(sid[um], wg[um], ss, cum, cum[-1])) / 2
                d = float(np.mean([p[0] - p[1] for p in pts]))
                lo, hi = (float(np.percentile(boots, q)) for q in (2.5, 97.5))
                key = KEY[sc]
                r1 = W / "out_v2" / (cell + ("_knnmc" if sc == "knn_mean_cosine" else "_maha")) / "paper_ci.csv"
                item1 = next((float(r["delta"]) for r in csv.DictReader(open(r1)) if r["scorer"] == sc), None) \
                    if r1.exists() else None
                rows.append(dict(model=model, scorer=sc, f=f, delta=d, ci_lo=lo, ci_hi=hi,
                                 auroc_seen=float(np.mean([p[0] for p in pts])),
                                 auroc_unseen=float(np.mean([p[1] for p in pts])),
                                 n_fit=[p[2] + p[3] for p in pts], n_dose_patches=[p[2] for p in pts],
                                 n_groups_fit=[p[4] for p in pts], K=2, d=int(xtr.shape[1]),
                                 id_acc=cells[cell]["id_acc"], trackA_delta_fit=float(si[model]["dfit_%s" % key]),
                                 item1_delta=item1, slides_A=len(A), slides_B=len(B), slides_C=len(Cf),
                                 hospitals_per_set=[sorted({hosp[x] for x in st}) for st in (A, B, Cf)]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "dose.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report(rows)
    return 0


def report(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    sanity = {(r["model"], r["scorer"]): r["ci_lo"] <= 0 <= r["ci_hi"] for r in rows if r["f"] == 0.0}
    L = ["# R3 item 6: dose-response (Camelyon, post-hoc)", "", "commit: %s" % commit, "",
         "K = 2 folds (dose / unseen swapped); slides per set A / B / C = %d / %d / %d (C = filler)."
         % (rows[0]["slides_A"], rows[0]["slides_B"], rows[0]["slides_C"]), "",
         "Split stratified within hospital: training hospitals in sets A / B / C = %s (each hospital's slides cut"
         " 3 / 3 / 4), so Delta(f) does not mix in a hospital shift." % " / ".join(map(str, rows[0]["hospitals_per_set"])), "",
         "| model | d | ID acc | scorer | f | Delta | 95% CI | n_fit (fold 0, 1) | n_groups_fit (fold 0, 1) | Delta(0) CI contains 0 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ok = sanity[(r["model"], r["scorer"])]
        val = ("%+.4f" % r["delta"], "[%+.4f, %+.4f]" % (r["ci_lo"], r["ci_hi"])) if ok or r["f"] == 0.0 else ("STOP", "-")
        L.append("| %s | %d | %.3f | %s | %.2f | %s | %s | %s | %s | %s |" % (
            r["model"], r["d"], float(r["id_acc"]), r["scorer"], r["f"], val[0], val[1], r["n_fit"], r["n_groups_fit"],
            "yes" if ok else "NO -> STOP"))
    L += ["", "## Delta(1) beside Track A Delta_fit and item-1 Delta (for context only)", "",
          "Different split (three-way, fixed fit size) from Track A and item 1 (H11c 2-fold), so this is not a check.", "",
          "| model | scorer | Delta(1) | Track A Delta_fit | item-1 Delta |", "|---|---|---|---|---|"]
    for r in rows:
        if r["f"] == 1.0:
            L.append("| %s | %s | %+.4f | %+.4f | %s |" % (r["model"], r["scorer"], r["delta"], r["trackA_delta_fit"],
                                                          "%+.4f" % r["item1_delta"] if r["item1_delta"] is not None else "n/a"))
    n_ok = sum(sanity.values())
    L += ["", "![dose](dose.png)", "",
          "Verdict: Delta(0) CI contains 0 in %d/%d model x scorer rows (rows failing the sanity check are STOP)."
          % (n_ok, len(sanity)), "",
          "Caveats: post-hoc (not in precommit); three-way slide split (dose / unseen / filler, stratified within"
          " hospital) instead of the Track A 15 / 15 split, so Delta(1) is not the Track A estimand."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    fig, ax = plt.subplots(1, len(SCORERS), figsize=(5 * len(SCORERS), 3.6))
    for j, sc in enumerate(SCORERS):
        for m in MODELS:
            rr = [r for r in rows if r["model"] == m and r["scorer"] == sc]
            ax[j].errorbar([r["f"] for r in rr], [r["delta"] for r in rr],
                           yerr=[[r["delta"] - r["ci_lo"] for r in rr], [r["ci_hi"] - r["delta"] for r in rr]],
                           marker="o", capsize=3, label=m)
        ax[j].axhline(0, color="k", lw=0.5)
        ax[j].set_title(sc)
        ax[j].set_xlabel("dose f")
        ax[j].set_ylabel("Delta")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "dose.png", dpi=130)
    print("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
