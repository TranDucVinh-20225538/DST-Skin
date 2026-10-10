#!/usr/bin/env python3
"""T1 item E3 (CPU): label-free group recovery on real fit-set features with known groups.

    item_E3.py run NPROC   -> results/t1/E/raw/e3.jsonl (one line per unit = registry cell, both paper_2fold folds)
Per fold: fit-set patches (Camelyon: 4,000 stratified equally over the fold's 15 slides; other cells: all fit samples,
or 4,000 drawn uniformly without replacement when the fit set is larger); raw float64 features; M1 / M2 as in E2;
pair precision / recall, ARI (floor 0), k_hat vs G; rho_f = mean per-coordinate one-way ANOVA ICC, rho_f sqrt(PR/2),
rho_f sqrt(d/2). Lemma prediction: rho_f sqrt(d/2) >= 3 and G >= 60. Observed recoverability (as in E2): CR1 coverage
>= 0.90 of the mean of a synthetic flag score (theta 0.3, ICC 0.2 on the true groups, 500 replicates) using M1 clusters.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import item_E2 as E2  # noqa: E402

RAW = Path(os.environ.get("T1_RAW", str(CM.RES / "E" / "raw")))
NSUB, SCORE_REPS = 4000, 500


def anova_rho(X, g):
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    N, G = len(X), len(u)
    m = X.mean(0)
    M = np.zeros((G, X.shape[1]))
    np.add.at(M, inv, X)
    M = M / cnt[:, None]
    ssb = (cnt[:, None] * (M - m) ** 2).sum(0)
    sst = ((X - m) ** 2).sum(0)
    ssw = sst - ssb
    n0 = (N - (cnt ** 2).sum() / N) / (G - 1)
    msb, msw = ssb / (G - 1), ssw / (N - G)
    den = msb + (n0 - 1) * msw
    ok = den > 0
    return float(((msb - msw)[ok] / den[ok]).mean()), float(ssb.sum() / sst.sum())


def score_coverage(true, labs, rng):
    """Coverage of the CR1 CI with each label vector over SCORE_REPS synthetic flag scores on the true groups."""
    G = true.max() + 1
    ab = 1 / E2.RHO_Y - 1
    cov = {k: [] for k in labs}
    for _ in range(SCORE_REPS):
        m = rng.beta(E2.THETA * ab, (1 - E2.THETA) * ab, G)
        y = rng.binomial(1, m[true]).astype(float)
        for k, lab in labs.items():
            if k == "naive":
                mu = y.mean()
                cov[k].append(abs(mu - E2.THETA) <= 1.96 * np.sqrt(mu * (1 - mu) / len(y)))
            else:
                mu, hw = E2.ci_cluster(y, lab)
                cov[k].append(abs(mu - E2.THETA) <= hw if np.isfinite(hw) else np.nan)
    return {k: float(np.nanmean(np.array(v, float))) for k, v in cov.items()}


def one_unit(uid):
    from crossfit_ood import core as C
    t0 = time.time()
    cell = uid.split("|", 1)[1]
    reg = {r["cell_id"]: r for r in __import__("csv").DictReader(open(CM.RES / "I" / "registry.csv"))}[cell]
    d = CM.load_cell(cell, ["features_train", "groups_train", "features_id_eval", "groups_id_eval", "features_ood", "fold_train"])
    d["features_ood"] = d["features_ood"][:1]
    xtr, gtr, xid, gid, _, fm = CM.paper2fold_inputs(d, int(reg["seed"]))
    del d, xid
    tf = C._apply_map(fm[0], gtr)
    folds = []
    for f in (0, 1):
        rng = np.random.default_rng(CM.seed_seq("E", dict(unit=uid, fold=f), 0))
        idx = np.flatnonzero(tf == f)
        if reg["dataset"] == "camelyon":
            sl = np.unique(gtr[idx])
            per = np.full(len(sl), NSUB // len(sl))
            per[:NSUB - per.sum()] += 1
            idx = np.concatenate([rng.choice(idx[gtr[idx] == s], k, replace=False) for s, k in zip(sl, per)])
        elif len(idx) > NSUB:
            idx = rng.choice(idx, NSUB, replace=False)
        idx = np.sort(idx)
        X = np.asarray(xtr[idx], np.float64)
        _, true = np.unique(gtr[idx], return_inverse=True)
        N, dim, G = len(X), X.shape[1], int(true.max() + 1)
        rho_f, rho_tr = anova_rho(X, true)
        Xc = X - X.mean(0)
        ev = np.clip(np.linalg.eigvalsh(Xc.T @ Xc / (N - 1)), 0, None)
        pr = float(ev.sum() ** 2 / (ev ** 2).sum())
        sq = (X ** 2).sum(1)
        D = sq[:, None] + sq[None, :] - 2 * X @ X.T
        np.fill_diagonal(D, 0)
        labs = {"M1": E2.m1_labels(D), "M2": E2.m2_labels(X, D, rng), "oracle": true, "naive": None}
        row = dict(fold=f, N=N, d=dim, G=G, rho_f=rho_f, rho_f_trace=rho_tr, pr=pr,
                   lemma_stat_d=rho_f * np.sqrt(dim / 2), lemma_stat_pr=rho_f * np.sqrt(pr / 2))
        row["pred_recoverable"] = bool(row["lemma_stat_d"] >= 3 and G >= 60)
        row["pred_recoverable_pr"] = bool(row["lemma_stat_pr"] >= 3 and G >= 60)
        for k in ("M1", "M2"):
            if labs[k] is None:
                continue
            row[f"{k}_khat"] = int(len(np.unique(labs[k])))
            row.update({f"{k}_{s}": float(v) for s, v in E2.pair_scores(true, labs[k]).items()})
        cov = score_coverage(true, labs, rng)
        row.update({f"cov_{k}": v for k, v in cov.items()})
        row["obs_recoverable"] = bool(cov["M1"] >= 0.90)
        row["match"] = row["pred_recoverable"] == row["obs_recoverable"]
        folds.append(row)
        print(uid, f, f"{time.time() - t0:.0f}s", flush=True)
    return dict(unit=uid, cell=cell, dataset=reg["dataset"], backbone=reg["backbone"], folds=folds,
                match_both_folds=all(r["match"] for r in folds), seconds=time.time() - t0, precommit=CM.precommit_hash())


def run(nproc):
    RAW.mkdir(parents=True, exist_ok=True)
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_E.json"))
    order = [u for u in add["work_lists"][0]["execution_order"] if u.startswith("e3|")]
    out = RAW / "e3.jsonl"
    have = {json.loads(line)["unit"] for line in open(out)} if out.exists() else set()
    todo = [u for u in order if u not in have][:int(os.environ.get("T1_MAXUNITS", "100000000"))]
    with Pool(nproc, maxtasksperchild=4) as pool, open(out, "a") as fh:
        for res in pool.imap(one_unit, todo):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
    if "T1_MAXUNITS" not in os.environ:
        Path(str(out) + ".done").write_text("ok\n")


if __name__ == "__main__":
    run(int(sys.argv[2]))
