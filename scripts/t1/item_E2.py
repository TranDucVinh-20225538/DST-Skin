#!/usr/bin/env python3
"""T1 item E2: label-free group recovery (CPU). Features x = sqrt(rho_f) u_g + sqrt(1-rho_f) e (d dims), flag score
y ~ Bernoulli(m_g), m_g ~ Beta with ICC 0.2, theta = 0.3 (as bundle s1c), G = 60 equal groups of n.

    item_E2.py run TASK NTASKS NPROC   -> results/t1/E/raw/e2_<task>.jsonl
M1 = average linkage cut at the midpoint of the median NN and median pairwise squared distances (s1c, 500 replicates);
M2 = k-means (k-means++, 1 init) with k by maximum silhouette over 30 geometrically spaced k in [2, N/3] (first 100
replicates); M3 = oracle labels. CI: CR1 cluster-robust, t_{G-1} quantile (s1c ci_cluster); naive: binomial iid.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

RAW = Path(os.environ.get("T1_RAW", str(CM.RES / "E" / "raw")))
REPS, REPS_M2, THETA, RHO_Y, NK_M2 = 500, 100, 0.3, 0.2, 30


def m1_labels(D):
    nn = np.min(D + np.diag(np.full(len(D), np.inf)), 1)
    iu = np.triu_indices(len(D), 1)
    thr = 0.5 * (np.median(nn) + np.median(D[iu]))
    Z = linkage(np.maximum(D[iu], 0), method="average")
    return fcluster(Z, thr, criterion="distance")


def m2_labels(X, D, rng):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    N = len(X)
    ks = np.unique(np.round(np.geomspace(2, N // 3, NK_M2)).astype(int))
    Ds = np.sqrt(np.maximum(D, 0))
    np.fill_diagonal(Ds, 0)
    best, lab_best = -np.inf, None
    for k in ks:
        lab = KMeans(k, n_init=1, random_state=int(rng.integers(2**31))).fit_predict(X)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(Ds, lab, metric="precomputed")
        if s > best:
            best, lab_best = s, lab
    return lab_best


def ci_cluster(y, lab, alpha=0.05):
    ks, inv = np.unique(lab, return_inverse=True)
    mu, N, G = y.mean(), len(y), len(ks)
    r = np.bincount(inv, weights=y - mu)
    se = np.sqrt(G / (G - 1) * (r ** 2).sum()) / N if G > 1 else np.inf
    return mu, tdist.ppf(1 - alpha / 2, max(G - 1, 1)) * se


def pair_scores(true, lab):
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.cluster import pair_confusion_matrix
    C = pair_confusion_matrix(true, lab)
    tp, fn, fp = C[1, 1], C[1, 0], C[0, 1]
    return dict(prec=tp / (tp + fp) if tp + fp else np.nan, rec=tp / (tp + fn), ari=max(0.0, adjusted_rand_score(true, lab)))


def one_rep(args):
    uid, p, rep = args
    rng = np.random.default_rng(CM.seed_seq("E", uid, rep))
    G, n, d, rf = p["G"], p["n"], p["d"], p["rho_f"]
    u = rng.standard_normal((G, d)) * np.sqrt(rf)
    X = np.repeat(u, n, 0) + rng.standard_normal((G * n, d)) * np.sqrt(1 - rf)
    ab = 1 / RHO_Y - 1
    m = rng.beta(THETA * ab, (1 - THETA) * ab, G)
    y = rng.binomial(1, np.repeat(m, n)).astype(float)
    true = np.repeat(np.arange(G), n)
    sq = (X ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.fill_diagonal(D, 0)
    mu = y.mean()
    se = np.sqrt(mu * (1 - mu) / len(y))
    out = dict(rep=rep, cov_naive=float(abs(mu - THETA) <= 1.96 * se))
    mo, ho = ci_cluster(y, true)
    out["cov_oracle"] = float(abs(mo - THETA) <= ho)
    methods = {"M1": m1_labels(D)}
    if rep < REPS_M2:
        methods["M2"] = m2_labels(X, D, rng)
    for k, lab in methods.items():
        if lab is None:
            continue
        me, he = ci_cluster(y, lab)
        out[f"{k}_cov"] = float(abs(me - THETA) <= he) if np.isfinite(he) else np.nan
        out[f"{k}_khat_over_k"] = len(np.unique(lab)) / G
        for s, v in pair_scores(true, lab).items():
            out[f"{k}_{s}"] = float(v)
    return out


def run(task, ntasks, nproc):
    RAW.mkdir(parents=True, exist_ok=True)
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_E.json"))
    order = [u for u in add["work_lists"][0]["execution_order"] if u.startswith("e2|")]
    params = add["unit_params"]
    mine = [u for i, u in enumerate(order) if i % ntasks == task]
    out = RAW / f"e2_{task}.jsonl"
    have = {json.loads(line)["unit"] for line in open(out)} if out.exists() else set()
    pre = CM.precommit_hash()
    with Pool(nproc) as pool, open(out, "a") as fh:
        for uid in mine:
            if uid in have:
                continue
            t0 = time.time()
            p = params[uid]
            rows = pool.map(one_rep, [(uid, p, r) for r in range(int(os.environ.get("T1_E2_REPS", REPS)))], chunksize=1)
            res = dict(unit=uid, params=p, precommit=pre, reps=len(rows), lemma_stat=p["rho_f"] * np.sqrt(p["d"] / 2),
                       lemma_pred=bool(p["rho_f"] * np.sqrt(p["d"] / 2) >= 3 and p["G"] >= 60), seconds=time.time() - t0)
            for k in sorted({k for r in rows for k in r if k != "rep"}):
                v = np.array([r[k] for r in rows if k in r], float)
                res[k] = float(np.nanmean(v))
                res[k + "|n"] = int(np.isfinite(v).sum())
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(uid, f"{res['seconds']:.0f}s M1cov={res.get('M1_cov')} khat/k={res.get('M1_khat_over_k')}", flush=True)
    if "T1_E2_REPS" not in os.environ:
        Path(str(out) + ".done").write_text("ok\n")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
