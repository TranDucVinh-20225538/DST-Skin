#!/usr/bin/env python3
"""R3 item 4: anisotropic synthetic test, as fixed in results/r3/4/PRECOMMIT.json (written before any run).

Phases (one process pool each; BLAS forced to 1 thread per worker):
  calibrate : delta per d (OOD shift; sigma_b = 0, exact-covariance Mahalanobis disjoint AUROC ~ 0.85, n/d = 10)
              and sigma_b per (d, placement) (slide-ID probe balanced accuracy 0.90 +/- 0.02, mean of 5 draws)
              -> <out>/calibration.json
  run       : 50 replicates per (d, n/d, placement), crossfit_auroc(protocol='paper_2fold') with
              lw_maha / exact_maha / knn / vim -> <out>/runs.jsonl
  analyze   : results/r3/4/{cells.csv, contrasts.csv, REPORT.md}
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CF = Path.home() / "handoff/crossfit-ood_v3"
sys.path.insert(0, str(CF / "scripts"))
from paper2fold_designs import design  # noqa: E402

from crossfit_ood import crossfit_auroc, get_scorer  # noqa: E402
from crossfit_ood.scorers import ViMScorer  # noqa: E402

DS = (768, 2560)
RATIOS = (1, 10)
PLACEMENTS = ("high", "low")
K_DIRS = 16
N_REP = 50
PROBE_TARGET, PROBE_TOL = 0.90, 0.02
AUROC_TARGET = 0.85
FULL_FOLD_MEAN = 302_436 / 2
RES = REPO / "results/r3/4"


class ExactMaha:
    """Empirical (MLE) covariance precision; ridge eps = 1e-3 tr(S)/d when n_fit <= d."""

    def fit(self, x, labels=None):
        x = np.asarray(x, np.float64)
        self.mu_ = x.mean(0)
        xc = x - self.mu_
        S = xc.T @ xc / len(x)
        if len(x) <= x.shape[1]:
            S = S + 1e-3 * np.trace(S) / x.shape[1] * np.eye(x.shape[1])
        self.P_ = np.linalg.inv(S)
        return self

    def score(self, x):
        d = np.asarray(x, np.float64) - self.mu_
        return -np.sum((d @ self.P_) * d, axis=1)


def population(d):
    rng = np.random.default_rng([20261007, d])
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = 1.0 / np.arange(1, d + 1)
    lam *= d / lam.sum()
    v = rng.standard_normal(d)
    return U, lam, v / np.linalg.norm(v)


_POP = {}


def pop(d):
    if d not in _POP:
        _POP[d] = population(d)
    return _POP[d]


def noise(rng, n, d):
    U, lam, _ = pop(d)
    return (rng.standard_normal((n, d)) * np.sqrt(lam)) @ U.T


def offsets(rng, n_groups, d, placement, sigma_b):
    U, lam, _ = pop(d)
    S = np.arange(K_DIRS) if placement == "high" else np.arange(d - K_DIRS, d)
    return sigma_b * (rng.standard_normal((n_groups, K_DIRS)) * np.sqrt(lam[S])) @ U[:, S].T


def sizes(d, r):
    D = design("camelyon", scale=r * d / FULL_FOLD_MEAN)
    return D["train_sizes"], D["eval_sizes"], D["ood_sizes"]


def make(d, r, placement, sigma_b, delta, seed):
    tr, ev, oo = sizes(d, r)
    rng = np.random.default_rng([seed, d, r])  # common random numbers across placements
    mu = offsets(rng, len(tr), d, placement, sigma_b)
    gtr, gid = np.repeat(np.arange(len(tr)), tr), np.repeat(np.arange(len(ev)), ev)
    xtr = mu[gtr] + noise(rng, len(gtr), d)
    xid = mu[gid] + noise(rng, len(gid), d)
    xood = delta * pop(d)[2] + noise(rng, int(oo.sum()), d)
    return xtr, gtr, xid, gid, xood


def probe_bacc(d, placement, sigma_b, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng([seed, d, 77])
    mu = offsets(rng, 30, d, placement, sigma_b)
    g = np.repeat(np.arange(30), 400)
    X = mu[g] + noise(rng, len(g), d)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    tr = np.tile(np.r_[np.ones(300, bool), np.zeros(100, bool)], 30)
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), g[tr])
    return float(balanced_accuracy_score(g[~tr], clf.predict(sc.transform(X[~tr]))))


def _probe_task(a):
    return probe_bacc(*a)


def _auroc_task(a):
    d, delta, seed = a
    xtr, gtr, xid, gid, xood = make(d, 10, "high", 0.0, delta, seed)
    rep = crossfit_auroc(xtr, gtr, xid, gid, xood, scorers={"exact_maha": ExactMaha()}, protocol="paper_2fold",
                         uncertainty="bootstrap", n_bootstrap=0, random_state=seed)
    return rep.results["exact_maha"].auroc_crossfit


def _search(f, target, tol, hi, log, key, max_hi=1e4, iters=18):
    """Bisection on a monotone increasing f over [0, hi]; hi doubled until f(hi) >= target."""
    while True:
        v = f(hi)
        log.append({**key, "x": hi, "f": v})
        print(key, "%.4f -> %.4f" % (hi, v), flush=True)
        if v >= target or hi >= max_hi:
            break
        hi *= 2
    lo, x = 0.0, hi
    for _ in range(iters):
        if abs(v - target) <= tol:
            break
        x = (lo + hi) / 2
        v = f(x)
        log.append({**key, "x": x, "f": v})
        print(key, "%.4f -> %.4f" % (x, v), flush=True)
        lo, hi = (x, hi) if v < target else (lo, x)
    return x, v


def calibrate(out, workers):
    cal = {"delta": {}, "sigma_b": {}, "probe": {}, "auroc_at_delta": {}, "log": []}
    with Pool(workers) as pool:
        for d in DS:
            fa = lambda x: float(np.mean(pool.map(_auroc_task, [(d, x, 30_000 + s) for s in range(3)])))  # noqa: E731
            x, v = _search(fa, AUROC_TARGET, 0.005, 4.0, cal["log"], {"d": d, "what": "delta"})
            cal["delta"][str(d)], cal["auroc_at_delta"][str(d)] = x, v
            for pl in PLACEMENTS:
                fp = lambda s: float(np.mean(pool.map(_probe_task, [(d, pl, s, 20_000 + i) for i in range(5)])))  # noqa: E731
                x, v = _search(fp, PROBE_TARGET, PROBE_TOL / 4, 1.0, cal["log"], {"d": d, "what": "sigma_b_" + pl})
                cal["sigma_b"]["%d_%s" % (d, pl)], cal["probe"]["%d_%s" % (d, pl)] = x, v
            Path(out, "calibration.json").write_text(json.dumps(cal, indent=1))
    Path(out, "calibration.json").write_text(json.dumps(cal, indent=1))


def _run_task(a):
    d, r, pl, sigma_b, delta, seed = a
    t = time.time()
    xtr, gtr, xid, gid, xood = make(d, r, pl, sigma_b, delta, seed)
    sc = {"lw_maha": get_scorer("mahalanobis"), "exact_maha": ExactMaha(), "knn": get_scorer("knn", k=50),
          "vim": ViMScorer()}
    rep = crossfit_auroc(xtr, gtr, xid, gid, xood, scorers=sc, protocol="paper_2fold", uncertainty="bootstrap",
                         n_bootstrap=0, random_state=seed)
    return {"d": d, "r": r, "placement": pl, "seed": seed, "n_train": int(len(xtr)),
            "n_fit_fold_mean": float(len(xtr) / 2), "n_id": int(len(xid)), "n_ood": int(len(xood)),
            "time": time.time() - t,
            "res": {n: [x.auroc_leaky, x.auroc_crossfit, x.delta] for n, x in rep.results.items()}}


def run(out, workers):
    cal = json.loads(Path(out, "calibration.json").read_text())
    path = Path(out, "runs.jsonl")
    done = set()
    if path.exists():
        done = {(j["d"], j["r"], j["placement"], j["seed"]) for j in map(json.loads, path.read_text().splitlines())}
    tasks = [(d, r, pl, cal["sigma_b"]["%d_%s" % (d, pl)], cal["delta"][str(d)], s)
             for d in DS[::-1] for r in RATIOS[::-1] for pl in PLACEMENTS for s in range(N_REP)
             if (d, r, pl, s) not in done]
    print("tasks", len(tasks), flush=True)
    with Pool(workers) as pool, open(path, "a") as fh:
        for i, j in enumerate(pool.imap_unordered(_run_task, tasks)):
            fh.write(json.dumps(j) + "\n")
            fh.flush()
            if (i + 1) % 20 == 0:
                print("%d/%d" % (i + 1, len(tasks)), flush=True)


def analyze(out):
    import pandas as pd
    from scipy.stats import t as tdist
    pre = json.loads((RES / "PRECOMMIT.json").read_text())
    cal = json.loads(Path(out, "calibration.json").read_text())
    rows = [dict(d=j["d"], r=j["r"], placement=j["placement"], seed=j["seed"], scorer=n, seen=v[0], unseen=v[1],
                 delta=v[2], n_fit=j["n_fit_fold_mean"]) for j in map(json.loads, Path(out, "runs.jsonl").read_text()
                                                                     .splitlines()) for n, v in j["res"].items()]
    R = pd.DataFrame(rows)
    C = R.groupby(["d", "r", "placement", "scorer"]).agg(n_rep=("delta", "size"), delta_mean=("delta", "mean"),
                                                        delta_se=("delta", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                                                        auroc_seen=("seen", "mean"), auroc_unseen=("unseen", "mean"),
                                                        n_fit_fold=("n_fit", "mean")).reset_index()
    C["sigma_b"] = [cal["sigma_b"]["%d_%s" % (d, p)] for d, p in zip(C.d, C.placement)]
    C["probe_bacc"] = [cal["probe"]["%d_%s" % (d, p)] for d, p in zip(C.d, C.placement)]
    P = R.pivot_table(index=["d", "r", "scorer", "seed"], columns="placement", values="delta").reset_index()
    P["D"] = P["high"] - P["low"]
    K = []
    for (d, r, s), g in P.groupby(["d", "r", "scorer"]):
        n = len(g)
        m, se = g.D.mean(), g.D.std(ddof=1) / np.sqrt(n)
        q = tdist.ppf(0.975, n - 1)
        K.append(dict(d=d, r=r, scorer=s, n_pairs=n, D=m, D_lo=m - q * se, D_hi=m + q * se))
    K = pd.DataFrame(K)

    def verdict(row, Dk):
        s = row.scorer
        if s == "knn":
            return "hold" if row.D_lo > 0 else ("fail" if row.D_hi < 0 else "inconclusive")
        if s == "vim":
            return "hold" if row.D_hi < 0 else ("fail" if row.D_lo > 0 else "inconclusive")
        if s == "exact_maha":
            return "hold" if abs(row.D) <= 0.01 and row.D_lo >= -0.02 and row.D_hi <= 0.02 else "fail"
        de, dk = Dk[(row.d, row.r, "exact_maha")], Dk[(row.d, row.r, "knn")]
        return "hold" if min(de, dk) < row.D < max(de, dk) and de < dk else "fail"
    Dk = {(a, b, c): v for a, b, c, v in K[["d", "r", "scorer", "D"]].itertuples(index=False)}
    K["prediction"] = [pre["predictions"][s] for s in K.scorer]
    K["verdict"] = [verdict(r, Dk) for r in K.itertuples()]
    C.to_csv(RES / "cells.csv", index=False)
    K.to_csv(RES / "contrasts.csv", index=False)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    L = ["# R3 item 4: anisotropic synthetic test", "", "commit: %s" % commit, "",
         "Design and predictions: PRECOMMIT.json (committed before any run). Protocol paper_2fold, K = 2,"
         " n_groups_fit = 15 slides per fold (30 training slides), d in {768, 2560}, n/d = mean per-fold fit size / d."
         " Model ID accuracy: not applicable (synthetic features, no classifier); probe column = slide-ID probe"
         " balanced accuracy at the calibrated sigma_b.", "",
         "## Calibration", "", "| d | placement | sigma_b | probe bacc | delta (OOD shift) |", "|---|---|---|---|---|"]
    for d in DS:
        for p in PLACEMENTS:
            L.append("| %d | %s | %.4f | %.3f | %.4f |" % (d, p, cal["sigma_b"]["%d_%s" % (d, p)],
                                                        cal["probe"]["%d_%s" % (d, p)], cal["delta"][str(d)]))
    L += ["", "## Delta per cell (mean over replicates)", "",
          "| d | n/d | n_fit/fold | placement | scorer | n_rep | Delta | SE | AUROC seen | AUROC unseen |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in C.itertuples():
        L.append("| %d | %d | %.0f | %s | %s | %d | %+.4f | %.4f | %.4f | %.4f |" % (
            r.d, r.r, r.n_fit_fold, r.placement, r.scorer, r.n_rep, r.delta_mean, r.delta_se, r.auroc_seen,
            r.auroc_unseen))
    L += ["", "## Contrast D = Delta_high - Delta_low (paired over seeds, 95% t-CI) and verdict", "",
          "| d | n/d | scorer | prediction | D | 95% CI | verdict |", "|---|---|---|---|---|---|---|"]
    for r in K.sort_values(["scorer", "d", "r"]).itertuples():
        L.append("| %d | %d | %s | %s | %+.4f | [%+.4f, %+.4f] | %s |" % (r.d, r.r, r.scorer, r.prediction, r.D, r.D_lo,
                                                                     r.D_hi, r.verdict))
    tally = {s: "%d/4 hold" % (K[(K.scorer == s)].verdict == "hold").sum() for s in ("knn", "exact_maha", "lw_maha", "vim")}
    L += ["", "Verdict: " + "; ".join("%s %s" % kv for kv in tally.items()) + " (per PRECOMMIT decision rule).", "",
          "Caveats: precommitted within R3 (PRECOMMIT.json, before any run) but not part of the original paper precommit; synthetic Gaussian features."]
    (RES / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("calibrate", "run", "analyze"))
    ap.add_argument("--out", default=str(Path.home() / "r3work/item4"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"calibrate": lambda: calibrate(a.out, a.workers), "run": lambda: run(a.out, a.workers),
     "analyze": lambda: analyze(a.out)}[a.phase]()


if __name__ == "__main__":
    main()
