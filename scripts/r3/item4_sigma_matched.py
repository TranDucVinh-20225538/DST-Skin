#!/usr/bin/env python3
"""R3 item 4, post-hoc check of the sigma_b explanation (not part of PRECOMMIT; precommit verdict unchanged).

The precommitted run calibrated sigma_b per placement to a slide-ID probe accuracy of 0.90, which needed a larger
sigma_b for the low placement. Offsets are sigma_b * z * sqrt(lambda) along the placement eigendirections, so in
population-whitened coordinates their size is sigma_b for both placements. Here both placements use the SAME
sigma_b: once at the calibrated high-placement value and once at the calibrated low-placement value (per d).
Same generator, seeds (common random numbers across placements), delta, sizes and scorers (exact_maha, lw_maha)
as item 4; 50 replicates per (d, n/d, placement, sigma_b). D = Delta_high - Delta_low, paired over seeds, 95% t-CI.

  run     -> <out>/sigma_matched.jsonl
  analyze -> results/r3/4/posthoc_sigma_matched.{csv,md}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

import item4_anisotropic as A  # sets BLAS to 1 thread per worker
import numpy as np
from crossfit_ood import crossfit_auroc, get_scorer


def _task(a):
    d, r, pl, which, sigma_b, delta, seed = a
    xtr, gtr, xid, gid, xood = A.make(d, r, pl, sigma_b, delta, seed)
    sc = {"lw_maha": get_scorer("mahalanobis"), "exact_maha": A.ExactMaha()}
    rep = crossfit_auroc(xtr, gtr, xid, gid, xood, scorers=sc, protocol="paper_2fold", uncertainty="bootstrap",
                         n_bootstrap=0, random_state=seed)
    return {"d": d, "r": r, "placement": pl, "sigma_from": which, "sigma_b": sigma_b, "seed": seed,
            "res": {n: x.delta for n, x in rep.results.items()}}


def run(out, workers):
    cal = json.loads(Path(out, "calibration.json").read_text())
    path = Path(out, "sigma_matched.jsonl")
    done = set()
    if path.exists():
        done = {(j["d"], j["r"], j["placement"], j["sigma_from"], j["seed"])
                for j in map(json.loads, path.read_text().splitlines())}
    tasks = [(d, r, pl, w, cal["sigma_b"]["%d_%s" % (d, w)], cal["delta"][str(d)], s)
             for d in A.DS[::-1] for r in A.RATIOS[::-1] for w in A.PLACEMENTS for pl in A.PLACEMENTS
             for s in range(A.N_REP) if (d, r, pl, w, s) not in done]
    print("tasks", len(tasks), flush=True)
    with Pool(workers) as pool, open(path, "a") as fh:
        for j in pool.imap_unordered(_task, tasks):
            fh.write(json.dumps(j) + "\n")
            fh.flush()
    with Pool(workers) as pool:  # probe accuracy at the matched sigma_b (context only)
        probe = {"%d_%s_at_%s" % (d, pl, w): float(np.mean(pool.map(A._probe_task, [
            (d, pl, cal["sigma_b"]["%d_%s" % (d, w)], 20_000 + i) for i in range(5)])))
            for d in A.DS for w in A.PLACEMENTS for pl in A.PLACEMENTS}
    Path(out, "sigma_matched_probe.json").write_text(json.dumps(probe, indent=1))


def analyze(out):
    import pandas as pd
    from scipy.stats import t as tdist
    probe = json.loads(Path(out, "sigma_matched_probe.json").read_text())
    rows = [dict(d=j["d"], r=j["r"], placement=j["placement"], sigma_from=j["sigma_from"], sigma_b=j["sigma_b"],
                 seed=j["seed"], scorer=n, delta=v)
            for j in map(json.loads, Path(out, "sigma_matched.jsonl").read_text().splitlines())
            for n, v in j["res"].items()]
    R = pd.DataFrame(rows)
    P = R.pivot_table(index=["d", "r", "sigma_from", "sigma_b", "scorer", "seed"], columns="placement",
                      values="delta").reset_index()
    P["D"] = P["high"] - P["low"]
    K = []
    for (d, r, w, sb, s), g in P.groupby(["d", "r", "sigma_from", "sigma_b", "scorer"]):
        n = len(g)
        m, se = g.D.mean(), g.D.std(ddof=1) / np.sqrt(n)
        q = tdist.ppf(0.975, n - 1)
        K.append(dict(d=d, n_over_d=r, scorer=s, sigma_from=w, sigma_b=sb, n_pairs=n, delta_high=g.high.mean(),
                      delta_low=g.low.mean(), D=m, D_lo=m - q * se, D_hi=m + q * se,
                      probe_high=probe["%d_high_at_%s" % (d, w)], probe_low=probe["%d_low_at_%s" % (d, w)]))
    K = pd.DataFrame(K).sort_values(["scorer", "sigma_from", "d", "n_over_d"])
    K.to_csv(A.RES / "posthoc_sigma_matched.csv", index=False)
    commit = subprocess.run(["git", "-C", str(A.REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    L = ["# R3 item 4: post-hoc check of the sigma_b explanation (exact / LW Mahalanobis)", "",
         "commit: %s" % commit, "",
         "Post-hoc check, not in PRECOMMIT.json; the precommitted verdict in REPORT.md is unchanged. Both placements"
         " use the same sigma_b (the calibrated high- or low-placement value for that d); everything else as in"
         " item 4 (paper_2fold, K = 2, n_groups_fit = 15 per fold, 50 paired replicates). ID accuracy: not"
         " applicable (synthetic); probe columns = slide-ID probe balanced accuracy at the matched sigma_b.", "",
         "| scorer | sigma_b from | sigma_b | d | n/d | Delta_high | Delta_low | D = high - low | 95% CI |"
         " probe high / low |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in K.itertuples():
        L.append("| %s | %s | %.4f | %d | %d | %+.4f | %+.4f | %+.4f | [%+.4f, %+.4f] | %.3f / %.3f |" % (
            r.scorer, r.sigma_from, r.sigma_b, r.d, r.n_over_d, r.delta_high, r.delta_low, r.D, r.D_lo, r.D_hi,
            r.probe_high, r.probe_low))
    (A.RES / "posthoc_sigma_matched.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("run", "analyze"))
    ap.add_argument("--out", default=str(Path.home() / "r3work/item4"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    {"run": lambda: run(a.out, a.workers), "analyze": lambda: analyze(a.out)}[a.phase]()


if __name__ == "__main__":
    main()
