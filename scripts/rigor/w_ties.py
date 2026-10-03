#!/usr/bin/env python
"""H17: Kendall W when near-equal AUROCs are treated as ties (CSV only, seconds).

Reviewer point: Energy / ELogitNorm / ReAct (and Maha / kNN) often differ by < 0.01 AUROC.
A rank flip between two such scores counts as full disagreement in W, although no test could
tell them apart. Here AUROCs closer than eps (single-linkage on the sorted values: a gap
< eps joins the group) share a mid-rank, then the tie-corrected W of the repo is computed.
eps = 0 reproduces the normal W (REPRO line).

Output: outputs/reports/rigor_pack/w_ties/{w_ties.csv, w_ties_log.txt}
Rows: domain, eps, kind (cross_arch at each seed, k=8; cross_seed per arch, k=5), label, W.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rigor import common as C  # noqa: E402

EPS = (0.0, 0.005, 0.01, 0.02)


def tied_ranks(v: np.ndarray, eps: float) -> np.ndarray:
    """1 = highest. Values within an eps-chain share the mean rank of the group."""
    order = np.argsort(-v, kind="mergesort")
    sv = v[order]
    ranks = np.empty(len(v), dtype=np.float64)
    start = 0
    for i in range(1, len(sv) + 1):
        if i == len(sv) or (sv[i - 1] - sv[i]) > eps or (eps == 0 and sv[i - 1] != sv[i]):
            ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    return ranks


def w_of(mat: np.ndarray, eps: float) -> float:
    if mat.shape[0] < 2 or np.isnan(mat).any():
        return float("nan")
    return float(C.kendall_w(np.stack([tied_ranks(r, eps) for r in mat])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--reports", default=None)
    ap.add_argument("--domains", nargs="+", default=list(C.DOMAINS))
    args = ap.parse_args()
    root = Path(args.root)
    out = C.ensure_dir(Path(args.reports) if args.reports else C.default_reports(root) / "w_ties")
    archs, seeds = list(C.ARCHS), list(C.SEEDS)
    rows, log, ok_all = [], ["w_ties  W source: %s" % C.W_SOURCE], True
    for domain in args.domains:
        cube = C.load_cube(root, domain, archs, seeds)
        for eps in EPS:
            for si, s in enumerate(seeds):
                rows.append({"domain": domain, "eps": eps, "kind": "cross_arch", "label": "seed%d" % s, "k": len(archs),
                             "W": w_of(cube[:, si, :], eps)})
            for ai, a in enumerate(archs):
                rows.append({"domain": domain, "eps": eps, "kind": "cross_seed", "label": a, "k": len(seeds),
                             "W": w_of(cube[ai], eps)})
        if domain == "camelyon17":
            v = [r["W"] for r in rows if r["domain"] == domain and r["eps"] == 0 and r["label"] == "seed42"][0]
            ok = bool(np.isfinite(v) and abs(v - 0.694) <= 0.0015)
            ok_all &= ok
            log.append("  REPRO %s camelyon17 W seed42 eps=0  published 0.694 recomputed %.4f" % ("PASS" if ok else "FAIL", v))
    pd.DataFrame(rows).to_csv(out / "w_ties.csv", index=False)
    log.append("Wrote w_ties.csv. Read only against H17 in the precommit.")
    C.write_text(out / "w_ties_log.txt", log)
    print("\n".join(log))
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
