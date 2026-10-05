#!/usr/bin/env python3
"""Multibench scoring on CPU (decisions/precommit_group_leakage_multibench_2026-10-06.md).

--mode A  : standard model features ({arch}_s42_full.npz). (A) group-disjoint 2-fold for the 4 fit
            scores (same vs disjoint, Δ = disjoint − same) and (C) the 7-score standard vector
            (full-train fit, all ID) vs the group-disjoint vector (fit scores disjoint 2-fold, logit
            scores of the same model on the same fold-(1−f) ID subsets). Writes A_{arch}.json.
--mode B  : retrained model ({arch}_s{s}_f{f}.npz): all 7 scores, fit = fold train, ID = unseen /
            seen groups. Writes scores_{arch}_s{s}_f{f}.json.
Published scorer config; features cast to float64 (Ledoit-Wolf float64 Mahalanobis); calc_auroc.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import multibench_common as M  # noqa: E402

FIT = ("Mahalanobis", "kNN", "ViM", "ReAct")


def scorer(z, sel):
    from src.utils.scoring import OODScorer
    g = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
    sc = OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None)
    sc.fit(g("train_feats")[sel], train_logits=g("train_logits")[sel], fc_weight=g("fc_weight"), fc_bias=g("fc_bias"))
    return sc


def all_scores(sc, z, key, sel=None):
    lo, fe = np.asarray(z[key + "_logits"], dtype=np.float64), np.asarray(z[key + "_feats"], dtype=np.float64)
    if sel is not None:
        lo, fe = lo[sel], fe[sel]
    s = sc.get_all_scores(lo, fe)
    return {C.DISPLAY[k]: v for k, v in s.items() if k in C.DISPLAY}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(M.SPECS))
    ap.add_argument("--arch", required=True)
    ap.add_argument("--mode", required=True, choices=["A", "B"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    from src.utils.benchmark_metrics import calc_auroc

    t0 = time.time()
    rd = M.report_dir(args.ds)
    if args.mode == "B":
        z = np.load(M.feat_dir(args.ds) / ("%s_s%d_f%d.npz" % (args.arch, args.seed, args.fold)))
        f = args.fold
        sc = scorer(z, slice(None))
        si, so = all_scores(sc, z, "id"), all_scores(sc, z, "ood")
        fi = z["id_fold"]
        res = {"ds": args.ds, "arch": args.arch, "seed": args.seed, "fold": f, "n_fit": int(len(z["train_feats"]))}
        for m in C.METHODS_ORDER:
            res["auroc_%s_unseen" % m] = calc_auroc(si[m][fi == 1 - f], so[m])
            res["auroc_%s_seen" % m] = calc_auroc(si[m][fi == f], so[m])
        res["seconds"] = round(time.time() - t0, 1)
        (rd / ("scores_%s_s%d_f%d.json" % (args.arch, args.seed, args.fold))).write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res), flush=True)
        return 0

    z = np.load(M.feat_dir(args.ds) / ("%s_s42_full.npz" % args.arch))
    tf, fi = z["train_fold"], z["id_fold"]
    if (fi < 0).any():
        print("ID images in groups without train images (excluded):", int((fi < 0).sum()), flush=True)
    res = {"ds": args.ds, "arch": args.arch, "seed": 42, "n_train_fold": [int((tf == k).sum()) for k in (0, 1)],
           "n_id_fold": [int((fi == k).sum()) for k in (0, 1)], "n_id_excluded": int((fi < 0).sum()),
           "n_ood": int(len(z["ood_feats"]))}
    sc = scorer(z, slice(None))
    si, so = all_scores(sc, z, "id"), all_scores(sc, z, "ood")
    res["standard"] = {m: calc_auroc(si[m], so[m]) for m in C.METHODS_ORDER}
    same, dis = {m: [] for m in C.METHODS_ORDER}, {m: [] for m in C.METHODS_ORDER}
    for f in (0, 1):
        sc = scorer(z, tf == f)
        si, so = all_scores(sc, z, "id"), all_scores(sc, z, "ood")
        for m in C.METHODS_ORDER:
            same[m].append(calc_auroc(si[m][fi == f], so[m]))
            dis[m].append(calc_auroc(si[m][fi == 1 - f], so[m]))
        del sc
    res["same_2fold"] = {m: float(np.mean(same[m])) for m in C.METHODS_ORDER}
    res["disjoint_2fold"] = {m: float(np.mean(dis[m])) for m in C.METHODS_ORDER}
    res["delta"] = {m: res["disjoint_2fold"][m] - res["same_2fold"][m] for m in FIT}
    res["seconds"] = round(time.time() - t0, 1)
    (rd / ("A_%s.json" % args.arch)).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
