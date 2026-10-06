#!/usr/bin/env python3
"""Medbench scoring for one run npz (decisions/precommit_group_leakage_medbench_2026-10-06.md, L3-L5).

Published scorer fitted on the arm's training images (float64), 7 scores on every ID / OOD set.
- seen / unseen ID sets: std arms test_seen / test_unseen (+ Kermany unseen_extra), BreakHis std_r seen /
  unseen, (B) arms seen / unseen.
- class-matched AUROC: largest common per-class count, 20 random subsamples (default_rng(0)), mean.
  gap = AUROC(seen) - AUROC(unseen) for all 7 scores and every OOD set.
- A-fit (std arms): group-disjoint 2-fold fit of the 4 fit scores on the training groups; ID = seen-set
  images; Δ_fit = same-group-fit AUROC - disjoint-fit AUROC, fold mean.
- paired cluster bootstrap, B = 2000 (groups of the ID sets, images of the OOD set, same resample for
  every score); class matching via per-class weights; 95% percentile CIs of the gaps / Δ_fit and of every
  pairwise AUROC difference under the seen and unseen protocols.
Writes outputs/reports/rigor_pack/medbench/<ds>/scores_<arch>_s<seed>_<arm>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import medbench_common as MC  # noqa: E402

FIT = ("Mahalanobis", "kNN", "ViM", "ReAct")
ORDER = list(C.METHODS_ORDER)
NB, NSUB = 2000, 20


def scorer(feats, logits, w, b):
    from src.utils.scoring import OODScorer
    sc = OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None)
    sc.fit(feats, train_logits=logits, fc_weight=w, fc_bias=b)
    return sc


def all_scores(sc, logits, feats):
    s = sc.get_all_scores(logits, feats)
    return {C.DISPLAY[k]: np.asarray(v, dtype=np.float64) for k, v in s.items() if k in C.DISPLAY}


def auroc(i, o):
    from src.utils.benchmark_metrics import calc_auroc
    return float(calc_auroc(i, o))


def wauroc(si, wi, so_sorted, wo_cum, wo_tot):
    """Weighted AUROC P(s_id > s_ood) + 0.5 P(tie); OOD scores sorted with cumulative weights."""
    lo = np.searchsorted(so_sorted, si, side="left")
    hi = np.searchsorted(so_sorted, si, side="right")
    below = np.where(lo > 0, wo_cum[np.maximum(lo - 1, 0)], 0.0)
    tie = np.where(hi > 0, wo_cum[np.maximum(hi - 1, 0)], 0.0) - below
    return float(np.sum(wi * (below + 0.5 * tie)) / (wi.sum() * wo_tot))


def matched_counts(ys, yu):
    cls = np.intersect1d(np.unique(ys), np.unique(yu))
    return {c: int(min((ys == c).sum(), (yu == c).sum())) for c in cls}


def matched_auroc(S_seen, S_unseen, ys, yu, S_ood, rng):
    n = matched_counts(ys, yu)
    acc = {m: {"seen": [], "unseen": []} for m in ORDER}
    for _ in range(NSUB):
        pick_s = np.concatenate([rng.choice(np.flatnonzero(ys == c), k, replace=False) for c, k in n.items()])
        pick_u = np.concatenate([rng.choice(np.flatnonzero(yu == c), k, replace=False) for c, k in n.items()])
        for m in ORDER:
            acc[m]["seen"].append(auroc(S_seen[m][pick_s], S_ood[m]))
            acc[m]["unseen"].append(auroc(S_unseen[m][pick_u], S_ood[m]))
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}, n


def class_weights(y, n):
    w = np.zeros(len(y))
    for c, k in n.items():
        m = y == c
        w[m] = k / m.sum()
    return w


def cluster_resample(groups, rng):
    """Multiplicity weights from resampling groups with replacement."""
    u, inv = np.unique(groups, return_inverse=True)
    cnt = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
    return cnt[inv].astype(np.float64)


def bootstrap(S_seen, S_unseen, S_ood, ys, yu, gs, gu, n, rng, fit_extra=None):
    ws0, wu0 = class_weights(ys, n), class_weights(yu, n)
    order = {m: np.argsort(S_ood[m], kind="mergesort") for m in ORDER}
    sor = {m: S_ood[m][order[m]] for m in ORDER}
    seen = np.zeros((NB, len(ORDER)))
    unseen = np.zeros((NB, len(ORDER)))
    for bi in range(NB):
        ws = ws0 * cluster_resample(gs, rng)
        wu = wu0 * cluster_resample(gu, rng)
        wo = np.bincount(rng.integers(0, len(S_ood["MSP"]), len(S_ood["MSP"])), minlength=len(S_ood["MSP"])).astype(float)
        for j, m in enumerate(ORDER):
            cum = np.cumsum(wo[order[m]])
            seen[bi, j] = wauroc(S_seen[m], ws, sor[m], cum, cum[-1])
            unseen[bi, j] = wauroc(S_unseen[m], wu, sor[m], cum, cum[-1])
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    out = {"gap_ci": {m: ci(seen[:, j] - unseen[:, j]) for j, m in enumerate(ORDER)}, "pair_ci_seen": {}, "pair_ci_unseen": {}}
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if i < j:
                out["pair_ci_seen"][f"{a}-{b}"] = ci(seen[:, i] - seen[:, j])
                out["pair_ci_unseen"][f"{a}-{b}"] = ci(unseen[:, i] - unseen[:, j])
    return out


def fold_map(ds: str, train_groups: np.ndarray, train_labels: np.ndarray) -> dict:
    """Group -> fold: the Phase-0 (B) fold where the group has one, else assigned like Phase 0
    (classes shuffled with default_rng(0), greedy by images) among the remaining groups."""
    from medbench_phase0 import assign_folds
    d = pd.read_csv(MC.SPL / f"{ds}.csv.gz", low_memory=False)
    fm = {}
    if "fold" in d.columns:
        x = d[(d.fold >= 0) & d.group.notna()]
        fm = dict(zip(x.group.astype(str), x.fold.astype(int)))
    rest = ~np.isin(train_groups, list(fm))
    if rest.any():
        fm.update(assign_folds(train_groups[rest], train_labels[rest], np.random.default_rng(0)))
    return fm


def groups_of(ds: str, keys: np.ndarray) -> np.ndarray:
    d = pd.read_csv(MC.SPL / f"{ds}.csv.gz", low_memory=False)
    if ds == "dermamnist":
        m = dict(zip(d.key, d.group))
    elif ds == "isic2019":
        m = dict(zip("isic:" + d.image, d.group))
    elif ds == "kermany":
        m = dict(zip(d.version + ":" + d.key, d.group))
    else:
        m = dict(zip(d.key, d.group))
    return np.array([str(m.get(k, "KEY:" + k)) for k in keys])


def main() -> int:
    global NB
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--nb", type=int, default=2000)
    args = ap.parse_args()
    NB = args.nb
    t0 = time.time()
    ds, tag = args.ds, f"{args.arch}_s{args.seed}_{args.arm}"
    z = np.load(MC.REPO / f"outputs/rigor_pack/medbench/{ds}/{tag}.npz")
    f64 = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
    w, b = f64("fc_weight"), f64("fc_bias")
    sets = sorted({k[:-6] for k in z.files if k.endswith("_feats")})
    sc = scorer(f64("train_feats"), f64("train_logits"), w, b)
    S = {k: all_scores(sc, f64(f"{k}_logits"), f64(f"{k}_feats")) for k in sets if k != "train"}
    res = {"ds": ds, "arch": args.arch, "seed": args.seed, "arm": args.arm,
           "n": {k: int(len(z[f"{k}_labels"])) for k in sets}}
    for k in sets:
        if not k.startswith("ood"):
            y, p = z[f"{k}_labels"], f64(f"{k}_logits").argmax(1)
            res[f"acc_{k}"] = float((p == y).mean())
            res[f"bacc_{k}"] = float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))
    oods = [k for k in sets if k.startswith("ood")]
    res["auroc_all"] = {k: {m: auroc(S[k][m], S[o][m]) for m in ORDER} for k in sets if not k.startswith("ood") and k != "train"
                        for o in oods[:1]}
    for o in oods[1:]:
        res[f"auroc_all_{o}"] = {k: {m: auroc(S[k][m], S[o][m]) for m in ORDER} for k in sets
                                  if not k.startswith("ood") and k != "train"}

    # seen / unseen definition
    if "test_seen" in sets:
        seen = ["test_seen"]
        unseen = ["test_unseen"] + (["unseen_extra"] if "unseen_extra" in sets else [])
    elif "seen" in sets:
        seen, unseen = ["seen"], ["unseen"]
    else:
        seen = unseen = []
    if seen:
        cat = lambda names, f: np.concatenate([f(n) for n in names])  # noqa: E731
        Ss = {m: cat(seen, lambda n: S[n][m]) for m in ORDER}
        Su = {m: cat(unseen, lambda n: S[n][m]) for m in ORDER}
        ys, yu = cat(seen, lambda n: z[f"{n}_labels"]), cat(unseen, lambda n: z[f"{n}_labels"])
        ks, ku = cat(seen, lambda n: z[f"{n}_keys"]), cat(unseen, lambda n: z[f"{n}_keys"])
        gs, gu = groups_of(ds, ks), groups_of(ds, ku)
        res["n_seen"], res["n_unseen"] = int(len(ys)), int(len(yu))
        res["groups_seen"], res["groups_unseen"] = int(len(np.unique(gs))), int(len(np.unique(gu)))
        rng = np.random.default_rng(0)
        for o in oods:
            mt, n = matched_auroc(Ss, Su, ys, yu, S[o], rng)
            res[f"matched_{o}"] = mt
            res[f"gap_{o}"] = {m: mt[m]["seen"] - mt[m]["unseen"] for m in ORDER}
        res["matched_counts"] = {int(c): k for c, k in n.items()}
        res["bootstrap_ood"] = bootstrap(Ss, Su, S[oods[0]], ys, yu, gs, gu, n, np.random.default_rng(1))

    # A-fit on std arms: group-disjoint 2-fold fit of the fit scores, ID = seen set
    if seen and (args.arm == "std" or args.arm.startswith("std_r")):
        kt = z["train_keys"]
        gt = groups_of(ds, kt)
        fm = fold_map(ds, gt, z["train_labels"])
        tf = np.array([fm[g] for g in gt])
        fs = np.array([fm.get(g, -1) for g in gs])
        same, dis = {m: [] for m in FIT}, {m: [] for m in FIT}
        per_fold = []
        for f in (0, 1):
            sel = tf == f
            scf = scorer(f64("train_feats")[sel], f64("train_logits")[sel], w, b)
            si = {m: np.concatenate([all_scores(scf, f64(f"{n}_logits"), f64(f"{n}_feats"))[m] for n in seen]) for m in FIT}
            so = all_scores(scf, f64(f"{oods[0]}_logits"), f64(f"{oods[0]}_feats"))
            per_fold.append((si, so))
            for m in FIT:
                same[m].append(auroc(si[m][fs == f], so[m]))
                dis[m].append(auroc(si[m][fs == 1 - f], so[m]))
        brng = np.random.default_rng(2)
        nood = len(per_fold[0][1]["kNN"])
        deltas = np.zeros((NB, len(FIT)))
        for bi in range(NB):
            wg = cluster_resample(gs, brng)
            wo = np.bincount(brng.integers(0, nood, nood), minlength=nood).astype(float)
            for j, m in enumerate(FIT):
                d = 0.0
                for f, (si, so) in enumerate(per_fold):
                    o = np.argsort(so[m], kind="mergesort")
                    cum = np.cumsum(wo[o])
                    a_s = wauroc(si[m][fs == f], wg[fs == f], so[m][o], cum, cum[-1])
                    a_d = wauroc(si[m][fs == 1 - f], wg[fs == 1 - f], so[m][o], cum, cum[-1])
                    d += (a_s - a_d) / 2
                deltas[bi, j] = d
        res["afit"] = {"n_train_fold": [int((tf == k).sum()) for k in (0, 1)],
                       "n_seen_fold": [int((fs == k).sum()) for k in (0, 1)],
                       "same": {m: float(np.mean(same[m])) for m in FIT},
                       "disjoint": {m: float(np.mean(dis[m])) for m in FIT},
                       "delta_fit": {m: float(np.mean(same[m]) - np.mean(dis[m])) for m in FIT},
                       "delta_fit_ci": {m: [float(np.percentile(deltas[:, j], 2.5)), float(np.percentile(deltas[:, j], 97.5))]
                                        for j, m in enumerate(FIT)}}
    res["seconds"] = round(time.time() - t0, 1)
    out = MC.REPO / f"outputs/reports/rigor_pack/medbench/{ds}/scores_{tag}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if not k.startswith(("bootstrap", "matched"))})[:3000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
