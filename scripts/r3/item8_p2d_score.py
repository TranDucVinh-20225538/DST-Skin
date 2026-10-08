#!/usr/bin/env python3
"""R3 item 8 / P2-d scoring for one (dataset, variant, run, scorer) cell (results/r3/8/p2d/PRECOMMIT.json).

run = resnet50_s42 | resnet50_s43 | convnext_tiny_s42 | convnext_tiny_s43 | dinov2_vitb14 | biomedclip;
scorer = mahalanobis_l2 | knn_mean_cosine (crossfit_ood, Track A definitions, CPU, float64). Primary OOD candidate only.
  seen    = test images (role seen) whose group (video / patient) has >= 1 fit image; the others are dropped and counted
  leaky   = AUROC(seen vs OOD), scorer fitted on all fit images
  F2      = A-fit cross-fit: fold of a seen image = afit fold of its group (Phase 0, medbench fold rule, default_rng(0));
            fit k drops the fit images of the groups of fold k's seen images; matched pooling (n_k weights)
  Delta_fit = leaky - F2
  K_seg only: leaky_gap = leaky with the fit images flagged gap25 (Phase 0: within +/- 25 frames of a K_seg seen frame of
            the same video) removed from the scorer fit; Delta_fit_gap = leaky_gap - F2; gap_effect = Delta_fit - Delta_fit_gap
  truth   = AUROC(unseen vs OOD), full fit; A_gap = leaky - truth (descriptive);
  A_gap_cm = seen post-stratified on ID class to the unseen class distribution (strata with < 5 images in either set
            dropped from both): weighted AUROC(seen vs OOD) - AUROC(unseen kept vs OOD)
Uncertainty, PRIMARY: refit-aware grouped jackknife (crossfit_ood _blocks / _jk_var_weighted, Busing): ID units = groups of
fit and unseen, <= 50 blocks deleted together with every refit; OOD: <= 50 blocks over OOD groups (videos / patients),
scores fixed; V = V_id + V_ood; 95% normal CI. REFERENCE ONLY: cluster bootstrap (B = 2000; seen / unseen / OOD resampled by
group), fitted scores and weights fixed.
leaky and F2 are checked against crossfit_auroc (1e-9 plus one pair per near-tied seen / OOD score pair, as P2-b).
FM runs: the linear-probe competence result is computed here (scorer mahalanobis_l2 only) and written to train/.
Writes results/r3/8/p2d/cells/{ds}_{variant}_{run}_{scorer}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from statistics import NormalDist

import numpy as np
from sklearn.metrics import roc_auc_score

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/r3")]
from item8_p2d_common import FEAT, FMS, P2D, macro_ovr_auroc, variant_sets  # noqa: E402

NB, B, MIN_STRATUM = 50, 2000, 5


def auroc(s_id, s_ood, w_id=None):
    y = np.r_[np.ones(len(s_id)), np.zeros(len(s_ood))]
    w = None if w_id is None else np.r_[w_id, np.ones(len(s_ood))]
    return float(roc_auc_score(y, np.r_[s_id, s_ood], sample_weight=w))


def near_ties(a, b, eps=1e-12):
    b = np.sort(b)
    return int((np.searchsorted(b, a + eps, side="right") - np.searchsorted(b, a - eps, side="left")).sum())


def cm_weights(ys, yu):
    import pandas as pd
    cs, cu = pd.Series(ys).value_counts(), pd.Series(yu).value_counts()
    keep = sorted(s for s in set(cs.index) & set(cu.index) if cs[s] >= MIN_STRATUM and cu[s] >= MIN_STRATUM)
    ms, mu = np.isin(ys, keep), np.isin(yu, keep)
    if not keep:
        return ms, mu, np.zeros(0), keep
    ps, pu = cs[keep] / ms.sum(), cu[keep] / mu.sum()
    return ms, mu, (pu / ps).reindex(ys[ms]).to_numpy(), keep


def probe(Xf, yf, Xs, ys, Xu, yu, classes):
    from sklearn.linear_model import LogisticRegression
    out = {}
    D = {"seen": (Xs, ys), "unseen": (Xu, yu)}
    P = {k: np.zeros((len(v[0]), len(classes))) for k, v in D.items()}
    for j, c in enumerate(classes):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(Xf, (yf == c).astype(int))
        for k, (X, _) in D.items():
            P[k][:, j] = clf.decision_function(X)
    for k, (_, y) in D.items():
        m, per = macro_ovr_auroc(y, P[k], classes)
        out[k] = {"macro": m, "per_class": per, "acc": float((np.array(classes)[P[k].argmax(1)] == y).mean())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["kvasir", "brain"])
    ap.add_argument("--variant", required=True, choices=["lit", "seg", "rec"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--scorer", required=True, choices=["mahalanobis_l2", "knn_mean_cosine"])
    ap.add_argument("--nb", type=int, default=NB, help="jackknife blocks (smoke: fewer)")
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--smoke", action="store_true", help="read *_smoke feature files")
    a = ap.parse_args()
    from crossfit_ood import crossfit_auroc, get_scorer
    from crossfit_ood.core import _blocks, _jk_var_weighted

    t0 = time.time()
    V = variant_sets(a.ds, a.variant)
    if V is None:
        print("no primary candidate for", a.ds)
        return 0
    S, classes, fold = V["sets"], V["classes"], V["fold"]
    sm = "_smoke" if a.smoke else ""
    if a.run in FMS:
        z = np.load(FEAT / f"{a.ds}_{a.run}{sm}.npz")
        F = dict(zip(z["keys"], np.asarray(z["feats"], dtype=np.float64)))
        get = lambda r, keys: np.stack([F[k] for k in keys])  # noqa: E731
        if a.smoke:
            S = {r: d[d.key.isin(F.keys())].reset_index(drop=True) for r, d in S.items()}
    else:
        z = np.load(FEAT / f"{a.ds}_{a.variant}_{a.run}{sm}.npz", allow_pickle=True)
        F = {r: dict(zip(np.asarray(z[f"{r}_keys"]).astype(str), np.asarray(z[f"{r}_feats"], dtype=np.float64)))
             for r in ("fit", "seen", "unseen", "ood")}
        get = lambda r, keys: np.stack([F[r][k] for k in keys])  # noqa: E731
        if a.smoke:
            S = {r: d[d.key.isin(F[r].keys())].reset_index(drop=True) if r in F else d for r, d in S.items()}
    fit, ood, un = S["fit"], S["ood"], S["unseen"]
    fit_g = set(fit.grp)
    seen_all = S["seen"]
    ok = seen_all.grp.isin(fit_g).to_numpy()
    seen = seen_all[ok].reset_index(drop=True)
    gap_mask = fit.gap25.astype(bool).to_numpy() if a.variant == "seg" else None
    if gap_mask is not None:
        assert seen.grp.isin(set(fit.grp[~gap_mask])).all(), "seen group without fit frames after the gap"
    X = {"fit": get("fit", fit.key), "seen": get("seen", seen.key), "unseen": get("unseen", un.key), "ood": get("ood", ood.key)}
    g = {"fit": fit.grp.to_numpy(), "seen": seen.grp.to_numpy(), "unseen": un.grp.to_numpy(), "ood": ood.grp.to_numpy()}
    y = {"fit": fit.label.to_numpy(), "seen": seen.label.to_numpy(), "unseen": un.label.to_numpy()}
    fold_seen = np.array([fold[x] for x in g["seen"]])
    assert not (set(g["unseen"]) & fit_g) and not (set(g["ood"]) & (fit_g | set(g["unseen"])))
    stats = ["leaky", "F2", "delta_fit", "truth", "a_gap", "a_gap_cm"] + (["leaky_gap", "delta_fit_gap", "gap_effect"]
                                                                         if gap_mask is not None else [])

    def fits(kf, ks, ku):
        full = get_scorer(a.scorer).fit(X["fit"][kf])
        sc = {"seen": np.asarray(full.score(X["seen"][ks]), dtype=np.float64),
              "unseen": np.asarray(full.score(X["unseen"][ku]), dtype=np.float64),
              "ood": np.asarray(full.score(X["ood"]), dtype=np.float64), "fold": []}
        if gap_mask is not None:
            gf = get_scorer(a.scorer).fit(X["fit"][kf & ~gap_mask])
            sc["seen_gap"] = np.asarray(gf.score(X["seen"][ks]), dtype=np.float64)
            sc["ood_gap"] = np.asarray(gf.score(X["ood"]), dtype=np.float64)
        ps, fs, pf = g["seen"][ks], fold_seen[ks], g["fit"][kf]
        for k in (0, 1):
            f = get_scorer(a.scorer).fit(X["fit"][kf][~np.isin(pf, np.unique(ps[fs == k]))])
            sc["fold"].append((np.asarray(f.score(X["seen"][ks][fs == k]), dtype=np.float64),
                               np.asarray(f.score(X["ood"]), dtype=np.float64)))
        sc["cm"] = cm_weights(y["seen"][ks], y["unseen"][ku])
        return sc

    def theta(sc, okeep=None):
        ix = np.arange(len(X["ood"])) if okeep is None else np.flatnonzero(okeep)
        so = sc["ood"][ix]
        leaky, truth = auroc(sc["seen"], so), auroc(sc["unseen"], so)
        n = [len(sc["fold"][k][0]) for k in (0, 1)]
        f2 = sum(n[k] * auroc(sc["fold"][k][0], sc["fold"][k][1][ix]) for k in (0, 1)) / sum(n)
        ms, mu, w, keep = sc["cm"]
        cm = auroc(sc["seen"][ms], so, w) - auroc(sc["unseen"][mu], so) if keep else np.nan
        out = [leaky, f2, leaky - f2, truth, leaky - truth, cm]
        if gap_mask is not None:
            lg = auroc(sc["seen_gap"], sc["ood_gap"][ix])
            out += [lg, lg - f2, (leaky - f2) - (lg - f2)]
        return np.array(out)

    all_f, all_s, all_u = (np.ones(len(X[k]), bool) for k in ("fit", "seen", "unseen"))
    sc0 = fits(all_f, all_s, all_u)
    th = theta(sc0)
    rep = crossfit_auroc(X["fit"], g["fit"], X["seen"], g["seen"], X["ood"], scorers=[a.scorer], protocol="default",
                         n_splits=2, fold_ids=fold_seen, uncertainty="bootstrap", n_bootstrap=0,
                         random_state=0).results[a.scorer]
    t_lk = near_ties(sc0["seen"], sc0["ood"])
    t_f2 = [near_ties(*sc0["fold"][k]) for k in (0, 1)]
    nk = [len(sc0["fold"][k][0]) for k in (0, 1)]
    no = len(X["ood"])
    chk = {"leaky_diff": abs(rep.auroc_leaky - th[0]), "F2_diff": abs(rep.auroc_crossfit - th[1]),
           "leaky_package": float(rep.auroc_leaky), "F2_package": float(rep.auroc_crossfit),
           "near_tied_pairs_leaky": t_lk, "near_tied_pairs_F2": t_f2,
           "tol_leaky": 1e-9 + t_lk / (len(X["seen"]) * no),
           "tol_F2": 1e-9 + sum(nk[k] / sum(nk) * t_f2[k] / (nk[k] * no) for k in (0, 1))}
    assert chk["leaky_diff"] <= chk["tol_leaky"] and chk["F2_diff"] <= chk["tol_F2"], chk
    print(a.ds, a.variant, a.run, a.scorer, dict(zip(stats, np.round(th, 4))), round(time.time() - t0), flush=True)

    rng = np.random.default_rng(1)
    id_blocks = _blocks(np.unique(np.concatenate([g["fit"], g["unseen"]])), a.nb, rng)
    ood_blocks = _blocks(np.unique(g["ood"]), a.nb, rng)
    th_id = []
    for i, b in enumerate(id_blocks):
        th_id.append(theta(fits(~np.isin(g["fit"], b), ~np.isin(g["seen"], b), ~np.isin(g["unseen"], b))))
        if i % 10 == 9:
            print("jk id", i + 1, "/", len(id_blocks), round(time.time() - t0), flush=True)
    th_ood = [theta(sc0, ~np.isin(g["ood"], b)) for b in ood_blocks]
    var = _jk_var_weighted(th, np.array(th_id), np.array([len(b) for b in id_blocks], float)) + \
        _jk_var_weighted(th, np.array(th_ood), np.array([len(b) for b in ood_blocks], float))
    se = np.sqrt(var)

    brng = np.random.default_rng(2)
    ms, mu, w, keep = sc0["cm"]
    wfull = np.zeros(len(ms))
    wfull[ms] = w
    grp_idx = {k: [np.flatnonzero(g[k] == u) for u in np.unique(g[k])] for k in ("seen", "unseen", "ood")}
    pos = [np.flatnonzero(fold_seen == k) for k in (0, 1)]
    bt = []
    for _ in range(a.boot):
        i_s, i_u, i_o = (np.concatenate([grp_idx[k][j] for j in brng.integers(0, len(grp_idx[k]), len(grp_idx[k]))])
                         for k in ("seen", "unseen", "ood"))
        so = sc0["ood"][i_o]
        leaky, truth = auroc(sc0["seen"][i_s], so), auroc(sc0["unseen"][i_u], so)
        num = den = 0.0
        for k in (0, 1):
            loc = np.searchsorted(pos[k], i_s[fold_seen[i_s] == k])
            if len(loc):
                num += len(loc) * auroc(sc0["fold"][k][0][loc], sc0["fold"][k][1][i_o])
                den += len(loc)
        f2 = num / den
        ks_, ku_ = i_s[ms[i_s]], i_u[mu[i_u]]
        cm = auroc(sc0["seen"][ks_], so, wfull[ks_]) - auroc(sc0["unseen"][ku_], so) if keep and len(ks_) and len(ku_) else np.nan
        row = [leaky, f2, leaky - f2, truth, leaky - truth, cm]
        if gap_mask is not None:
            lg = auroc(sc0["seen_gap"][i_s], sc0["ood_gap"][i_o])
            row += [lg, lg - f2, (leaky - f2) - (lg - f2)]
        bt.append(row)
    bt = np.array(bt)

    z95 = NormalDist().inv_cdf(0.975)
    groups_fit_fold = [int(len(np.setdiff1d(np.unique(g["fit"]), np.unique(g["seen"][fold_seen == k])))) for k in (0, 1)]
    res = {"ds": a.ds, "variant": a.variant, "candidate": V["candidate"], "run": a.run, "scorer": a.scorer,
           "d": int(X["fit"].shape[1]), "K": 2, "classes": classes,
           "n": {"fit": int(len(X["fit"])), "seen": int(len(X["seen"])), "seen_dropped_no_fit_group": int((~ok).sum()),
                 "unseen": int(len(X["unseen"])), "ood": no,
                 **({"fit_gap_removed": int(gap_mask.sum())} if gap_mask is not None else {})},
           "n_groups": {k: int(len(np.unique(g[k]))) for k in g},
           "n_groups_fit_per_fold": groups_fit_fold,
           "n_fit_per_fold": [int((~np.isin(g["fit"], np.unique(g["seen"][fold_seen == k]))).sum()) for k in (0, 1)],
           "case_mix": {"strata_kept": keep, "seen_share_dropped": float(1 - ms.mean()),
                        "unseen_share_dropped": float(1 - mu.mean()),
                        "ess_seen_weighted": float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else None},
           "package_check": chk, "n_blocks_id": len(id_blocks), "n_blocks_ood": len(ood_blocks), "bootstrap_B": a.boot,
           "stats": {}}
    for i, s in enumerate(stats):
        res["stats"][s] = {"est": float(th[i]), "jk_se": float(se[i]),
                           "jk_ci95": [float(th[i] - z95 * se[i]), float(th[i] + z95 * se[i])],
                           "boot_ci95_ref": [float(np.nanpercentile(bt[:, i], 2.5)), float(np.nanpercentile(bt[:, i], 97.5))]}
    x_, y_ = th[0], th[3]
    near = bool(0.45 <= x_ <= 0.55 and 0.45 <= y_ <= 0.55)
    res["flags"] = {"near_chance": near, "ceiling": bool(x_ > 0.98 and y_ > 0.98),
                    "below_chance": bool(min(x_, y_) < 0.5 and not near),
                    "ood_side_ci_unreliable": bool(res["n_groups"]["ood"] < 5),
                    "a_gap_underpowered": bool(len(X["unseen"]) < 200 or res["n_groups"]["unseen"] < 20)}
    if a.run in FMS and a.scorer == "mahalanobis_l2":
        pr = probe(X["fit"], y["fit"], X["seen"], y["seen"], X["unseen"], y["unseen"], classes)
        pj = {"ds": a.ds, "variant": a.variant, "candidate": V["candidate"], "fm": a.run,
              "probe": "LogisticRegression(C=1.0, lbfgs, max_iter=2000) one-vs-rest per ID class, raw features, fit images",
              "macro_auroc": {k: v["macro"] for k, v in pr.items()}, "acc": {k: v["acc"] for k, v in pr.items()},
              "per_class_auroc": {k: v["per_class"] for k, v in pr.items()},
              "G_competence": {"probe_seen_macro_auroc_ge_0.70": pr["seen"]["macro"] >= 0.70, "pass": pr["seen"]["macro"] >= 0.70}}
        (P2D / "train").mkdir(parents=True, exist_ok=True)
        (P2D / "train" / f"{a.ds}_{a.variant}_{a.run}_probe{sm}.json").write_text(json.dumps(pj, indent=2) + "\n")
    res["seconds"] = round(time.time() - t0, 1)
    out = P2D / "cells"
    out.mkdir(parents=True, exist_ok=True)
    tag = sm + ("" if a.nb == NB else f"_nb{a.nb}")
    (out / f"{a.ds}_{a.variant}_{a.run}_{a.scorer}{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({s: round(res["stats"][s]["est"], 4) for s in stats}), res["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
