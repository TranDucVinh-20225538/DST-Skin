#!/usr/bin/env python3
"""R3 item 8 / P2-a scoring for one (run, scorer) cell (results/r3/8/PRECOMMIT.json; M_std = wang2017).

run = resnet50_s42 | resnet50_s43 | convnext_tiny_s42 | convnext_tiny_s43 | dinov2_vitb14 | rad_dino;
scorer = mahalanobis_l2 | knn_mean_cosine (crossfit_ood, Track A definitions, CPU, float64).
Per OOD set (shenzhen = primary, all 662; kermany_ped = secondary; shenzhen_tb / shenzhen_normal = descriptive):
  leaky  = AUROC(seen vs OOD), scorer fitted on all fit images
  F2     = A-fit cross-fit on the seen images: fold of a seen image = afit_fold of its patient (medbench fold rule);
           fit k drops the fit images of the patients of fold k's seen images; matched pooling (n_k weights)
  Delta_fit = leaky - F2
  truth  = AUROC(unseen vs OOD), full fit;  A_gap = leaky - truth
  A_gap_cm = case-mix-corrected A-gap: seen images post-stratified to the unseen distribution over
           View Position (AP, PA) x finding pattern ('No Finding'; each single finding; >= 2 findings);
           weight = p_unseen(stratum) / p_seen(stratum); strata with < 5 images in either set dropped (from both sets);
           weighted AUROC(seen vs OOD) - AUROC(unseen in kept strata vs OOD)
Uncertainty, PRIMARY: refit-aware grouped jackknife (crossfit_ood _blocks / _jk_var_weighted, Busing): ID units =
patients of fit and unseen (seen patients are fit patients), 50 blocks deleted together with every refit; OOD: 50
blocks over all OOD images with scores fixed; V = V_id + V_ood; 95% normal CI.
REFERENCE ONLY: paired cluster bootstrap (B = 2000; seen / unseen resampled by patient, OOD by image within source),
fitted scores and case-mix weights held fixed.
leaky and F2 on shenzhen are checked against crossfit_auroc (1e-9).
Writes results/r3/8/p2a/cells/{run}_{scorer}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
P2A = REPO / "results/r3/8/p2a"
FEAT = REPO / "outputs/rigor_pack/r3_item8/p2a"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
OODS = ("shenzhen", "kermany_ped", "shenzhen_tb", "shenzhen_normal")
STATS = ("leaky", "F2", "delta_fit", "truth", "a_gap", "a_gap_cm")
NB, B, MIN_STRATUM = 50, 2000, 5


def auroc(s_id, s_ood, w_id=None):
    y = np.r_[np.ones(len(s_id)), np.zeros(len(s_ood))]
    w = None if w_id is None else np.r_[w_id, np.ones(len(s_ood))]
    return float(roc_auc_score(y, np.r_[s_id, s_ood], sample_weight=w))


def strata(df):
    nf = df[FINDINGS].to_numpy().sum(1)
    pat = np.where(nf == 0, "NoFinding", np.where(nf >= 2, "multi", ""))
    single = np.array(FINDINGS)[df[FINDINGS].to_numpy().argmax(1)]
    pat = np.where(nf == 1, single, pat)
    return np.char.add(np.char.add(df.view.to_numpy().astype(str), "|"), pat.astype(str))


def cm_weights(st_seen, st_un):
    cs, cu = pd.Series(st_seen).value_counts(), pd.Series(st_un).value_counts()
    keep = sorted(s for s in set(cs.index) & set(cu.index) if cs[s] >= MIN_STRATUM and cu[s] >= MIN_STRATUM)
    ms, mu = np.isin(st_seen, keep), np.isin(st_un, keep)
    ps, pu = cs[keep] / ms.sum(), cu[keep] / mu.sum()
    w = (pu / ps).reindex(st_seen[ms]).to_numpy()
    return ms, mu, w, keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scorer", required=True, choices=["mahalanobis_l2", "knn_mean_cosine"])
    ap.add_argument("--nb", type=int, default=NB, help="jackknife blocks (smoke: fewer)")
    a = ap.parse_args()
    from crossfit_ood import crossfit_auroc, get_scorer
    from crossfit_ood.core import _blocks, _jk_var_weighted

    t0 = time.time()
    z = np.load(FEAT / f"{a.run}.npz", allow_pickle=True)
    zo = np.load(FEAT / f"{a.run}_ood.npz", allow_pickle=True) if (FEAT / f"{a.run}_ood.npz").exists() else z
    S = pd.read_csv(P2A / "sets.csv.gz").set_index("image")
    O = pd.read_csv(P2A / "ood.csv.gz", keep_default_na=False).set_index("key")
    X = {k: np.asarray(z[f"{k}_feats"], dtype=np.float64) for k in ("fit", "seen", "unseen")}
    X["ood"] = np.asarray(zo["ood_feats"], dtype=np.float64)
    keys = {k: np.asarray(z[f"{k}_keys"]).astype(str) for k in ("fit", "seen", "unseen")}
    okeys = np.asarray(zo["ood_keys"]).astype(str)
    for k in keys:
        assert (S.loc[keys[k], "role"] == k).all(), k
    pid = {k: S.loc[keys[k], "patient"].to_numpy() for k in keys}
    fold_seen = S.loc[keys["seen"], "afit_fold"].to_numpy()
    assert set(pid["seen"]) <= set(pid["fit"]) and not (set(pid["unseen"]) & set(pid["fit"])) and (fold_seen >= 0).all()
    st_seen, st_un = strata(S.loc[keys["seen"]]), strata(S.loc[keys["unseen"]])
    src = O.loc[okeys, "source"].to_numpy()
    tb = O.loc[okeys, "shenzhen_tb"].to_numpy()
    oidx = {"shenzhen": np.flatnonzero(src == "shenzhen"), "kermany_ped": np.flatnonzero(src == "kermany_ped"),
            "shenzhen_tb": np.flatnonzero((src == "shenzhen") & (tb == 1)),
            "shenzhen_normal": np.flatnonzero((src == "shenzhen") & (tb == 0))}
    assert len(oidx["shenzhen"]) == 662 and len(oidx["kermany_ped"]) == 5856

    def fits(kf, ks, ku):
        full = get_scorer(a.scorer).fit(X["fit"][kf])
        sc = {"seen": np.asarray(full.score(X["seen"][ks]), dtype=np.float64),
              "unseen": np.asarray(full.score(X["unseen"][ku]), dtype=np.float64),
              "ood": np.asarray(full.score(X["ood"]), dtype=np.float64), "fold": []}
        ps, fs, pf = pid["seen"][ks], fold_seen[ks], pid["fit"][kf]
        for k in (0, 1):
            f = get_scorer(a.scorer).fit(X["fit"][kf][~np.isin(pf, np.unique(ps[fs == k]))])
            sc["fold"].append((np.asarray(f.score(X["seen"][ks][fs == k]), dtype=np.float64),
                               np.asarray(f.score(X["ood"]), dtype=np.float64)))
        sc["cm"] = cm_weights(st_seen[ks], st_un[ku])
        return sc

    def theta(sc, okeep=None):
        out = []
        ms, mu, w, _ = sc["cm"]
        for o in OODS:
            ix = oidx[o] if okeep is None else oidx[o][okeep[oidx[o]]]
            so = sc["ood"][ix]
            leaky, truth = auroc(sc["seen"], so), auroc(sc["unseen"], so)
            n = [len(sc["fold"][k][0]) for k in (0, 1)]
            f2 = sum(n[k] * auroc(sc["fold"][k][0], sc["fold"][k][1][ix]) for k in (0, 1)) / sum(n)
            cm = auroc(sc["seen"][ms], so, w) - auroc(sc["unseen"][mu], so)
            out += [leaky, f2, leaky - f2, truth, leaky - truth, cm]
        return np.array(out)

    all_f, all_s, all_u = (np.ones(len(X[k]), bool) for k in ("fit", "seen", "unseen"))
    sc0 = fits(all_f, all_s, all_u)
    th = theta(sc0)
    rep = crossfit_auroc(X["fit"], pid["fit"], X["seen"], pid["seen"], X["ood"][oidx["shenzhen"]], scorers=[a.scorer],
                         protocol="default", n_splits=2, fold_ids=fold_seen, uncertainty="bootstrap", n_bootstrap=0,
                         random_state=0).results[a.scorer]
    chk = {"leaky_diff": abs(rep.auroc_leaky - th[0]), "F2_diff": abs(rep.auroc_crossfit - th[1])}
    assert chk["leaky_diff"] < 1e-9 and chk["F2_diff"] < 1e-9, chk
    print(a.run, a.scorer, dict(zip(STATS, np.round(th[:6], 4))), round(time.time() - t0), flush=True)

    rng = np.random.default_rng(1)
    id_blocks = _blocks(np.unique(np.concatenate([pid["fit"], pid["unseen"]])), a.nb, rng)
    ood_blocks = _blocks(np.arange(len(X["ood"])), a.nb, rng)
    th_id = []
    for i, b in enumerate(id_blocks):
        th_id.append(theta(fits(~np.isin(pid["fit"], b), ~np.isin(pid["seen"], b), ~np.isin(pid["unseen"], b))))
        if i % 10 == 9:
            print("jk id", i + 1, round(time.time() - t0), flush=True)
    th_ood = [theta(sc0, ~np.isin(np.arange(len(X["ood"])), b)) for b in ood_blocks]
    var = _jk_var_weighted(th, np.array(th_id), np.array([len(b) for b in id_blocks], float)) + \
        _jk_var_weighted(th, np.array(th_ood), np.array([len(b) for b in ood_blocks], float))
    se = np.sqrt(var)

    brng = np.random.default_rng(2)
    ms, mu, w, keep = sc0["cm"]
    up_s, inv_s = np.unique(pid["seen"], return_inverse=True)
    up_u, inv_u = np.unique(pid["unseen"], return_inverse=True)
    gs = [np.flatnonzero(inv_s == j) for j in range(len(up_s))]
    gu = [np.flatnonzero(inv_u == j) for j in range(len(up_u))]
    bt = []
    for _ in range(B):
        i_s = np.concatenate([gs[j] for j in brng.integers(0, len(gs), len(gs))])
        i_u = np.concatenate([gu[j] for j in brng.integers(0, len(gu), len(gu))])
        i_o = np.concatenate([brng.choice(oidx[o], len(oidx[o])) for o in ("shenzhen", "kermany_ped")])
        row = []
        for o in OODS:
            io = i_o[np.isin(i_o, oidx[o])]
            so = sc0["ood"][io]
            leaky, truth = auroc(sc0["seen"][i_s], so), auroc(sc0["unseen"][i_u], so)
            num = den = 0.0
            for k in (0, 1):
                pos = np.flatnonzero(fold_seen == k)
                loc = np.searchsorted(pos, i_s[fold_seen[i_s] == k])
                num += len(loc) * auroc(sc0["fold"][k][0][loc], sc0["fold"][k][1][io])
                den += len(loc)
            f2 = num / den
            wfull = np.zeros(len(ms))
            wfull[ms] = w
            ks_, ku_ = i_s[ms[i_s]], i_u[mu[i_u]]
            cm = auroc(sc0["seen"][ks_], so, wfull[ks_]) - auroc(sc0["unseen"][ku_], so)
            row += [leaky, f2, leaky - f2, truth, leaky - truth, cm]
        bt.append(row)
    bt = np.array(bt)

    z95 = NormalDist().inv_cdf(0.975)
    res = {"run": a.run, "scorer": a.scorer, "d": int(X["fit"].shape[1]), "K": 2,
           "n": {"fit": int(len(X["fit"])), "seen": int(len(X["seen"])), "unseen": int(len(X["unseen"])),
                 **{o: int(len(oidx[o])) for o in OODS}},
           "n_patients": {k: int(len(np.unique(pid[k]))) for k in pid},
           "n_groups_fit_per_fold": [int(len(np.setdiff1d(np.unique(pid["fit"]), np.unique(pid["seen"][fold_seen == k]))))
                                     for k in (0, 1)],
           "n_fit_per_fold": [int((~np.isin(pid["fit"], np.unique(pid["seen"][fold_seen == k]))).sum()) for k in (0, 1)],
           "case_mix": {"strata_kept": len(keep), "seen_share_dropped": float(1 - ms.mean()),
                        "unseen_share_dropped": float(1 - mu.mean()), "ess_seen_weighted": float(w.sum() ** 2 / (w ** 2).sum()),
                        "n_seen_kept": int(ms.sum()), "n_unseen_kept": int(mu.sum())},
           "package_check": chk, "n_blocks_id": len(id_blocks), "n_blocks_ood": len(ood_blocks), "bootstrap_B": B,
           "ood": {}}
    for j, o in enumerate(OODS):
        r = {}
        for i, s in enumerate(STATS):
            q = j * len(STATS) + i
            r[s] = {"est": float(th[q]), "jk_se": float(se[q]),
                    "jk_ci95": [float(th[q] - z95 * se[q]), float(th[q] + z95 * se[q])],
                    "boot_ci95_ref": [float(np.percentile(bt[:, q], 2.5)), float(np.percentile(bt[:, q], 97.5))]}
        x, y = r["leaky"]["est"], r["truth"]["est"]
        near = bool(0.45 <= x <= 0.55 and 0.45 <= y <= 0.55)
        r["flags"] = {"near_chance": near, "ceiling": bool(x > 0.98 and y > 0.98),
                      "below_chance": bool(min(x, y) < 0.5 and not near)}
        res["ood"][o] = r
    res["seconds"] = round(time.time() - t0, 1)
    out = P2A / "cells"
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.nb == NB else f"_nb{a.nb}"
    (out / f"{a.run}_{a.scorer}{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({o: {s: round(res["ood"][o][s]["est"], 4) for s in STATS} for o in OODS}), res["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
