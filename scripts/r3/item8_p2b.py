#!/usr/bin/env python3
"""R3 item 8 / P2-b (results/r3/8/PRECOMMIT.json): scorer and model channels on the medbench (B) arm, cached features.

One (dataset, arch, seed, fold) per call, scorers mahalanobis_l2 and knn_mean_cosine (crossfit_ood, Track A
definitions, CPU). Per cell, with the fold's model fixed:
  leaky = AUROC(seen vs OOD), truth = AUROC(unseen vs OOD), scorer fitted on all of the fold's training images;
  F2    = crossfit_auroc(protocol="default") on the seen images: within-S folds = medbench fold rule
          (medbench_phase0.assign_folds over the training groups, default_rng(0)); fit k drops the training images
          of the groups of fold k's seen images; matched pooling (n_k weights).
  scorer channel = leaky - F2, model channel = F2 - truth, fraction closed = (leaky - F2) / (leaky - truth).
Jackknife (refit-aware, grouped delete-a-block as in crossfit_ood, Busing weighted): ID units = groups of
train / seen / unseen, deleted together with all fits redone (fold map held fixed); OOD samples in blocks, scores
fixed; 50 blocks each; V = V_id + V_ood, 95% normal CI for every statistic.
leaky and F2 point estimates are checked against crossfit_auroc (1e-9, plus one pair per near-tied seen / OOD score pair).
Orphan seen images (group with no training image: the medbench val carve-out took the group's last training image)
are dropped from seen, as in the package paper_2fold (R3 item 1); counts recorded per cell.
Writes results/r3/8/p2b/cells/{ds}_{arch}_s{seed}_{fold}.json
--ood-clean (post hoc sensitivity, results/r3/8/ood_overlap/audit.json): OOD restricted to images whose group is
absent from every ID set of the fold (train, val, seen, unseen); writes results/r3/8/p2b/cells_ood_clean/.
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
sys.path.insert(0, str(REPO / "scripts/rigor"))
from medbench_phase0 import assign_folds  # noqa: E402

FEAT = REPO / "outputs/rigor_pack/medbench"
SPL = REPO / "outputs/reports/rigor_pack/medbench/splits"
OUT = REPO / "results/r3/8/p2b/cells"
SCORERS = ("mahalanobis_l2", "knn_mean_cosine")
NB = 50


def groups_of(ds, keys):
    d = pd.read_csv(SPL / f"{ds}.csv.gz", low_memory=False)
    m = dict(zip("isic:" + d.image, d.group)) if ds == "isic2019" else dict(zip(d.version + ":" + d.key, d.group))
    out = np.array([str(m.get(k, "")) for k in keys])
    assert (out != "").all(), "keys without group"
    return out


def auroc(s_id, s_ood):
    return float(roc_auc_score(np.r_[np.ones(len(s_id)), np.zeros(len(s_ood))], np.r_[s_id, s_ood]))


def near_ties(a, b, eps=1e-12):
    b = np.sort(b)
    return int((np.searchsorted(b, a + eps, side="right") - np.searchsorted(b, a - eps, side="left")).sum())


def estimates(name, xtr, gtr, xse, gse, xun, xood, fold_of_seen_group):
    from crossfit_ood import get_scorer
    full = get_scorer(name).fit(xtr)
    so = np.asarray(full.score(xood), dtype=np.float64)
    leaky = auroc(np.asarray(full.score(xse), dtype=np.float64), so)
    truth = auroc(np.asarray(full.score(xun), dtype=np.float64), so)
    fse = np.array([fold_of_seen_group[g] for g in gse])
    num, den = 0.0, 0
    for k in (0, 1):
        drop = sorted(set(gse[fse == k]))
        f = get_scorer(name).fit(xtr[~np.isin(gtr, drop)])
        n = int((fse == k).sum())
        num += n * auroc(np.asarray(f.score(xse[fse == k]), dtype=np.float64), np.asarray(f.score(xood), dtype=np.float64))
        den += n
    F2 = num / den
    return np.array([leaky, truth, F2, leaky - F2, F2 - truth]), (full, so)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["kermany", "isic2019"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", required=True, choices=["b0", "b1"])
    ap.add_argument("--ood-clean", action="store_true")
    a = ap.parse_args()
    out_dir = OUT.parent / "cells_ood_clean" if a.ood_clean else OUT
    from crossfit_ood import crossfit_auroc
    from crossfit_ood.core import _blocks, _jk_var_weighted

    z = np.load(FEAT / a.ds / ("%s_s%d_%s.npz" % (a.arch, a.seed, a.fold)))
    xtr, xse, xun, xood = (np.asarray(z[k + "_feats"], dtype=np.float64) for k in ("train", "seen", "unseen", "ood"))
    gtr, gse, gun = (groups_of(a.ds, z[k + "_keys"]) for k in ("train", "seen", "unseen"))
    n_ood_all = len(xood)
    if a.ood_clean:
        d = pd.read_csv(SPL / f"{a.ds}.csv.gz", low_memory=False)
        col = f"b_f{a.fold[1]}"
        if a.ds == "kermany":
            d = d[(d.version == "v3") & (d.label != "CXR")]
        id_g = set(d.group[d[col].isin(["train", "val", "seen", "unseen"])].astype(str))
        keep = ~np.isin(groups_of(a.ds, z["ood_keys"]), list(id_g))
        audit = json.loads((REPO / "results/r3/8/ood_overlap/audit.json").read_text())[f"{a.ds}_{a.fold}"]
        assert int(keep.sum()) == audit["clean_ood_images"], (int(keep.sum()), audit["clean_ood_images"])
        xood = xood[keep]
    orphan = ~np.isin(gse, gtr)
    n_orphan, n_orphan_groups = int(orphan.sum()), int(len(np.unique(gse[orphan])))
    xse, gse = xse[~orphan], gse[~orphan]
    assert set(gse) <= set(gtr) and not (set(gun) & set(gtr))
    fold_map = assign_folds(gtr, np.asarray(z["train_labels"]), np.random.default_rng(0))
    q = pd.read_csv(REPO / "outputs/reports/rigor_pack/medbench/model_quality.csv")
    q = q[(q.dataset == a.ds) & (q.arch == a.arch) & (q.seed == a.seed) & (q.arm == a.fold)].iloc[0]
    res = {"ds": a.ds, "arch": a.arch, "seed": a.seed, "fold": a.fold, "d": int(xtr.shape[1]),
           "n_train": len(xtr), "n_seen": len(xse), "n_orphan_seen_dropped": n_orphan,
           "n_orphan_seen_groups": n_orphan_groups, "n_unseen": len(xun), "n_ood": len(xood), "n_ood_before_restriction": n_ood_all, "ood_clean": a.ood_clean,
           "n_groups_train": int(len(set(gtr))), "n_groups_seen": int(len(set(gse))), "n_groups_unseen": int(len(set(gun))),
           "K_within": 2, "acc_seen": float(q.acc_seen), "acc_unseen": float(q.acc_unseen),
           "G1": bool(q.G1), "G2": bool(q.G2), "passes_quality": bool(q.passes), "scorers": {}}
    fids = np.array([fold_map[g] for g in gse])
    rng = np.random.default_rng(1)
    id_units = np.unique(np.concatenate([gtr, gun]))
    id_blocks = _blocks(id_units, NB, rng)
    ood_blocks = _blocks(np.arange(len(xood)), NB, rng)
    z95 = NormalDist().inv_cdf(0.975)
    for sc in SCORERS:
        t0 = time.time()
        th, (full, so) = estimates(sc, xtr, gtr, xse, gse, xun, xood, fold_map)
        rep = crossfit_auroc(xtr, gtr, xse, gse, xood, scorers=[sc], protocol="default", n_splits=2,
                             fold_ids=fids, uncertainty="bootstrap", n_bootstrap=0, random_state=0)
        r = rep.results[sc]
        s_se_full = np.asarray(full.score(xse), dtype=np.float64)
        s_un_full = np.asarray(full.score(xun), dtype=np.float64)
        fse = fids
        arm = []
        for k in (0, 1):
            from crossfit_ood import get_scorer
            f = get_scorer(sc).fit(xtr[~np.isin(gtr, sorted(set(gse[fse == k])))])
            arm.append((np.asarray(f.score(xse[fse == k]), dtype=np.float64), np.asarray(f.score(xood), dtype=np.float64)))
        # Pixel-identical seen / OOD images give exactly tied scores; the tie can break differently between the two
        # code paths (BLAS blocking), moving AUROC by one pair. Allowed slack: 1 / (n_seen * n_ood) per near-tied pair.
        t_lk = near_ties(s_se_full, so)
        t_f2 = [near_ties(*arm[k]) for k in (0, 1)]
        n_arm = [len(arm[k][0]) for k in (0, 1)]
        tol_lk = 1e-9 + t_lk / (len(xse) * len(xood))
        tol_f2 = 1e-9 + sum(n_arm[k] / sum(n_arm) * t_f2[k] / (n_arm[k] * len(xood)) for k in (0, 1))
        chk = {"leaky_diff": abs(r.auroc_leaky - th[0]), "F2_diff": abs(r.auroc_crossfit - th[2]),
               "leaky_package": float(r.auroc_leaky), "F2_package": float(r.auroc_crossfit),
               "near_tied_pairs_leaky": t_lk, "near_tied_pairs_F2": t_f2, "tol_leaky": tol_lk, "tol_F2": tol_f2}
        assert chk["leaky_diff"] <= tol_lk and chk["F2_diff"] <= tol_f2, chk
        th_id = []
        for b in id_blocks:
            kt, ks, ku = ~np.isin(gtr, b), ~np.isin(gse, b), ~np.isin(gun, b)
            th_id.append(estimates(sc, xtr[kt], gtr[kt], xse[ks], gse[ks], xun[ku], xood, fold_map)[0])
        th_id = np.array(th_id)
        th_ood = []
        for b in ood_blocks:
            keep = ~np.isin(np.arange(len(xood)), b)
            lk, tr_ = auroc(s_se_full, so[keep]), auroc(s_un_full, so[keep])
            n = [len(arm[k][0]) for k in (0, 1)]
            f2 = sum(n[k] * auroc(arm[k][0], arm[k][1][keep]) for k in (0, 1)) / sum(n)
            th_ood.append([lk, tr_, f2, lk - f2, f2 - tr_])
        th_ood = np.array(th_ood)
        m_id = np.array([len(b) for b in id_blocks], dtype=float)
        m_ood = np.array([len(b) for b in ood_blocks], dtype=float)
        sd = np.sqrt(_jk_var_weighted(th, th_id, m_id) + _jk_var_weighted(th, th_ood, m_ood))
        names = ("leaky", "truth", "F2", "scorer_channel", "model_channel")
        out = {n: {"est": float(th[i]), "se": float(sd[i]), "ci95": [float(th[i] - z95 * sd[i]), float(th[i] + z95 * sd[i])]}
               for i, n in enumerate(names)}
        gap = th[0] - th[1]
        out["leaky_minus_truth"] = float(gap)
        out["fraction_closed"] = float((th[0] - th[2]) / gap) if abs(gap) >= 0.01 else None
        flags = {}
        for nm, (x, y) in {"seen_vs_unseen": (th[0], th[1])}.items():
            flags["near_chance"] = bool(0.45 <= x <= 0.55 and 0.45 <= y <= 0.55)
            flags["ceiling"] = bool(x > 0.98 and y > 0.98)
            flags["below_chance"] = bool(min(x, y) < 0.5 and not flags["near_chance"])
        out.update(flags=flags, package_check=chk, n_blocks_id=len(id_blocks), n_blocks_ood=len(ood_blocks),
                   seconds=round(time.time() - t0, 1))
        res["scorers"][sc] = out
        print(a.ds, a.arch, a.seed, a.fold, sc, {n: round(out[n]["est"], 4) for n in names}, out["seconds"], flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("%s_%s_s%d_%s.json" % (a.ds, a.arch, a.seed, a.fold))).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
