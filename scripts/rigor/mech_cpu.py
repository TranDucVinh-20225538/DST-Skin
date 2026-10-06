#!/usr/bin/env python3
"""CPU part of decisions/precommit_leak_mechanism_fix_2026-10-06.md for one (dataset, arch) cell.

M2 group decodability, M3 same-group neighbour fraction, centroid distance (P candidates),
M4 instance vs group for the 4 fit scores (published model), F1 reference vector, F2 cross-fitted
scorer, F3a group centring, F3b batch centring, F4 kNN with same-group exclusion.
--m4-logit instead runs M4 for MSP / Energy on the retrained seed-42 models of the dataset.
Published scorer config, float64 features, calc_auroc. Δ = AUROC_same − AUROC_disjoint (> 0 = inflation).
Writes outputs/reports/rigor_pack/mechanism_fix/cells/{ds}_{arch}.json (or m4logit_{ds}_{arch}.json).
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

FIT = ("Mahalanobis", "kNN", "ViM", "ReAct")
RNG_SEED = 20261006
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/reports/rigor_pack/mechanism_fix/cells"


def load_cell(ds: str, arch: str) -> dict:
    if ds == "camelyon":
        import torch
        from leakfree_fit_scores import REPRO_FOLD_SIZES, folds
        d = torch.load(C.feature_path(REPO, "camelyon17", arch, 42, indexed=True), map_location="cpu",
                       weights_only=False)
        g = lambda k: np.asarray(d[k].numpy() if hasattr(d[k], "numpy") else d[k])  # noqa: E731
        meta = C.load_camelyon_metadata(REPO)
        tsl, vsl = g("train_slide"), g("val_slide")
        tf, vf = folds(tsl, meta.center.to_numpy()[g("train_idx")], vsl, 42)
        if ((tf == 0).sum(), (tf == 1).sum()) != REPRO_FOLD_SIZES:
            raise SystemExit("STOP: H11c fold sizes differ")
        z = {"train_feats": g("train_feats"), "train_logits": g("train_logits"), "train_group": tsl,
             "train_fold": tf, "id_feats": g("val_feats"), "id_logits": g("val_logits"), "id_group": vsl,
             "id_fold": vf, "ood_feats": g("ood_feats"), "ood_logits": g("ood_logits"), "ood_group": g("ood_slide"),
             "fc_weight": g("fc_weight"), "fc_bias": g("fc_bias")}
    else:
        import multibench_common as M
        f = np.load(M.feat_dir(ds) / ("%s_s42_full.npz" % arch))
        z = {k: f[k] for k in ("train_feats", "train_logits", "train_group", "train_fold", "id_feats", "id_logits",
                               "id_group", "id_fold", "ood_feats", "ood_logits", "ood_group", "fc_weight", "fc_bias")}
    for k in list(z):
        if k.endswith(("_feats", "_logits")) or k.startswith("fc_"):
            z[k] = np.asarray(z[k], dtype=np.float64)
    for k in ("train_group", "id_group", "ood_group"):
        z[k] = np.asarray(z[k]).astype(str)
    return z


def fit(feats, logits, w, b):
    from src.utils.scoring import OODScorer
    sc = OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None)
    sc.fit(feats, train_logits=logits, fc_weight=w, fc_bias=b)
    return sc


def scores(sc, logits, feats) -> dict:
    s = sc.get_all_scores(logits, feats)
    return {C.DISPLAY[k]: v for k, v in s.items() if k in C.DISPLAY}


def auroc(a, b) -> float:
    from src.utils.benchmark_metrics import calc_auroc
    return float(calc_auroc(a, b))


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def knn_brute(train_n, query_n, k=50, q_group=None, t_group=None, chunk=2048):
    """Mean cosine distance to the k nearest train rows (float32 torch), optionally ignoring train rows
    of the query's own group; also returns the neighbour indices without exclusion."""
    import torch
    T = torch.from_numpy(train_n.astype(np.float32))
    tg = None
    if q_group is not None:
        codes = {g: i for i, g in enumerate(np.unique(np.concatenate([t_group, q_group])))}
        tg = torch.from_numpy(np.array([codes[g] for g in t_group]))
        qg = np.array([codes[g] for g in q_group])
    out, idx = np.empty(len(query_n)), np.empty((len(query_n), k), dtype=np.int64)
    for i in range(0, len(query_n), chunk):
        q = torch.from_numpy(query_n[i:i + chunk].astype(np.float32))
        sim = q @ T.T
        top = sim.topk(k, dim=1)
        idx[i:i + chunk] = top.indices.numpy()
        if tg is not None:
            sim[torch.from_numpy(qg[i:i + chunk])[:, None] == tg[None, :]] = -np.inf
            top = sim.topk(k, dim=1)
        out[i:i + chunk] = (1.0 - top.values.double()).mean(1).numpy()
    return -out, idx


def centre(feats, keys):
    out = feats.copy()
    for g in np.unique(keys):
        m = keys == g
        out[m] -= feats[m].mean(0)
    return out


def batch_keys(n, rng, bs=256):
    perm = rng.permutation(n)
    k = np.empty(n, dtype=np.int64)
    k[perm] = np.arange(n) // bs
    return k.astype(str)


def fit_scores_on(z, tr, idf, oof):
    w, b = z["fc_weight"], z["fc_bias"]
    sc = fit(tr, tr @ w.T + b, w, b)
    si, so = scores(sc, idf @ w.T + b, idf), scores(sc, oof @ w.T + b, oof)
    return {m: auroc(si[m], so[m]) for m in FIT}


def run_cell(ds: str, arch: str) -> dict:
    t0 = time.time()
    z = load_cell(ds, arch)
    w, b = z["fc_weight"], z["fc_bias"]
    tf, fi = z["train_fold"], z["id_fold"]
    keep = fi >= 0
    res = {"ds": ds, "arch": arch, "seed": 42, "n_train": int(len(tf)), "n_id": int(keep.sum()),
           "n_ood": int(len(z["ood_feats"])), "n_id_excluded": int((~keep).sum())}

    # standard + 2-fold (Δ, F1, F2)
    sc = fit(z["train_feats"], z["train_logits"], w, b)
    si, so = scores(sc, z["id_logits"], z["id_feats"]), scores(sc, z["ood_logits"], z["ood_feats"])
    res["standard"] = {m: auroc(si[m][keep], so[m]) for m in C.METHODS_ORDER}
    del sc
    same, dis = {m: [] for m in C.METHODS_ORDER}, {m: [] for m in C.METHODS_ORDER}
    f2_id = {m: np.full(len(fi), np.nan) for m in FIT}
    f2_ood = {m: np.zeros(len(z["ood_feats"])) for m in FIT}
    rng = np.random.default_rng(RNG_SEED)
    m4 = {m: {"seen_inst": [], "unseen_inst_seen_group": [], "unseen_group": []} for m in FIT}
    for f in (0, 1):
        sel = tf == f
        sc = fit(z["train_feats"][sel], z["train_logits"][sel], w, b)
        si, so = scores(sc, z["id_logits"], z["id_feats"]), scores(sc, z["ood_logits"], z["ood_feats"])
        for m in C.METHODS_ORDER:
            same[m].append(auroc(si[m][fi == f], so[m]))
            dis[m].append(auroc(si[m][fi == 1 - f], so[m]))
        for m in FIT:
            f2_id[m][fi == 1 - f] = si[m][fi == 1 - f]
            f2_ood[m] += so[m] / 2.0
        del sc
        # M4 (fit scores): fit on a random half of fold-f train images
        idx = np.flatnonzero(sel)
        half = rng.permutation(idx)[: len(idx) // 2]
        sc = fit(z["train_feats"][half], z["train_logits"][half], w, b)
        sa = scores(sc, z["train_logits"][half], z["train_feats"][half])
        si, so = scores(sc, z["id_logits"], z["id_feats"]), scores(sc, z["ood_logits"], z["ood_feats"])
        for m in FIT:
            m4[m]["seen_inst"].append(auroc(sa[m], so[m]))
            m4[m]["unseen_inst_seen_group"].append(auroc(si[m][fi == f], so[m]))
            m4[m]["unseen_group"].append(auroc(si[m][fi == 1 - f], so[m]))
        del sc
    res["same_2fold"] = {m: float(np.mean(same[m])) for m in C.METHODS_ORDER}
    res["disjoint_2fold"] = {m: float(np.mean(dis[m])) for m in C.METHODS_ORDER}
    res["delta"] = {m: res["same_2fold"][m] - res["disjoint_2fold"][m] for m in FIT}
    res["F1"] = res["disjoint_2fold"]
    res["F2"] = dict(res["standard"], **{m: auroc(f2_id[m][keep], f2_ood[m]) for m in FIT})
    res["M4_fit"] = {}
    for m in FIT:
        a = {k: float(np.mean(v)) for k, v in m4[m].items()}
        den = a["seen_inst"] - a["unseen_group"]
        a["R"] = (a["unseen_inst_seen_group"] - a["unseen_group"]) / den if abs(den) >= 0.01 else None
        res["M4_fit"][m] = a
    print("2-fold / F2 / M4 done %.0fs" % (time.time() - t0), flush=True)

    # F3a group centring, F3b batch centring (full-train fit, all ID)
    tr, idf, oof = z["train_feats"], z["id_feats"][keep], z["ood_feats"]
    gi = z["id_group"][keep]
    res["F3a"] = dict(res["standard"], **fit_scores_on(
        z, centre(tr, z["train_group"]), centre(idf, gi), centre(oof, z["ood_group"])))
    rb = np.random.default_rng(RNG_SEED)
    res["F3b"] = dict(res["standard"], **fit_scores_on(
        z, centre(tr, batch_keys(len(tr), rb)), centre(idf, batch_keys(len(idf), rb)),
        centre(oof, batch_keys(len(oof), rb))))
    print("F3 done %.0fs" % (time.time() - t0), flush=True)

    # F4 kNN with same-group exclusion; M3 neighbour fraction from the same search
    trn, idn, oon = l2n(tr), l2n(idf), l2n(oof)
    kid_ex, nn = knn_brute(trn, idn, q_group=gi, t_group=z["train_group"])
    kood, _ = knn_brute(trn, oon)
    kid_std, _ = knn_brute(trn, idn)
    res["F4"] = dict(res["standard"], kNN=auroc(kid_ex, kood))
    res["F4_check_brute_std_kNN"] = auroc(kid_std, kood)
    tg = z["train_group"]
    frac = (tg[nn] == gi[:, None]).mean(1)
    share = {g: c / len(tg) for g, c in zip(*np.unique(tg, return_counts=True))}
    chance = np.array([share.get(g, 0.0) for g in gi])
    res["M3"] = {"same_group_nn_frac": float(frac.mean()), "chance": float(chance.mean()),
                 "excess": float((frac - chance).mean())}
    # centroid distance (L2-normalised features)
    dists = []
    for g in np.unique(gi):
        if g in share:
            a, c = trn[tg == g].mean(0), idn[gi == g].mean(0)
            dists.append(1.0 - a @ c / (np.linalg.norm(a) * np.linalg.norm(c)))
    res["centroid_distance"] = float(np.mean(dists))
    print("F4 / M3 done %.0fs" % (time.time() - t0), flush=True)

    # M2 group decodability
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([tr, idf])
    y = np.concatenate([tg, gi])
    r2 = np.random.default_rng(RNG_SEED)
    n = min(20000, len(y))
    pick = []
    for g, c in zip(*np.unique(y, return_counts=True)):
        k = int(round(n * c / len(y)))
        pick.append(r2.choice(np.flatnonzero(y == g), size=min(k, c), replace=False))
    pick = np.concatenate(pick)
    Xs, ys = X[pick], y[pick]
    gs, cs = np.unique(ys, return_counts=True)
    ok = np.isin(ys, gs[cs >= 5])
    Xs, ys = Xs[ok], ys[ok]
    pred = np.empty(len(ys), dtype=ys.dtype)
    for trI, teI in StratifiedKFold(5, shuffle=True, random_state=RNG_SEED).split(Xs, ys):
        ss = StandardScaler().fit(Xs[trI])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(ss.transform(Xs[trI]), ys[trI])
        pred[teI] = clf.predict(ss.transform(Xs[teI]))
    ng = len(np.unique(ys))
    res["M2"] = {"balanced_acc": float(balanced_accuracy_score(ys, pred)), "n_groups": int(ng),
                 "chance": 1.0 / ng, "n_images": int(len(ys)), "n_groups_excluded_lt5": int((cs < 5).sum())}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def run_m4_logit(ds: str, arch: str) -> dict:
    t0 = time.time()
    res = {"ds": ds, "arch": arch, "seed": 42}
    acc = {m: {"seen_inst": [], "unseen_inst_seen_group": [], "unseen_group": []} for m in ("MSP", "Energy")}
    from src.utils.scoring import OODScorer
    for f in (0, 1):
        if ds == "camelyon":
            z = np.load(REPO / "outputs/rigor_pack/logit_retrain_slide_disjoint_v2" / ("%s_s42_f%d.npz" % (arch, f)))
            idl, fold = z["val_logits"], z["id_val_fold"]
        else:
            import multibench_common as M
            z = np.load(M.feat_dir(ds) / ("%s_s42_f%d.npz" % (arch, f)))
            idl, fold = z["id_logits"], z["id_fold"]
        for m, fn in (("MSP", OODScorer.score_msp), ("Energy", OODScorer.score_energy)):
            so = fn(z["ood_logits"])
            acc[m]["seen_inst"].append(auroc(fn(z["train_logits"]), so))
            acc[m]["unseen_inst_seen_group"].append(auroc(fn(idl[fold == f]), so))
            acc[m]["unseen_group"].append(auroc(fn(idl[fold == 1 - f]), so))
    for m in acc:
        a = {k: float(np.mean(v)) for k, v in acc[m].items()}
        den = a["seen_inst"] - a["unseen_group"]
        a["R"] = (a["unseen_inst_seen_group"] - a["unseen_group"]) / den if abs(den) >= 0.01 else None
        res[m] = a
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["camelyon", "iwildcam", "rxrx1"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--m4-logit", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.m4_logit:
        res, name = run_m4_logit(args.ds, args.arch), "m4logit_%s_%s.json" % (args.ds, args.arch)
    else:
        res, name = run_cell(args.ds, args.arch), "%s_%s.json" % (args.ds, args.arch)
    (OUT / name).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
