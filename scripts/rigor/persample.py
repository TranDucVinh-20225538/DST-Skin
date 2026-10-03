#!/usr/bin/env python3
"""Per-sample rigor analyses on the score caches (CPU; needs build_score_cache.py output).

Precommit: decisions/decision_precommit_rigor_pack.md (H3, H5b, H7, H8, H9, H10).

  1. Every cell x method: AUROC, DeLong 95% CI, patch-iid bootstrap CI, patient-cluster
     bootstrap CI (two-sample: ID clusters = id_val patients, OOD clusters = hospital-2 patients),
     AUPR-In, AUPR-Out, FPR@95TPR.
  2. Bootstrap over samples (B reps; iid and cluster; the SAME resample is shared by all
     cells, so all comparisons are paired): CIs for cross-arch W (k=8, each seed), cross-seed W
     (k=5, each arch), mean rater-matched k=5 cross-arch W, and their differences.
  3. W and feature-rank-1 counts under AUPR-In / AUPR-Out / FPR95 (metric robustness).
  4. Paired DeLong per cell: best feature score vs best other score; Mahalanobis vs kNN.
  5. MSP-jump vs accuracy: does the arch effect on MSP AUROC survive adjusting for ID-val and
     hospital-2 accuracy? (OLS over the 40 runs + run bootstrap; accuracy-matched pairs)
  6. Temperature: binary MSP AUROC is invariant to T (checked numerically), Energy is not.
  7. Coverage@risk 5/10/20 % with MSP / Energy / Mahalanobis / kNN as triage score, with
     iid and patient-cluster bootstrap CIs; DenseNet-vs-MobileNet retrospective gap per seed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TRIAGE = ("msp", "energy", "mahalanobis", "knn")


def load_cells(out_root: Path, domain: str, archs, seeds):
    cells = {}
    for a in archs:
        for s in seeds:
            p = C.score_cache_path(out_root, domain, a, s)
            if p.exists():
                z = np.load(p, allow_pickle=False)
                cells[(a, s)] = {k: z[k] for k in z.files if k != "source"}
    return cells


def clusters_for(domain: str, feat_root: Path, n_id: int, n_ood: int, id_labels, ood_labels):
    """Return (id_cluster_codes, ood_cluster_codes, ood_patient_names, note)."""
    if domain != "camelyon17":
        return None, None, None, "no cluster metadata for %s (iid only)" % domain
    meta = C.load_camelyon_metadata(feat_root)
    if meta is None:
        return None, None, None, "metadata.csv missing: cluster bootstrap skipped (iid only)"
    h2 = meta[meta.wilds_split == 2]
    iv = meta[meta.wilds_split == 1]
    if len(h2) != n_ood or not np.array_equal(h2.tumor.to_numpy(), ood_labels):
        return None, None, None, "hospital-2 metadata order does not match OOD labels: cluster bootstrap skipped"
    if len(iv) != n_id or not np.array_equal(iv.tumor.to_numpy(), id_labels):
        return None, None, None, "id_val metadata order does not match ID labels: cluster bootstrap skipped"
    _, idc = np.unique(iv.patient.to_numpy(), return_inverse=True)
    names, oc = np.unique(h2.patient.to_numpy(), return_inverse=True)
    return idc, oc, names, "clusters: %d ID patients, %d OOD patients" % (idc.max() + 1, oc.max() + 1)


def cluster_weights(codes: np.ndarray, rng) -> np.ndarray:
    k = int(codes.max()) + 1
    cnt = np.bincount(rng.integers(0, k, k), minlength=k).astype(np.float64)
    return cnt[codes]


def iid_weights(n: int, rng) -> np.ndarray:
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)


def cov_multi(sorter: C.CoverageSorter, w, targets):
    if w is None:
        w = np.ones_like(sorter.c)
    else:
        w = np.asarray(w, dtype=np.float64)[sorter.order]
    cw = np.cumsum(w)
    tot = cw[-1]
    cc = np.cumsum(w * sorter.c)
    with np.errstate(divide="ignore", invalid="ignore"):
        risk = 1.0 - cc / cw
    out = []
    for t in targets:
        ok = (cw > 0) & (risk <= t + 1e-12) & (w > 0)
        out.append(float(cw[np.nonzero(ok)[0][-1]] / tot) if np.any(ok) else 0.0)
    return out


def rankdata(x):
    try:
        from scipy.stats import rankdata as rd

        return rd(x)
    except Exception:
        return C._midrank(np.asarray(x, dtype=np.float64))


def delong_fast(id_list, ood_list):
    """Same as common.delong but with scipy midranks (fast)."""
    pos = [np.asarray(s, dtype=np.float32).astype(np.float64) for s in id_list]
    neg = [np.asarray(s, dtype=np.float32).astype(np.float64) for s in ood_list]
    m, n = len(pos[0]), len(neg[0])
    v01, v10, aucs = [], [], []
    for p_, q_ in zip(pos, neg):
        tx, ty = rankdata(p_), rankdata(q_)
        tz = rankdata(np.concatenate([p_, q_]))
        aucs.append(tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n)
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    v01, v10 = np.stack(v01), np.stack(v10)
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return np.array(aucs), cov


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--feat-root", default=None)
    ap.add_argument("--out", default=None, help="score caches root (default outputs/rigor_pack)")
    ap.add_argument("--reports", default=None)
    ap.add_argument("--domain", default="camelyon17")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-cluster", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    feat_root = Path(args.feat_root) if args.feat_root else root
    out_root = Path(args.out) if args.out else C.default_out(root)
    rep = Path(args.reports) if args.reports else C.default_reports(root) / ("persample_%s" % args.domain)
    C.ensure_dir(rep)
    rng = np.random.default_rng(args.seed)
    from sklearn.metrics import average_precision_score
    from src.utils.benchmark_metrics import calc_fpr95

    archs, seeds = list(C.ARCHS), list(C.SEEDS)
    cells = load_cells(out_root, args.domain, archs, seeds)
    log = ["persample %s  %s" % (args.domain, C.env_info()), "cells loaded: %d / %d" % (len(cells), len(archs) * len(seeds))]
    if not cells:
        C.write_text(rep / "persample_log.txt", log + ["no score caches; run build_score_cache.py"])
        print("\n".join(log))
        sys.exit(3)
    # keep complete archs/seeds grid only
    seeds = [s for s in seeds if all((a, s) in cells for a in archs) or sum((a, s) in cells for a in archs) >= 2]
    first = next(iter(cells.values()))
    n_id, n_ood = len(first["id_labels"]), len(first["ood_labels"])
    for k, c in cells.items():
        if len(c["id_labels"]) != n_id or not np.array_equal(c["ood_labels"], first["ood_labels"]):
            raise RuntimeError("cell %s has different samples; paired bootstrap impossible" % (k,))
    methods = [m for m in C.METHOD_KEYS if all(("id_" + m) in c for c in cells.values())]
    log.append("methods present in all cells: %s" % methods)
    idc, oc, onames, note = clusters_for(args.domain, feat_root, n_id, n_ood, first["id_labels"], first["ood_labels"])
    log.append(note)
    keys = sorted(cells.keys(), key=lambda k: (archs.index(k[0]), k[1]))

    # ---------- 1. point metrics + DeLong ----------
    t0 = time.time()
    rows = []
    asort, csort, correct = {}, {}, {}
    for (a, s) in keys:
        c = cells[(a, s)]
        correct[(a, s)] = (c["ood_logits"].argmax(1) == c["ood_labels"]).astype(np.float64)
        for m in methods:
            asort[(a, s, m)] = C.AurocSorter(c["id_" + m], c["ood_" + m])
        for m in TRIAGE:
            if ("ood_" + m) in c:
                csort[(a, s, m)] = C.CoverageSorter(c["ood_" + m], correct[(a, s)])
        aucs, cov = delong_fast([c["id_" + m] for m in methods], [c["ood_" + m] for m in methods])
        for j, m in enumerate(methods):
            se = float(np.sqrt(max(cov[j, j], 0)))
            y = np.r_[np.ones(n_id), np.zeros(n_ood)]
            sc = np.r_[c["id_" + m], c["ood_" + m]].astype(np.float32)
            fpr, _ = calc_fpr95(c["id_" + m], c["ood_" + m])
            rows.append({"arch": a, "seed": s, "method": C.DISPLAY[m], "auroc": asort[(a, s, m)].auroc(),
                         "auroc_delong": aucs[j], "delong_lo": aucs[j] - 1.96 * se, "delong_hi": aucs[j] + 1.96 * se,
                         "aupr_in": float(average_precision_score(y, sc)),
                         "aupr_out": float(average_precision_score(1 - y, -sc)), "fpr95": fpr})
        # paired DeLong: best feature vs best other, Maha vs kNN
        pa = {m: asort[(a, s, m)].auroc() for m in methods}
        feats = [m for m in methods if C.DISPLAY[m] in C.FEATURE_METHODS]
        others = [m for m in methods if C.DISPLAY[m] not in C.FEATURE_METHODS]
        cells[(a, s)]["_pa"] = pa
        cells[(a, s)]["_pairs"] = []
        if feats and others:
            bf, bo = max(feats, key=pa.get), max(others, key=pa.get)
            pairs = [("best_feature_vs_best_other", bf, bo)]
            if len(feats) == 2:
                pairs.append(("maha_vs_knn", "mahalanobis", "knn"))
            for name, m1, m2 in pairs:
                au, cv = delong_fast([c["id_" + m1], c["id_" + m2]], [c["ood_" + m1], c["ood_" + m2]])
                var = cv[0, 0] + cv[1, 1] - 2 * cv[0, 1]
                z = (au[0] - au[1]) / np.sqrt(var) if var > 0 else np.inf
                p2 = 2 * C.norm_sf(abs(z)) if np.isfinite(z) else 0.0
                cells[(a, s)]["_pairs"].append({"arch": a, "seed": s, "comparison": name, "m1": C.DISPLAY[m1],
                                                "m2": C.DISPLAY[m2], "auroc1": au[0], "auroc2": au[1],
                                                "diff": au[0] - au[1], "z": z, "p_two_sided": p2})
    cell_df = pd.DataFrame(rows)
    log.append("point metrics + DeLong: %.0fs" % (time.time() - t0))

    # ---------- 2+7. bootstrap ----------
    A, S, Mn = len(archs), len(seeds), len(methods)
    present = np.array([[(a, s) in cells for s in seeds] for a in archs])
    modes = ["iid"] + ([] if (idc is None or args.skip_cluster) else ["cluster"])
    boot_auc = {md: np.full((args.B, A, S, Mn), np.nan) for md in modes}
    boot_cov = {md: np.full((args.B, A, S, len(TRIAGE), len(C.RISKS)), np.nan) for md in modes}
    for md in modes:
        t0 = time.time()
        for b in range(args.B):
            if md == "iid":
                wi, wo = iid_weights(n_id, rng), iid_weights(n_ood, rng)
            else:
                wi, wo = cluster_weights(idc, rng), cluster_weights(oc, rng)
            for ai, a in enumerate(archs):
                for si, s in enumerate(seeds):
                    if not present[ai, si]:
                        continue
                    for mi, m in enumerate(methods):
                        boot_auc[md][b, ai, si, mi] = asort[(a, s, m)].auroc(wi, wo)
                    for ti, m in enumerate(TRIAGE):
                        if (a, s, m) in csort:
                            boot_cov[md][b, ai, si, ti] = cov_multi(csort[(a, s, m)], wo, C.RISKS)
        log.append("bootstrap %s B=%d: %.0fs" % (md, args.B, time.time() - t0))

    # per-cell CIs
    for md in modes:
        lo = np.nanquantile(boot_auc[md], 0.025, axis=0)
        hi = np.nanquantile(boot_auc[md], 0.975, axis=0)
        L, H = [], []
        for _, r in cell_df.iterrows():
            ai, si, mi = archs.index(r.arch), seeds.index(r.seed), methods.index(C.INV_DISPLAY[r.method])
            L.append(lo[ai, si, mi])
            H.append(hi[ai, si, mi])
        cell_df["boot_%s_lo" % md] = L
        cell_df["boot_%s_hi" % md] = H
    cell_df.to_csv(rep / "auroc_cells.csv", index=False)

    # W statistics on point + bootstrap
    point = np.full((A, S, Mn), np.nan)
    for ai, a in enumerate(archs):
        for si, s in enumerate(seeds):
            if present[ai, si]:
                point[ai, si] = [cells[(a, s)]["_pa"][m] for m in methods]

    subsets = C.all_subsets(range(A), 5)

    def w_stats(cube):
        """cube (A,S,M) -> dict of named W statistics."""
        outd = {}
        rk = np.full_like(cube, np.nan)
        for ai in range(A):
            for si in range(S):
                if not np.isnan(cube[ai, si]).any():
                    rk[ai, si] = C.ranks_high_is_1(cube[ai, si])
        for si, s in enumerate(seeds):
            ok = [ai for ai in range(A) if not np.isnan(rk[ai, si]).any()]
            if len(ok) == A:
                outd["cross_arch_k8_seed%d" % s] = C.kendall_w(rk[:, si])
                sub = np.stack([rk[list(ss), si] for ss in subsets])
                outd["crossarch_k5_mean_seed%d" % s] = float(np.mean(C.kendall_w_batch(sub, C.tie_terms(sub))))
        k5 = [v for k_, v in outd.items() if k_.startswith("crossarch_k5_mean")]
        if k5:
            outd["crossarch_k5_mean_allseeds"] = float(np.mean(k5))
        for ai, a in enumerate(archs):
            if not np.isnan(rk[ai]).any():
                outd["cross_seed_k5_%s" % a] = C.kendall_w(rk[ai])
                if k5:
                    outd["diff_crossseed_minus_crossarchk5_%s" % a] = outd["cross_seed_k5_%s" % a] - outd["crossarch_k5_mean_allseeds"]
        cs = [outd["cross_seed_k5_%s" % a] for a in archs if "cross_seed_k5_%s" % a in outd]
        if cs and k5:
            outd["diff_mean_crossseed_minus_crossarchk5"] = float(np.mean(cs)) - outd["crossarch_k5_mean_allseeds"]
        return outd

    wp = w_stats(point)
    wrows = []
    wb = {md: [w_stats(boot_auc[md][b]) for b in range(args.B)] for md in modes}
    for k in wp:
        rec = {"statistic": k, "point": wp[k]}
        for md in modes:
            v = np.array([d.get(k, np.nan) for d in wb[md]])
            rec["%s_lo" % md], rec["%s_hi" % md] = C.percentile_ci(v)
            if k.startswith("diff"):
                rec["%s_frac_gt0" % md] = float(np.nanmean(v > 0))
        wrows.append(rec)
    pd.DataFrame(wrows).to_csv(rep / "w_bootstrap.csv", index=False)

    # ---------- 3. alternative metrics ----------
    alt = []
    for metric, sign in (("auroc", 1), ("aupr_in", 1), ("aupr_out", 1), ("fpr95", -1)):
        cube = np.full((A, S, Mn), np.nan)
        for _, r in cell_df.iterrows():
            cube[archs.index(r.arch), seeds.index(r.seed), methods.index(C.INV_DISPLAY[r.method])] = sign * r[metric]
        ws = w_stats(cube)
        n_r1 = n_tot = 0
        for ai in range(A):
            for si in range(S):
                if np.isnan(cube[ai, si]).any():
                    continue
                rr = C.ranks_high_is_1(cube[ai, si])
                fr = min(rr[methods.index(C.INV_DISPLAY[m])] for m in C.FEATURE_METHODS if C.INV_DISPLAY[m] in methods)
                n_r1 += int(fr == 1.0)
                n_tot += 1
        for k, v in ws.items():
            alt.append({"metric": metric, "statistic": k, "value": v})
        alt.append({"metric": metric, "statistic": "feature_rank1_cells", "value": "%d/%d" % (n_r1, n_tot)})
    pd.DataFrame(alt).to_csv(rep / "alt_metric_w.csv", index=False)

    # ---------- 4. paired DeLong ----------
    pairs = pd.DataFrame([p for k in keys for p in cells[k]["_pairs"]])
    pf = []
    if len(pairs):
        for comp, g in pairs.groupby("comparison"):
            pairs.loc[g.index, "p_holm_within_comparison"] = C.holm(g.p_two_sided.to_numpy())
            for _, r in g.iterrows():
                pf.append({"family": "delong_" + comp, "domain": args.domain, "test": "%s seed%d" % (r.arch, r.seed),
                           "p": r.p_two_sided, "source": "persample delong"})
    pairs.to_csv(rep / "delong_pairs.csv", index=False)

    # ---------- 5. accuracy confound for MSP ----------
    acc = []
    for (a, s) in keys:
        c = cells[(a, s)]
        acc.append({"arch": a, "seed": s, "id_acc": float((c["id_logits"].argmax(1) == c["id_labels"]).mean()),
                    "ood_acc": float(correct[(a, s)].mean()), "msp_auroc": c["_pa"].get("msp", np.nan)})
    acc = pd.DataFrame(acc)
    acc.to_csv(rep / "accuracy_by_run.csv", index=False)
    conf = []
    if acc.arch.nunique() >= 3 and set(C.RESNETS) <= set(acc.arch):
        def fit(df, covs):
            X = [np.ones(len(df))]
            names = ["intercept"]
            for a in archs:
                if a == "resnet18" or a not in set(df.arch):
                    continue
                X.append((df.arch == a).to_numpy(float))
                names.append(a)
            for cv in covs:
                X.append(df[cv].to_numpy(float) - df[cv].mean())
                names.append(cv)
            beta = ols(np.stack(X, 1), df.msp_auroc.to_numpy(float))
            bd = dict(zip(names, beta))
            r50 = bd.get("resnet50", 0.0)
            # jump of arch a vs ResNet mean = beta_a - beta_r50/2
            return {a: bd[a] - r50 / 2.0 for a in archs if a in bd and a not in C.RESNETS}, bd

        for covs in ([], ["ood_acc"], ["id_acc"], ["ood_acc", "id_acc"]):
            est, bd = fit(acc, covs)
            boots = {a: [] for a in est}
            for _ in range(2000):
                samp = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31)))
                                  for _, g in acc.groupby("arch")])
                try:
                    e, _ = fit(samp, covs)
                except Exception:
                    continue
                for a in boots:
                    if a in e:
                        boots[a].append(e[a])
            for a, v in est.items():
                lo, hi = C.percentile_ci(np.array(boots[a]))
                conf.append({"covariates": "+".join(covs) or "none", "arch": a, "adj_jump_vs_resnet_mean": v,
                             "run_boot_lo": lo, "run_boot_hi": hi,
                             "covariate_coefs": ";".join("%s=%.4f" % (k, bd[k]) for k in covs)})
        # accuracy-matched pairs (|ood_acc diff| <= 0.02) non-ResNet run vs ResNet run
        res = acc[acc.arch.isin(C.RESNETS)]
        for _, r in acc[~acc.arch.isin(C.RESNETS)].iterrows():
            mt = res[np.abs(res.ood_acc - r.ood_acc) <= 0.02]
            conf.append({"covariates": "matched_ood_acc_0.02", "arch": r.arch, "seed": r.seed, "n_matched_resnet_runs": len(mt),
                         "adj_jump_vs_resnet_mean": float(r.msp_auroc - mt.msp_auroc.mean()) if len(mt) else np.nan})
    pd.DataFrame(conf).to_csv(rep / "accuracy_confound.csv", index=False)

    # ---------- 6. temperature ----------
    trows = []
    try:
        from scipy.optimize import minimize_scalar
    except Exception:
        minimize_scalar = None
    from src.utils.benchmark_metrics import calc_auroc

    def softmax_max(z, T):
        z = z.astype(np.float64) / T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return (p / p.sum(1, keepdims=True)).max(1)

    def energy(z, T):
        z = z.astype(np.float64) / T
        mx = z.max(1, keepdims=True)
        return (T * (mx[:, 0] + np.log(np.exp(z - mx).sum(1))))

    for (a, s) in keys:
        c = cells[(a, s)]
        zi, zo, yi = c["id_logits"], c["ood_logits"], c["id_labels"]

        def nll(T):
            z = zi.astype(np.float64) / T
            z = z - z.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return -lp[np.arange(len(yi)), yi].mean()

        Tstar = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x) if minimize_scalar else float("nan")
        Ts = [0.5, 1.0, 2.0] + ([Tstar] if np.isfinite(Tstar) else [])
        msp_a = [calc_auroc(softmax_max(zi, T), softmax_max(zo, T)) for T in Ts]
        trows.append({"arch": a, "seed": s, "n_classes": zi.shape[1], "T_star": Tstar,
                      "msp_auroc_T1": msp_a[1], "msp_auroc_max_abs_change_over_T": float(np.max(np.abs(np.array(msp_a) - msp_a[1]))),
                      "energy_auroc_T1": calc_auroc(energy(zi, 1.0), energy(zo, 1.0)),
                      "energy_auroc_Tstar": calc_auroc(energy(zi, Tstar), energy(zo, Tstar)) if np.isfinite(Tstar) else np.nan})
    pd.DataFrame(trows).to_csv(rep / "temperature_check.csv", index=False)

    # ---------- 7. coverage table ----------
    crow = []
    for (a, s) in keys:
        for ti, m in enumerate(TRIAGE):
            if (a, s, m) not in csort:
                continue
            pts = cov_multi(csort[(a, s, m)], None, C.RISKS)
            for ri, r in enumerate(C.RISKS):
                rec = {"arch": a, "seed": s, "triage_score": C.DISPLAY[m], "risk": r, "coverage": pts[ri]}
                for md in modes:
                    v = boot_cov[md][:, archs.index(a), seeds.index(s), ti, ri]
                    rec["%s_lo" % md], rec["%s_hi" % md] = C.percentile_ci(v)
                crow.append(rec)
    cov_df = pd.DataFrame(crow)
    cov_df.to_csv(rep / "coverage_cells.csv", index=False)
    gap = []
    if "densenet121" in archs and "mobilenet_v3_large" in archs:
        for si, s in enumerate(seeds):
            if not (present[archs.index("densenet121"), si] and present[archs.index("mobilenet_v3_large"), si]):
                continue
            for ri, r in enumerate(C.RISKS):
                ti = TRIAGE.index("msp")
                p = cov_multi(csort[("mobilenet_v3_large", s, "msp")], None, C.RISKS)[ri] - \
                    cov_multi(csort[("densenet121", s, "msp")], None, C.RISKS)[ri]
                rec = {"seed": s, "risk": r, "mobilenet_minus_densenet_cov": p}
                for md in modes:
                    v = boot_cov[md][:, archs.index("mobilenet_v3_large"), si, ti, ri] - boot_cov[md][:, archs.index("densenet121"), si, ti, ri]
                    rec["%s_lo" % md], rec["%s_hi" % md] = C.percentile_ci(v)
                gap.append(rec)
    pd.DataFrame(gap).to_csv(rep / "coverage_gap_densenet_mobilenet.csv", index=False)
    if len(cov_df):
        pts = cov_df[(cov_df.seed == 42) & (cov_df.triage_score == "MSP") & (np.isclose(cov_df.risk, 0.10))]
        for a, pub in C.PUBLISHED["oracle_full"].items():
            v = pts[pts.arch == a].coverage
            if len(v):
                log.append("REPRO %s full-set cov@10 %s published %.3f recomputed %.4f" % (
                    "PASS" if abs(round(float(v.iloc[0]), 3) - pub) < 1e-9 else "FAIL", a, pub, float(v.iloc[0])))
    pd.DataFrame(pf).to_csv(rep / "pvalues_family.csv", index=False)
    C.write_text(rep / "persample_log.txt", log)
    print("\n".join(log))


if __name__ == "__main__":
    main()
