#!/usr/bin/env python3
"""Coverage@risk backbone selection under repeated patient-level splits of hospital 2 (CPU).

Precommit: decisions/decision_precommit_rigor_pack.md (H6, H9b, H6-LOPO).

For each split (val patients V, test patients T = rest) and each model seed:
  AUROC-select   : arch with max val AUROC (ID = WILDS id_val, OOD = hospital-2 patches of V)
  Coverage-select: arch with max val coverage@risk
  Oracle         : arch with max test coverage@risk (upper bound, not a procedure)
  Random         : mean test coverage over the 8 archs (what a coin flip gets on average)
and report the chosen arch's TEST coverage@risk.

Split families (patients of hospital 2; 9 patients):
  F1 primary  : every val set of 4 or 5 patients (C(9,4)+C(9,5) = 252 ordered splits)
  F2          : every val set of 3 or 6 patients (168; the frozen split is one of these)
  F3          : every non-empty proper val set (510)
  LOPO        : test = one patient, val = other 8 (9 splits)
  DROP1       : frozen split with one patient removed from hospital 2 (9 splits)
  FROZEN      : the frozen split (reproduction check: 0.539 / 0.754 / 1.000 at seed 42, MSP, 10 %)
Seed modes:
  per_seed    : select and evaluate within each seed (distribution over splits x seeds)
  seed_avg    : select on the seed-averaged val metric, report seed-averaged test coverage
  cross_seed  : select with seed s, evaluate the chosen arch's seed s' != s (retraining risk)
Triage score (ranking for coverage and for AUROC): MSP (primary), Energy, Mahalanobis, kNN.
Risk targets: 5 %, 10 % (primary), 20 %.
Also: Kendall tau between val-AUROC ranking and test-coverage ranking, and between
val-coverage and test-coverage ranking, per split.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SCORES = ("msp", "energy", "mahalanobis", "knn")


def kendall_tau(x, y):
    try:
        from scipy.stats import kendalltau

        return float(kendalltau(x, y)[0])
    except Exception:
        x, y = np.asarray(x), np.asarray(y)
        n = len(x)
        s = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
        return 2 * s / (n * (n - 1))


def build_splits(patients, frozen):
    P = list(patients)
    out = []

    def add(fam, val):
        val = tuple(sorted(val))
        test = tuple(p for p in P if p not in val)
        out.append({"family": fam, "val": val, "test": test, "drop": ""})

    for k in (4, 5):
        for v in itertools.combinations(P, k):
            add("F1", v)
    for k in (3, 6):
        for v in itertools.combinations(P, k):
            add("F2", v)
    for k in range(1, len(P)):
        for v in itertools.combinations(P, k):
            add("F3", v)
    for p in P:
        add("LOPO", [q for q in P if q != p])
    if frozen:
        add("FROZEN", frozen)
        for p in P:
            v = tuple(q for q in frozen if q != p)
            t = tuple(q for q in P if q not in frozen and q != p)
            if v and t:
                out.append({"family": "DROP1", "val": v, "test": t, "drop": p})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--feat-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reports", default=None)
    ap.add_argument("--families", nargs="+", default=["F1", "F2", "F3", "LOPO", "DROP1", "FROZEN"])
    ap.add_argument("--scores", nargs="+", default=list(SCORES))
    args = ap.parse_args()
    root = Path(args.root)
    feat_root = Path(args.feat_root) if args.feat_root else root
    out_root = Path(args.out) if args.out else C.default_out(root)
    rep = Path(args.reports) if args.reports else C.default_reports(root) / "coverage_splits"
    C.ensure_dir(rep)
    archs, seeds = list(C.ARCHS), list(C.SEEDS)
    log = ["coverage_splits %s" % C.env_info()]

    meta = C.load_camelyon_metadata(feat_root)
    cells = {}
    for a in archs:
        for s in seeds:
            p = C.score_cache_path(out_root, "camelyon17", a, s)
            if p.exists():
                z = np.load(p, allow_pickle=False)
                cells[(a, s)] = {k: z[k] for k in z.files if k != "source"}
    seeds = [s for s in seeds if all((a, s) in cells for a in archs)]
    log.append("complete seeds (all 8 archs cached): %s" % seeds)
    if not seeds:
        C.write_text(rep / "coverage_splits_log.txt", log + ["no complete seed; run build_score_cache.py"])
        print("\n".join(log))
        sys.exit(3)
    ood_y = cells[(archs[0], seeds[0])]["ood_labels"]
    if meta is None:
        log.append("metadata.csv missing -> cannot map patches to patients")
        C.write_text(rep / "coverage_splits_log.txt", log)
        print("\n".join(log))
        sys.exit(3)
    h2 = meta[meta.wilds_split == 2]
    if len(h2) != len(ood_y) or not np.array_equal(h2.tumor.to_numpy(), ood_y):
        raise RuntimeError("hospital-2 metadata order does not match OOD labels")
    patient = h2.patient.to_numpy()
    patients = sorted(np.unique(patient))
    pmask = {p: patient == p for p in patients}
    frozen = C.frozen_val_patients(root)
    if frozen:
        frozen = [p for p in frozen if p in pmask]
    splits = [s for s in build_splits(patients, frozen) if s["family"] in args.families]
    log.append("patients: %s; splits: %s" % (patients, pd.Series([s["family"] for s in splits]).value_counts().to_dict()))

    # sorters
    t0 = time.time()
    asort, csort = {}, {}
    for (a, s), c in cells.items():
        corr = (c["ood_logits"].argmax(1) == c["ood_labels"]).astype(np.float64)
        for m in args.scores:
            if ("ood_" + m) not in c:
                continue
            asort[(a, s, m)] = C.AurocSorter(c["id_" + m], c["ood_" + m])
            csort[(a, s, m)] = C.CoverageSorter(c["ood_" + m], corr)

    def cov3(srt, w):
        ww = w[srt.order]
        cw = np.cumsum(ww)
        tot = cw[-1]
        cc = np.cumsum(ww * srt.c)
        with np.errstate(divide="ignore", invalid="ignore"):
            risk = 1.0 - cc / cw
        res = []
        for t in C.RISKS:
            ok = (cw > 0) & (risk <= t + 1e-12) & (ww > 0)
            res.append(float(cw[np.nonzero(ok)[0][-1]] / tot) if np.any(ok) else 0.0)
        return res

    long_rows, tau_rows, meta_rows = [], [], []
    A = len(archs)
    for sid, sp in enumerate(splits):
        wv = np.zeros(len(patient))
        wt = np.zeros(len(patient))
        for p in sp["val"]:
            wv[pmask[p]] = 1.0
        for p in sp["test"]:
            wt[pmask[p]] = 1.0
        meta_rows.append({"split_id": sid, "family": sp["family"], "val": " ".join(sp["val"]), "test": " ".join(sp["test"]),
                          "drop": sp["drop"], "n_val": int(wv.sum()), "n_test": int(wt.sum()),
                          "tumor_frac_val": float(ood_y[wv > 0].mean()), "tumor_frac_test": float(ood_y[wt > 0].mean()),
                          "share_patches_val": float(wv.sum() / (wv.sum() + wt.sum()))})
        for m in args.scores:
            if not all((a, s, m) in csort for a in archs for s in seeds):
                continue
            va = np.zeros((A, len(seeds)))
            vc = np.zeros((A, len(seeds), len(C.RISKS)))
            tc = np.zeros((A, len(seeds), len(C.RISKS)))
            for ai, a in enumerate(archs):
                for si, s in enumerate(seeds):
                    va[ai, si] = asort[(a, s, m)].auroc(None, wv)
                    vc[ai, si] = cov3(csort[(a, s, m)], wv)
                    tc[ai, si] = cov3(csort[(a, s, m)], wt)
            for ri, r in enumerate(C.RISKS):
                base = {"split_id": sid, "family": sp["family"], "score": C.DISPLAY[m], "risk": r}
                # per seed
                for si, s in enumerate(seeds):
                    ia, ic, io = int(np.argmax(va[:, si])), int(np.argmax(vc[:, si, ri])), int(np.argmax(tc[:, si, ri]))
                    long_rows.append(dict(base, mode="per_seed", seed=s, auroc_pick=archs[ia], cov_pick=archs[ic],
                                          oracle_pick=archs[io], test_auroc_pick=tc[ia, si, ri],
                                          test_cov_pick=tc[ic, si, ri], test_oracle=tc[io, si, ri],
                                          test_random=float(tc[:, si, ri].mean())))
                    if ri == C.RISKS.index(0.10):
                        tau_rows.append({"split_id": sid, "family": sp["family"], "score": C.DISPLAY[m], "seed": s,
                                         "tau_valAUROC_testCov": kendall_tau(va[:, si], tc[:, si, ri]),
                                         "tau_valCov_testCov": kendall_tau(vc[:, si, ri], tc[:, si, ri]),
                                         "tau_valAUROC_valCov": kendall_tau(va[:, si], vc[:, si, ri])})
                # seed-averaged
                mva, mvc, mtc = va.mean(1), vc[:, :, ri].mean(1), tc[:, :, ri].mean(1)
                ia, ic, io = int(np.argmax(mva)), int(np.argmax(mvc)), int(np.argmax(mtc))
                long_rows.append(dict(base, mode="seed_avg", seed=-1, auroc_pick=archs[ia], cov_pick=archs[ic],
                                      oracle_pick=archs[io], test_auroc_pick=mtc[ia], test_cov_pick=mtc[ic],
                                      test_oracle=mtc[io], test_random=float(mtc.mean())))
                # cross-seed transfer: select on seed s, evaluate on each other seed s2
                if len(seeds) > 1:
                    vals = []
                    for si in range(len(seeds)):
                        ia, ic = int(np.argmax(va[:, si])), int(np.argmax(vc[:, si, ri]))
                        for s2 in range(len(seeds)):
                            if s2 == si:
                                continue
                            vals.append((tc[ia, s2, ri], tc[ic, s2, ri], tc[:, s2, ri].max(), tc[:, s2, ri].mean()))
                    v = np.array(vals)
                    long_rows.append(dict(base, mode="cross_seed", seed=-2, auroc_pick="", cov_pick="", oracle_pick="",
                                          test_auroc_pick=float(v[:, 0].mean()), test_cov_pick=float(v[:, 1].mean()),
                                          test_oracle=float(v[:, 2].mean()), test_random=float(v[:, 3].mean())))
        if sid % 100 == 0:
            print("split %d/%d (%.0fs)" % (sid, len(splits), time.time() - t0), flush=True)

    L = pd.DataFrame(long_rows)
    L["gain_cov_minus_auroc"] = L.test_cov_pick - L.test_auroc_pick
    L.to_csv(rep / "splits_long.csv.gz", index=False)
    pd.DataFrame(meta_rows).to_csv(rep / "splits_meta.csv", index=False)
    T = pd.DataFrame(tau_rows)
    T.to_csv(rep / "splits_tau.csv.gz", index=False)

    summ = []
    for (fam, mode, sc, r), g in L.groupby(["family", "mode", "score", "risk"]):
        d = g.gain_cov_minus_auroc.to_numpy()
        rec = {"family": fam, "mode": mode, "score": sc, "risk": r, "n": len(g),
               "median_gain": float(np.median(d)), "q25_gain": float(np.quantile(d, 0.25)), "q75_gain": float(np.quantile(d, 0.75)),
               "mean_gain": float(d.mean()), "frac_cov_wins": float(np.mean(d > 1e-12)),
               "frac_ties": float(np.mean(np.abs(d) <= 1e-12)), "frac_auroc_wins": float(np.mean(d < -1e-12))}
        for col in ("test_auroc_pick", "test_cov_pick", "test_oracle", "test_random"):
            rec["median_" + col] = float(g[col].median())
            rec["q25_" + col] = float(g[col].quantile(0.25))
            rec["q75_" + col] = float(g[col].quantile(0.75))
        if mode == "per_seed":
            per = g.groupby("seed").gain_cov_minus_auroc.median()
            rec["n_seeds_median_gain_gt0"] = int((per > 0).sum())
            rec["n_seeds"] = int(len(per))
        summ.append(rec)
    S = pd.DataFrame(summ)
    S.to_csv(rep / "splits_summary.csv", index=False)
    if len(T):
        T.groupby(["family", "score"])[["tau_valAUROC_testCov", "tau_valCov_testCov", "tau_valAUROC_valCov"]] \
            .median().reset_index().to_csv(rep / "splits_tau_summary.csv", index=False)

    fz = L[(L.family == "FROZEN") & (L["mode"] == "per_seed") & (L.seed == 42) & (L.score == "MSP") & np.isclose(L.risk, 0.10)]
    if len(fz):
        r = fz.iloc[0]
        pub = C.PUBLISHED["frozen_split"]
        for name, col, pick in (("auroc_pick", "test_auroc_pick", "auroc_pick"), ("cov_pick", "test_cov_pick", "cov_pick"),
                                ("best", "test_oracle", "oracle_pick")):
            ok = r[pick] == pub[name][0] and abs(round(float(r[col]), 3) - pub[name][1]) < 1e-9
            log.append("REPRO %s frozen split %s: %s %.4f (published %s %.3f)" % ("PASS" if ok else "FAIL", name, r[pick],
                                                                                 r[col], pub[name][0], pub[name][1]))
    log.append("wrote splits_long.csv.gz, splits_summary.csv, splits_meta.csv, splits_tau*.csv")
    C.write_text(rep / "coverage_splits_log.txt", log)
    print("\n".join(log))


if __name__ == "__main__":
    main()
