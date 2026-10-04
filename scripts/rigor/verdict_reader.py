#!/usr/bin/env python3
"""Apply the precommitted H1-H17 bars (decisions/decision_precommit_rigor_pack.md) to one
rigor-pack reports folder. Returns {H: (verdict, evidence)}; verdict strings are short labels
taken from the precommit wording. Used for the main reports tree and for every
numerical-stability variant tree, so all readings use the same code.

Camelyon17 is the primary domain; H12 reads skin / MIDOG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DOM = "camelyon17"


def _csv(rep: Path, rel: str):
    p = rep / rel
    return pd.read_csv(p) if p.exists() else None


def h1(rep, dom=DOM):
    df = _csv(rep, "w_csv/matched_w_summary.csv")
    if df is None:
        return "missing", ""
    d = df[df.domain == dom]
    above, within = int(d.above_q95.sum()), int(d.within_q05_q95.sum())
    v = "arch separable" if above >= 6 else ("not separable" if within >= 6 else "mixed")
    return v, "above q95 %d/8, inside [q05,q95] %d/8" % (above, within)


def h2(rep, dom=DOM):
    df = _csv(rep, "multiple_testing/pvalues_adjusted.csv")
    if df is None:
        return "missing", ""
    d = df[(df.family == "W_vs_chance") & (df.domain == dom)]
    n, k = len(d), int(d.sig_holm_family.sum())
    return ("all above chance" if k == n else "%d/%d above chance" % (k, n)), "Holm-sig %d/%d" % (k, n)


def h3(rep, dom=DOM):
    df = _csv(rep, "persample_%s/w_bootstrap.csv" % dom)
    if df is None:
        return "missing", ""
    d = df.set_index("statistic")
    D = d.loc["diff_mean_crossseed_minus_crossarchk5"]
    if D.cluster_lo <= 0 <= D.cluster_hi:
        a = "D not distinguishable from 0"
    elif D.cluster_lo > 0:
        a = "D>0 (cluster CI excludes 0)"
    else:
        a = "D<0 (cluster CI excludes 0)"
    w = d.loc["cross_arch_k8_seed42"]
    half = (w.iid_hi - w.iid_lo) / 2
    b = "sampling not what moves W" if half < 0.05 else "W also test-set-noise limited"
    return "%s; %s" % (a, b), "D=%.3f cluster CI [%.3f, %.3f]; W42 iid half-width %.3f (W42=%.3f)" % (
        D.point, D.cluster_lo, D.cluster_hi, half, w.point)


def h4(rep, dom=DOM):
    perm = _csv(rep, "w_csv/arch_label_perm.csv")
    vd = _csv(rep, "w_csv/variance_decomposition.csv")
    if perm is None or vd is None:
        return "missing", ""
    pa = float(perm[(perm.domain == dom) & (perm.set == "all8")].p_perm.iloc[0])
    row = vd[(vd.domain == dom) & (vd.set == "all8") & (vd.y == "rank")].iloc[0]
    pb = float(row.p_F_perm)
    if pa < 0.05 and pb < 0.05:
        v = "arch effect detected"
    elif pa >= 0.05 and pb >= 0.05:
        v = "not detected"
    else:
        v = "split"
    return v, "all8 p_perm(a)=%.4g, p_F_perm(b, ranks)=%.4g, share_varcomp_arch=%.3f" % (
        pa, pb, float(row.share_varcomp_arch))


def h5(rep, dom=DOM):
    fr = _csv(rep, "w_csv/feature_rank1_cells.csv")
    dl = _csv(rep, "persample_%s/delong_pairs.csv" % dom)
    if fr is None:
        return "missing", ""
    n = int(fr[fr.domain == dom].feature_rank1.sum())
    a = "every backbone and seed" if n == 40 else ("N/40 runs" if n >= 36 else "downgraded to 'usually'")
    ev = "feature rank1 %d/40" % n
    if dl is not None:
        b = dl[dl.comparison == "best_feature_vs_best_other"]
        nb = int(((b.p_holm_within_comparison < 0.05) & (b["diff"] > 0)).sum())
        mk = dl[dl.comparison == "maha_vs_knn"]
        nm = int((mk.p_holm_within_comparison < 0.05).sum())
        a += "; (b) %s" % ("pass" if nb >= 36 else "fail")
        ev += "; DeLong best-feature>best-other Holm-sig %d/%d; Maha vs kNN Holm-sig %d/%d" % (nb, len(b), nm, len(mk))
    return "(a) " + a, ev


def h6_reading(df, score="MSP", risk=0.10, family="F1", mode="per_seed"):
    r = df[(df.family == family) & (df["mode"] == mode) & (df.score == score) & (np.isclose(df.risk, risk))]
    if not len(r):
        return "missing", ""
    r = r.iloc[0]
    med, win, ns = float(r.median_gain), float(r.frac_cov_wins), float(r.n_seeds_median_gain_gt0)
    if med <= 0 or win < 0.5:
        v = "not supported"
    elif win >= 0.6 and ns >= 4:
        v = "supported"
    else:
        v = "helps on average, noisy"
    return v, "median gain %.3f, strict-win %.2f, seeds gain>0 %d/5" % (med, win, ns)


def h6(rep):
    df = _csv(rep, "coverage_splits/splits_summary.csv")
    if df is None:
        return "missing", ""
    return h6_reading(df)


def h7(rep, dom=DOM):
    df = _csv(rep, "persample_%s/alt_metric_w.csv" % dom)
    if df is None:
        return "missing", ""
    d = {}
    for m, s, v in zip(df.metric, df.statistic, df.value):
        v = str(v)
        d[(m, s)] = float(v.split("/")[0]) if s == "feature_rank1_cells" else float(v)

    def mean_cs(m):
        return float(np.nanmean([d.get((m, "cross_seed_k5_%s" % a), np.nan) for a in C.ARCHS]))

    out, ok_all = [], True
    for m in ("fpr95", "aupr_in", "aupr_out"):
        dw42 = abs(d[(m, "cross_arch_k8_seed42")] - d[("auroc", "cross_arch_k8_seed42")])
        dcs = abs(mean_cs(m) - mean_cs("auroc"))
        fr = d.get((m, "feature_rank1_cells"), np.nan)
        ok = dw42 <= 0.10 and dcs <= 0.10 and fr >= 36
        ok_all &= ok
        out.append("%s |dW42|=%.3f |dWcs|=%.3f rank1=%d%s" % (m, dw42, dcs, fr, "" if ok else " X"))
    return ("not AUROC-specific" if ok_all else "changes under some metric"), "; ".join(out)


def h8(rep, dom=DOM):
    t = _csv(rep, "persample_%s/temperature_check.csv" % dom)
    a = _csv(rep, "persample_%s/accuracy_confound.csv" % dom)
    if t is None or a is None:
        return "missing", ""
    mx = float(t.msp_auroc_max_abs_change_over_T.max())
    va = "T-invariance check %s (max |dAUROC| %.2g)" % ("pass" if mx < 1e-6 else "FAIL", mx)
    r = a[(a.covariates == "ood_acc+id_acc") & (a.arch == "densenet121")]
    if not len(r):
        return va, ""
    r = r.iloc[0]
    if r.run_boot_lo <= 0 <= r.run_boot_hi:
        vb = "explained"
    elif r.adj_jump_vs_resnet_mean >= 0.15:
        vb = "jump not explained by accuracy"
    else:
        vb = "partly explained"
    return "%s; (b) %s" % (va, vb), "DenseNet adj jump %.3f CI [%.3f, %.3f]" % (
        r.adj_jump_vs_resnet_mean, r.run_boot_lo, r.run_boot_hi)


def h9(rep):
    df = _csv(rep, "coverage_splits/splits_summary.csv")
    if df is None:
        return "missing", ""
    msp = [h6_reading(df, "MSP", r)[0] for r in C.RISKS]
    sc = {s: h6_reading(df, s, 0.10)[0] for s in ("MSP", "Energy", "Mahalanobis", "kNN")}
    rr = "risk-level robust" if len(set(msp)) == 1 else "risk-level dependent"
    ts = "triage-score robust" if len(set(sc.values())) == 1 else "triage-score dependent"
    return "%s; %s" % (rr, ts), "MSP@5/10/20: %s | @10: %s" % (
        "/".join(msp), ", ".join("%s=%s" % kv for kv in sc.items()))


def h10(rep, dom=DOM):
    g = _csv(rep, "persample_%s/coverage_gap_densenet_mobilenet.csv" % dom)
    if g is None:
        return "missing", ""
    r = g[(g.seed == 42) & np.isclose(g.risk, 0.10)].iloc[0]
    v = "gap kept (cluster CI excludes 0)" if r.cluster_lo > 0 or r.cluster_hi < 0 else "gap not resolved at patient level"
    return v + "; per-seed MSP-jump cluster CI not produced by persample.py", \
        "gap %.3f cluster CI [%.3f, %.3f]" % (r.mobilenet_minus_densenet_cov, r.cluster_lo, r.cluster_hi)


def h11(rep):
    lk = _csv(rep, "hospital2/leakage_checks.csv")
    if lk is None:
        return "missing", ""
    l1 = int(lk[lk.check.str.startswith("L1")].n_overlap.iloc[0])
    l2 = int(lk[lk.check.str.startswith("L2")].n_overlap.iloc[0])
    v = "(a,b) pass" if l1 == 0 and l2 == 0 else "(a,b) FAIL - stop"
    lf = None
    for p in sorted(rep.glob("leakfree*/*.csv")) + sorted(rep.glob("leakage*/*.csv")):
        lf = p
    return v + ("; (c) see %s" % lf.relative_to(rep) if lf else "; (c) pending (leak job)"), \
        "patients overlap %d, slides overlap %d" % (l1, l2)


def h12(rep):
    out = []
    for dom in ("skin_isic_pad", "midog"):
        out.append("%s: H1 %s, H4 %s" % (dom, h1(rep, dom)[0], h4(rep, dom)[0]))
    return "; ".join(out), "; ".join("%s %s | %s" % (d, h1(rep, d)[1], h4(rep, d)[1]) for d in ("skin_isic_pad", "midog"))


def h13(rep):
    p = rep / "vit_seeds/vit_seeds_summary.txt"
    if not p.exists():
        return "missing", ""
    txt = p.read_text().splitlines()
    v = next((l.split(":", 1)[1].strip() for l in txt if l.startswith("verdict")), "?")
    return v, next((l for l in txt if l.startswith("seeds complete")), "")


def h14(rep):
    df = _csv(rep, "multiple_testing/pvalues_adjusted.csv")
    if df is None:
        return "missing", ""
    d = df[df.domain == DOM].groupby("family").sig_holm_family.agg(["sum", "count"])
    return "descriptive (Holm within family)", "; ".join("%s %d/%d" % (f, r["sum"], r["count"]) for f, r in d.iterrows())


def h15(rep):
    p = rep / "recipe_seeds/recipe_seeds_summary.txt"
    if not p.exists():
        return "missing", ""
    txt = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    rows = [l for l in txt if l.startswith(("M1 ", "M2 "))]
    if any("INCOMPLETE" in l for l in rows):
        return "pending (recipe seeds incomplete)", " | ".join(rows)
    return "; ".join(l.split("->", 1)[-1].strip() for l in rows), " | ".join(rows)


def h16(rep, dom=DOM):
    s = _csv(rep, "transfer_regret/regret_summary.csv")
    pr = _csv(rep, "transfer_regret/regret_pairs.csv")
    if s is None:
        return "missing", ""
    d = s[s.domain == dom]
    nf = d[(d.family == "nonfeature") & (d.kind == "cross_arch")].iloc[0]
    pct = nf.get("pct_rank_of_0.185_seed42", np.nan)
    a = "0.185 = worst case wording" if pct >= 0.95 else "0.185 kept with percentile"
    ev = "pct of 0.185 among seed-42 nonfeature cross-arch %.3f" % pct
    if pr is not None and "src" in pr.columns:
        q = pr[(pr.domain == dom) & (pr.family == "nonfeature") & (pr.kind == "cross_arch")
               & (pr.src == "resnet50") & (pr.tgt == "convnext_tiny") & (pr.seed == 42)]
        if len(q):
            ev += "; R50->ConvNeXt s42 nonfeature regret %.4f" % float(q.regret.iloc[0])
    fe = d[(d.family == "feature") & (d.kind == "cross_arch")].iloc[0]
    b = "feature regret <=0.05 for all seeds" if fe["max"] <= 0.05 else "feature regret >0.05 in %.1f%% pairs" % (100 * fe.frac_gt_005 if "frac_gt_005" in fe else 100 * fe["frac_gt_0.05"])
    ev += "; feature cross-arch max %.3f" % fe["max"]
    cs = d[(d.family == "nonfeature") & (d.kind == "cross_seed")]
    if len(cs):
        ev += "; nonfeature median cross-seed %.3f vs cross-arch %.3f" % (cs.iloc[0]["median"], nf["median"])
    return "%s; %s" % (a, b), ev


def h17(rep, dom=DOM):
    df = _csv(rep, "w_ties/w_ties.csv")
    if df is None:
        return "missing", ""
    d = df[(df.domain == dom) & np.isclose(df.eps, 0.01)]
    cs = d[(d.kind == "cross_seed") & d.label.isin(C.ARCHS)].W.mean()
    ca = d[(d.kind == "cross_arch") & d.label.isin(["seed%d" % s for s in C.SEEDS])].W.mean()
    D = cs - ca
    v = "tie-robust (|D|<=0.10)" if abs(D) <= 0.10 else ("seeds agree more (D>0.10)" if D > 0 else "architectures agree more (D<-0.10)")
    return v, "eps=0.01 D=%.3f (cross-seed %.3f, cross-arch %.3f)" % (D, cs, ca)


ALL = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6, "H7": h7, "H8": h8, "H9": h9,
       "H10": h10, "H11": h11, "H12": h12, "H13": h13, "H14": h14, "H15": h15, "H16": h16, "H17": h17}
# hypotheses whose inputs include anchor Mahalanobis / ViM (addendum 2026-10-05)
DEPENDS_MAHA_VIM = ("H1", "H2", "H3", "H4", "H5", "H7", "H9", "H14", "H16", "H17")


def read_all(rep: Path, only=None) -> dict:
    out = {}
    for h, f in ALL.items():
        if only and h not in only:
            continue
        try:
            out[h] = f(rep)
        except Exception as e:  # report, never hide
            out[h] = ("error", "%s: %s" % (type(e).__name__, e))
    return out


if __name__ == "__main__":
    rep = Path(sys.argv[1]) if len(sys.argv) > 1 else C.default_reports(C.REPO)
    for h, (v, e) in read_all(rep).items():
        print("%-4s %-55s %s" % (h, v, e))
