#!/usr/bin/env python3
"""Final tables (CSV + LaTeX snippets) and the new Fig. 1b from rigor-pack outputs.

Reads only files written by the rigor pack plus the tracked frozen-split result
outputs/reports/camelyon_coverage_val_split.csv (for the 0.539 / 0.754 / 1.000 panel).
Never writes into manuscript/. Missing inputs -> that table is skipped with a note.
Output: outputs/reports/rigor_pack/tables/*.{csv,tex}, outputs/reports/rigor_pack/fig/fig1b_valsplit.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def tex_table(df: pd.DataFrame, caption: str, label: str, colfmt: str = None) -> str:
    cols = list(df.columns)
    colfmt = colfmt or ("l" + "c" * (len(cols) - 1))

    def esc(x):
        if isinstance(x, float):
            return "---" if np.isnan(x) else "%.3f" % x
        return str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    lines = [r"\begin{table}[t]", r"\floatconts", "  {%s}" % label, "  {\\caption{%s}}" % caption,
             "  {\\small", "  \\begin{tabular}{@{}%s@{}}" % colfmt, "  \\toprule",
             "  " + " & ".join(esc(c) for c in cols) + r" \\", "  \\midrule"]
    for _, r in df.iterrows():
        lines.append("  " + " & ".join(esc(r[c]) for c in cols) + r" \\")
    lines += ["  \\bottomrule", "  \\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def read(p: Path):
    if p.exists():
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--reports", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    rep = Path(args.reports) if args.reports else C.default_reports(root)
    tab = C.ensure_dir(rep / "tables")
    figd = C.ensure_dir(rep / "fig")
    notes = []

    # ---- T1: rater-matched W, arch-vs-seed tests (Camelyon; skin/MIDOG rows too) ----
    ms = read(rep / "w_csv/matched_w_summary.csv")
    alp = read(rep / "w_csv/arch_label_perm.csv")
    vd = read(rep / "w_csv/variance_decomposition.csv")
    if ms is not None and len(ms):
        t1 = ms[["domain", "arch", "w_cross_seed", "crossarch_k5_pooled_median", "crossarch_k5_pooled_q05",
                 "crossarch_k5_pooled_q95", "pct_in_crossarch_k5_pooled"]].copy()
        t1["arch"] = t1.arch.map(lambda a: C.SHORT.get(a, a))
        t1.to_csv(tab / "T1_rater_matched_w.csv", index=False)
        cam = t1[t1.domain == "camelyon17"].drop(columns="domain")
        cam.columns = ["Backbone", "W seed (k=5)", "W arch k=5 median", "q05", "q95", "percentile"]
        (tab / "T1_rater_matched_w.tex").write_text(tex_table(
            cam, "Cross-seed $W$ (5 seeds) vs.\\ cross-architecture $W$ over all 56 five-backbone subsets (seeds 42--46 pooled), Camelyon hospital~2.",
            "tab:ratermatched"))
    else:
        notes.append("T1 skipped (run w_from_csv.py)")
    if alp is not None and vd is not None and len(alp):
        v = vd[["domain", "set", "y", "F_methodxarch_vs_methodxseed", "p_F", "p_F_perm", "share_varcomp_arch",
                "share_ss_interaction_arch"]]
        t2 = alp[["domain", "set", "observed", "null_mean", "excess", "p_perm"]].merge(
            v[v.y == "rank"].drop(columns="y"), on=["domain", "set"], how="left")
        t2.to_csv(tab / "T2_arch_vs_seed_tests.csv", index=False)
        t2t = t2[["domain", "set", "observed", "null_mean", "p_perm", "F_methodxarch_vs_methodxseed", "p_F_perm",
                  "share_varcomp_arch"]].copy()
        t2t.columns = ["domain", "archs", "mean W seed", "null", "p (label perm)", "F", "p (F perm)", "arch share"]
        (tab / "T2_arch_vs_seed_tests.tex").write_text(tex_table(
            t2t, "Does architecture explain ranking variance beyond seed? Arch-label permutation of the 40 runs and nested ANOVA on within-run ranks.",
            "tab:archseed"))
    # ---- T3: bootstrap W CIs ----
    wb = read(rep / "persample_camelyon17/w_bootstrap.csv")
    if wb is not None and len(wb):
        wb.to_csv(tab / "T3_w_bootstrap.csv", index=False)
        keep = wb[wb.statistic.str.match(r"(cross_arch_k8_seed42|cross_seed_k5_|crossarch_k5_mean_allseeds|diff_mean)")]
        cols = [c for c in ["statistic", "point", "iid_lo", "iid_hi", "cluster_lo", "cluster_hi"] if c in keep.columns]
        (tab / "T3_w_bootstrap.tex").write_text(tex_table(keep[cols], "Kendall's $W$ with bootstrap over test samples (patch-iid and patient-cluster).", "tab:wboot"))
    else:
        notes.append("T3 skipped (run persample.py)")
    # ---- T4: feature rank-1 and alternative metrics ----
    fr = read(rep / "w_csv/feature_rank1_cells.csv")
    alt = read(rep / "persample_camelyon17/alt_metric_w.csv")
    if fr is not None and len(fr):
        t4 = fr.groupby("domain").agg(cells=("feature_rank1", "size"), feature_rank1=("feature_rank1", "sum"),
                                      both_top2=("both_features_top2", "sum")).reset_index()
        t4.to_csv(tab / "T4_feature_rank1.csv", index=False)
        (tab / "T4_feature_rank1.tex").write_text(tex_table(t4, "Cells (arch $\\times$ seed) where a feature-space score is rank~1 (AUROC).", "tab:rank1"))
    if alt is not None and len(alt):
        alt.to_csv(tab / "T4b_alt_metric_w.csv", index=False)
    # ---- T5: coverage splits ----
    ss = read(rep / "coverage_splits/splits_summary.csv")
    if ss is not None and len(ss):
        prim = ss[(ss.score == "MSP") & np.isclose(ss.risk, 0.10)]
        cols = ["family", "mode", "n", "median_test_auroc_pick", "median_test_cov_pick", "median_test_oracle",
                "median_test_random", "median_gain", "q25_gain", "q75_gain", "frac_cov_wins", "frac_auroc_wins"]
        prim[cols].to_csv(tab / "T5_coverage_splits.csv", index=False)
        t5 = prim[prim["mode"].isin(["per_seed", "seed_avg"]) & prim.family.isin(["F1", "F2", "F3", "LOPO", "FROZEN"])][cols]
        t5 = t5.rename(columns={"median_test_auroc_pick": "AUROC-pick", "median_test_cov_pick": "cov-pick",
                                "median_test_oracle": "oracle", "median_test_random": "random", "frac_cov_wins": "cov wins",
                                "frac_auroc_wins": "AUROC wins"})
        (tab / "T5_coverage_splits.tex").write_text(tex_table(
            t5, "Held-out coverage@risk\\,10\\% (MSP) of the backbone chosen on the validation patients, over repeated patient splits of hospital~2 (medians).",
            "tab:covsplits"))
        ss.to_csv(tab / "T5b_coverage_splits_all.csv", index=False)
    else:
        notes.append("T5 skipped (run coverage_splits.py)")
    cc = read(rep / "persample_camelyon17/coverage_cells.csv")
    if cc is not None and len(cc):
        cc.to_csv(tab / "T6_coverage_cells_ci.csv", index=False)

    # ---- Fig 1b ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fz = read(root / "outputs/reports/camelyon_coverage_val_split.csv")
    L = None
    lp = rep / "coverage_splits/splits_long.csv.gz"
    if lp.exists():
        L = pd.read_csv(lp)
        L = L[(L.family == "F1") & (L["mode"] == "per_seed") & (L.score == "MSP") & np.isclose(L.risk, 0.10)]
    ncol = 2 if L is not None and len(L) else 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 2.6), squeeze=False)
    ax = axes[0, 0]
    if fz is not None:
        p1 = fz.loc[fz.auroc_val.idxmax()]
        p2 = fz.loc[fz.cov10_val.idxmax()]
        p3 = fz.loc[fz.cov10_test.idxmax()]
        vals = [p1.cov10_test, p2.cov10_test, p3.cov10_test]
        labs = ["AUROC-pick\n(%s)" % C.SHORT.get(p1.backbone.replace("_224", ""), p1.backbone),
                "cov-pick\n(%s)" % C.SHORT.get(p2.backbone.replace("_224", ""), p2.backbone),
                "best\n(%s)" % C.SHORT.get(p3.backbone.replace("_224", ""), p3.backbone)]
        bars = ax.bar(range(3), vals, color=["#bbbbbb", "#3b7dd8", "#2a9d5c"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, "%.3f" % v, ha="center", fontsize=7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labs, fontsize=7)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("test-half coverage@risk 10%", fontsize=7)
        ax.set_title("frozen patient split (seed 42)", fontsize=8)
    else:
        notes.append("Fig1b left panel: camelyon_coverage_val_split.csv missing")
    if ncol == 2:
        ax = axes[0, 1]
        data = [L.test_auroc_pick.to_numpy(), L.test_cov_pick.to_numpy(), L.test_oracle.to_numpy()]
        ax.boxplot(data, widths=0.5, showfliers=False)
        rng = np.random.default_rng(0)
        for i, d in enumerate(data):
            ax.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(d)), d, s=2, alpha=0.25, color="k")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["AUROC-pick", "cov-pick", "oracle"], fontsize=7)
        ax.set_ylim(0, 1.05)
        wins = float(np.mean(L.gain_cov_minus_auroc > 1e-12))
        ax.set_title("%d splits x %d seeds; cov-pick wins %.0f%%" % (L.split_id.nunique(), L.seed.nunique(), 100 * wins), fontsize=8)
    fig.tight_layout()
    fig.savefig(figd / "fig1b_valsplit.pdf")
    fig.savefig(figd / "fig1b_valsplit.png", dpi=200)
    C.write_text(tab / "make_tables_log.txt", ["make_tables notes:"] + (notes or ["all inputs present"]))
    print("tables ->", tab, "\nfigure ->", figd / "fig1b_valsplit.pdf")
    for n in notes:
        print("NOTE", n)


if __name__ == "__main__":
    main()
