#!/usr/bin/env python3
"""Readout of the ISBI patch (decisions/precommit_isbi_patch_2026-10-05.md): applies the bars of
sections A-F to the outputs of leakfree_knn.py, leakfree_fit_scores.py,
logit_retrain_slide_disjoint.py, sample_size_disjoint.py and recipe_seeds.py.

Writes outputs/reports/rigor_pack/isbi_patch/: h5a_rerank.csv, phaseA_bar.csv, phaseB_bar.csv,
checklist_h15.md/.csv, scope_table.md, README.md.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

BAR = 0.02
PHASE_B_ARCHS = ("resnet50", "convnext_tiny", "densenet121")
F3 = ("Mahalanobis", "kNN", "ViM")
F2 = ("Mahalanobis", "kNN")
GPU_H_ESTIMATE = "9-12"


def md_table(df: pd.DataFrame, fmt="%.3f") -> list:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt % v if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                                     ("-" if isinstance(v, float) else str(v)) for v in r) + " |")
    return out


def phase_b(rep: Path) -> pd.DataFrame:
    rows = []
    d = rep / "leakage/logit_retrain_slide_disjoint"
    for a in PHASE_B_ARCHS:
        js = [json.loads((d / ("%s_fold%d.json" % (a, f))).read_text()) for f in (0, 1)
              if (d / ("%s_fold%d.json" % (a, f))).exists()]
        pub = C.metric_vector(C.REPO, "camelyon17", a, 42)
        r = {"arch": a, "folds_done": len(js)}
        for s, disp in (("msp", "MSP"), ("energy", "Energy")):
            r["%s_published" % s] = float(pub[C.METHODS_ORDER.index(disp)])
            if len(js) == 2:
                r["%s_slide_disjoint_retrain" % s] = float(np.mean([j["auroc_%s_slide_disjoint" % s] for j in js]))
                r["%s_same_slide_retrain" % s] = float(np.mean([j["auroc_%s_same_slides" % s] for j in js]))
                r["delta_%s" % s] = r["%s_published" % s] - r["%s_slide_disjoint_retrain" % s]
        r["gpu_hours"] = sum(j.get("seconds_total", 0.0) for j in js) / 3600.0
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    rep = C.default_reports(C.REPO)
    out = C.ensure_dir(rep / "isbi_patch")
    lk = pd.read_csv(rep / "leakage/leakfree_knn.csv").set_index("arch")
    vr = pd.read_csv(rep / "leakage/leakfree_vim_react.csv").set_index("arch")
    pb = phase_b(rep)
    b_done = pb[pb.folds_done == 2].set_index("arch") if len(pb) else pd.DataFrame()

    # A
    a_rows = []
    for s in ("vim", "react"):
        loss = -vr["delta_%s_2fold" % s]
        a_rows.append({"score": s, "max_loss": float(loss.max()), "arch_max_loss": loss.idxmax(),
                       "n_archs_loss_gt_0.02": int((loss > BAR).sum()), "n_archs": len(loss),
                       "bar": "appendix + H5(a)" if (loss > BAR).any() else "one sentence (<= 0.02)"})
    pa = pd.DataFrame(a_rows)
    pa.to_csv(out / "phaseA_bar.csv", index=False)

    # C
    idx = {m: i for i, m in enumerate(C.METHODS_ORDER)}
    rr = []
    for a in C.ARCHS:
        v0 = C.metric_vector(C.REPO, "camelyon17", a, 42)
        v2 = v0.copy()
        v2[idx["Mahalanobis"]] = lk.loc[a, "auroc_maha_2fold_slide_disjoint"]
        v2[idx["kNN"]] = lk.loc[a, "auroc_knn_2fold_slide_disjoint"]
        v3 = v2.copy()
        v3[idx["ViM"]] = vr.loc[a, "auroc_vim_2fold_slide_disjoint"]
        v3[idx["ReAct"]] = vr.loc[a, "auroc_react_2fold_slide_disjoint"]
        reads = {"i_original": v0, "ii_leakfree_maha_knn": v2, "iii_leakfree_maha_knn_vim_react": v3}
        if a in b_done.index:
            v4 = v3.copy()
            v4[idx["MSP"]] = b_done.loc[a, "msp_slide_disjoint_retrain"]
            v4[idx["Energy"]] = b_done.loc[a, "energy_slide_disjoint_retrain"]
            reads["iv_plus_logit_retrain"] = v4
        for name, v in reads.items():
            w = C.METHODS_ORDER[int(np.nanargmax(v))]
            rr.append({"arch": a, "reading": name, "winner": w, "winner_auroc": float(np.nanmax(v)),
                       "feature_win_F3": w in F3, "feature_win_F2": w in F2,
                       **{m: float(v[i]) for m, i in idx.items()}})
    rr = pd.DataFrame(rr)
    rr.to_csv(out / "h5a_rerank.csv", index=False)
    cnt = rr.groupby("reading").agg(n=("arch", "size"), F3=("feature_win_F3", "sum"), F2=("feature_win_F2", "sum"))

    # B
    pb_bar = "not run"
    if len(b_done) == 3:
        big = ((b_done.delta_msp.abs() > BAR) | (b_done.delta_energy.abs() > BAR)).sum()
        small = ((b_done.delta_msp.abs() <= BAR) & (b_done.delta_energy.abs() <= BAR)).sum()
        pb_bar = ("not a fair same-protocol comparison without noting training-slide overlap (%d/3 archs > 0.02)" % big
                  if big >= 2 else ("one sentence: MSP/Energy change <= 0.02 (3/3)" if small == 3
                                    else "per-arch deltas reported, no sentence (%d/3 archs > 0.02)" % big))
    elif len(pb):
        pb_bar = "incomplete (%s)" % ", ".join("%s %d/2" % (r.arch, r.folds_done) for r in pb.itertuples())
    pb.to_csv(out / "phaseB_bar.csv", index=False)

    # E
    txt = (rep / "recipe_seeds/recipe_seeds_summary.txt").read_text().splitlines()
    ck = []
    for line in txt:
        m = re.match(r"\s*(M[12])\s+(\S+)\s+seeds (\d+)/5\s+jump>=0.15:\s*(\d+)", line)
        if m:
            n = int(m.group(4))
            ck.append({"set": m.group(1), "arch": m.group(2), "seeds": int(m.group(3)), "seeds_jump": n,
                       "robust": n >= 4, "use": "may be cited as robust" if n >= 4 else "do not cite as robust"})
    ck = pd.DataFrame(ck)
    ck.to_csv(out / "checklist_h15.csv", index=False)
    C.write_text(out / "checklist_h15.md", [
        "# H15 checklist (deployment-facing sentences)", "",
        "Rule (unchanged H15): robust = >= 4/5 seeds with MSP jump >= 0.15 over the same-seed official ResNet mean.", "",
        "## Robust: may be cited", ""] + ["- %s %s: %d/5 seeds" % (r.set, C.SHORT.get(r.arch, r.arch), r.seeds_jump)
                                          for r in ck[ck.robust].itertuples()] +
        ["", "## Not robust: do not cite as robust", ""] +
        ["- %s %s: %d/5 seeds" % (r.set, C.SHORT.get(r.arch, r.arch), r.seeds_jump) for r in ck[~ck.robust].itertuples()])

    # F
    vb = pd.read_csv(rep / "numerical_stability/verdicts_before_after.csv").set_index("H")
    h12 = vb.loc["H12", "as_run"]
    st = pd.DataFrame({"arch": [C.SHORT[a] for a in C.ARCHS],
                       "d_Maha": [lk.loc[a, "delta_maha_2fold"] for a in C.ARCHS],
                       "d_kNN": [lk.loc[a, "delta_knn_2fold"] for a in C.ARCHS],
                       "d_ViM": [vr.loc[a, "delta_vim_2fold"] for a in C.ARCHS],
                       "d_ReAct": [vr.loc[a, "delta_react_2fold"] for a in C.ARCHS],
                       "d_MSP_retrain": [-b_done.loc[a, "delta_msp"] if a in b_done.index else np.nan for a in C.ARCHS],
                       "d_Energy_retrain": [-b_done.loc[a, "delta_energy"] if a in b_done.index else np.nan for a in C.ARCHS]})
    C.write_text(out / "scope_table.md", [
        "# Scope table (Limitations)", "",
        "Camelyon17, seed 42. d_* = slide-disjoint minus same-slide AUROC (2-fold, H11c folds); "
        "d_*_retrain = slide-disjoint retrain minus published (Phase B; '-' = not in the grid).", ""]
        + md_table(st) + ["", "MIDOG (H12, already computed, CSV only): %s" % h12.split(";")[-1].strip(), "",
                          "No other domain or setup is covered by these checks."])

    # D
    dj = pd.concat([pd.read_csv(p) for p in sorted((rep / "sample_size_disjoint").glob("min_size_*.csv"))])
    flips = []
    for p in sorted((rep / "sample_size_disjoint").glob("side_by_side_*.csv")):
        s = pd.read_csv(p)
        f = s[s.overlap_passed_disjoint_failed]
        key = "axis" if "axis" in s else None
        for _, r in f.iterrows():
            flips.append("%s %s %s" % (r.tree, r["axis"] if key else "grid",
                                       r["size"] if key else "%dx%d" % (r.k_seeds, r.m_archs)))

    gpu_actual = float(pb.gpu_hours.sum()) if len(pb) else 0.0
    lines = ["# ISBI patch readout", "",
             "Precommit: `decisions/precommit_isbi_patch_2026-10-05.md` (committed before any number below).", "",
             "## A. ViM / ReAct slide-disjoint 2-fold (bar: any arch loss > 0.02)", ""] + md_table(pa) + [
             "", "Per arch: `leakage/leakfree_vim_react.csv`.", "",
             "## B. Slide-disjoint logit retrain (MSP/Energy)", "", "Bar reading: **%s**" % pb_bar, ""] + (
             md_table(pb) if len(pb) else ["not run"]) + [
             "", "GPU-hours: precommit estimate %s GPU-h, actual %.1f GPU-h (sum of job wall times of the finished folds)."
             % (GPU_H_ESTIMATE, gpu_actual), "",
             "## C. H5(a) re-rank, seed 42 (feature family primary {Maha, kNN, ViM}; secondary {Maha, kNN})", ""] + md_table(
             cnt.reset_index(), fmt="%d") + ["", "Per arch: `h5a_rerank.csv`. Locked wording: \"feature-space wins on "
             "K/8 backbones under slide-disjoint scoring\" with K = %d (reading iii, primary family); never "
             "\"feature always best\"." % int(cnt.loc["iii_leakfree_maha_knn_vim_react", "F3"]), "",
             "## D. Disjoint-partition sample size", ""] + md_table(dj) + [
             "", "Sizes where overlapping subsets passed but disjoint partitions failed (manuscript must prefer the "
             "disjoint numbers there): " + ("; ".join(flips) if flips else "none"), "",
             "## E. H15 checklist", "", "`checklist_h15.md`: robust = %s." % ", ".join(
                 C.SHORT.get(a, a) for a in ck[ck.robust].arch), "",
             "## F. Scope", "", "`scope_table.md` (Camelyon17 seed-42 leak-free deltas + MIDOG H12 one-liner)."]
    C.write_text(out / "README.md", lines)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
