#!/usr/bin/env python3
"""Readout for numerical_stability.py (addendum 2026-10-05). Stages are called as
`numerical_stability.py {aggregate|variants|verdicts}`.

aggregate -> outputs/reports/rigor_pack/numerical_stability/
  old_runs.csv               one row per (arch, seed, repeat, method): old-code AUROC/FPR95/AUPR, neg d^2
  old_spread.csv             per anchor cell x method: published, mean, std, min, max, n negative d^2
  stable_vs_published.csv    per arch x seed x method: published, stable (mean, max-min over repeats), ViM dim
  w_per_variant.csv          Kendall W numbers (official 0.694, cross-seed, 4-anchor, k=8 per seed) per variant
  w_spread.csv               min / max / std of each W over the 10 old repeats, + stable
variants -> outputs/rigor_pack/stability/variants/{old_r0..old_r9,stable}/ (input trees, see numerical_stability.py)
verdicts -> outputs/reports/rigor_pack/numerical_stability/verdicts_by_variant.csv, verdicts_before_after.csv/.md
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

METHODS = ("mahalanobis", "vim")
N_OLD = 10
# Extra stable readings (added 2026-10-04, after the old/stable spread was seen): stable
# Mahalanobis with the published ViM path at 8 BLAS threads (repeat 2 = 8 threads), and stable
# Mahalanobis with ViM dropped (stages run with DST_EXCLUDE_METHODS=vim).
EXTRA = ["stable_maha_vim8", "stable_maha_novim"]
VARIANTS = ["old_r%d" % r for r in range(N_OLD)] + ["stable"] + EXTRA
NOVIM = "stable_maha_novim"


def spec(v: str) -> dict:
    """method -> (repeat, 'old'|'stable'); which archs are swapped is in archs_of(v)."""
    if v.startswith("old_r"):
        r = int(v[len("old_r"):])
        return {m: (r, "old") for m in METHODS}
    if v == "stable":
        return {m: (0, "stable") for m in METHODS}
    return {"mahalanobis": (0, "stable"), "vim": (2, "old")}


def archs_of(v: str):
    return C.ANCHORS if v.startswith("old_r") else C.ARCHS


def rep_dir(root: Path) -> Path:
    return C.ensure_dir(C.default_reports(root) / "numerical_stability")


def load_runs(out_root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((out_root / "stability" / "raw").glob("*.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        raise SystemExit("no stability results under %s" % (out_root / "stability" / "raw"))
    return pd.DataFrame(rows)


def published(root: Path, arch: str, seed: int, method: str, metric: str = "AUROC") -> float:
    v = C.metric_vector(root, "camelyon17", arch, seed, metric)
    return float(v[C.METHODS_ORDER.index(C.DISPLAY[method])])


def value_table(root: Path, runs: pd.DataFrame) -> dict:
    """{variant: {(arch, seed, method): (auroc, fpr95, aupr_in, aupr_out)}} for swapped cells only."""
    out = {v: {} for v in VARIANTS}
    by = {(r.arch, int(r.seed), int(r["repeat"])): r for _, r in runs.iterrows()}
    for v in VARIANTS:
        for a in archs_of(v):
            for s in C.SEEDS:
                for m, (rep, pre) in spec(v).items():
                    r = by.get((a, s, rep))
                    if r is not None:
                        out[v][(a, s, m)] = tuple(float(r["%s_%s_%s" % (pre, m, k)])
                                                  for k in ("auroc", "fpr95", "aupr_in", "aupr_out"))
    return out


def cube(root: Path, swap: dict) -> np.ndarray:
    """(8 archs, 5 seeds, 7 methods) AUROC with swapped Mahalanobis/ViM."""
    X = C.load_cube(root, "camelyon17", C.ARCHS, C.SEEDS)
    for (a, s, m), vals in swap.items():
        X[C.ARCHS.index(a), C.SEEDS.index(s), C.METHODS_ORDER.index(C.DISPLAY[m])] = vals[0]
    return X


def w_numbers(X: np.ndarray) -> dict:
    W = lambda M: float(C.kendall_w(C.ranks_rows(M)))
    out = {}
    for j, s in enumerate(C.SEEDS):
        out["cross_arch_k8_seed%d" % s] = W(X[:, j, :])
        out["anchor4_seed%d" % s] = W(X[[C.ARCHS.index(a) for a in C.ANCHORS], j, :])
    for i, a in enumerate(C.ARCHS):
        out["cross_seed_%s" % a] = W(X[i])
    out["mean_cross_seed_all8"] = float(np.mean([out["cross_seed_%s" % a] for a in C.ARCHS]))
    out["mean_cross_arch_k8"] = float(np.mean([out["cross_arch_k8_seed%d" % s] for s in C.SEEDS]))
    return out


def aggregate(root: Path, out_root: Path) -> None:
    runs = load_runs(out_root)
    rep = rep_dir(root)
    long = []
    for _, r in runs.iterrows():
        for m in METHODS:
            row = {"arch": r.arch, "seed": int(r.seed), "repeat": int(r["repeat"]), "threads": int(r.threads),
                   "method": m, "published_auroc": published(root, r.arch, int(r.seed), m)}
            for var in ("old", "stable"):
                for k in ("auroc", "fpr95", "aupr_in", "aupr_out"):
                    row["%s_%s" % (var, k)] = float(r["%s_%s_%s" % (var, m, k)])
            row["old_neg_d2"] = int(r["old_mahalanobis_neg_d2"]) if m == "mahalanobis" else np.nan
            row["stable_neg_d2"] = int(r["stable_mahalanobis_neg_d2"]) if m == "mahalanobis" else np.nan
            row["old_precision_n_neg_eig"] = int(r.old_precision_n_neg_eig) if m == "mahalanobis" else np.nan
            row["stable_precision_n_neg_eig"] = int(r.stable_precision_n_neg_eig) if m == "mahalanobis" else np.nan
            row["vim_dim_old"] = int(r.old_vim_dim) if m == "vim" else np.nan
            row["vim_dim_stable"] = int(r.stable_vim_dim) if m == "vim" else np.nan
            row["feat_dim"] = int(r.feat_dim)
            long.append(row)
    long = pd.DataFrame(long).sort_values(["arch", "seed", "method", "repeat"])
    long.to_csv(rep / "old_runs.csv", index=False)

    sp = []
    for (a, s, m), g in long[long.arch.isin(C.ANCHORS)].groupby(["arch", "seed", "method"], sort=False):
        sp.append({"arch": a, "seed": s, "method": m, "n_runs": len(g), "published": g.published_auroc.iloc[0],
                   "mean": g.old_auroc.mean(), "std": g.old_auroc.std(ddof=1), "min": g.old_auroc.min(),
                   "max": g.old_auroc.max(), "range": g.old_auroc.max() - g.old_auroc.min(),
                   "published_minus_mean": g.published_auroc.iloc[0] - g.old_auroc.mean(),
                   "n_runs_within_2e-3_of_published": int((abs(g.old_auroc - g.published_auroc.iloc[0]) <= 2e-3).sum()),
                   "neg_d2_total": g.old_neg_d2.sum() if m == "mahalanobis" else np.nan,
                   "runs_with_neg_d2": int((g.old_neg_d2 > 0).sum()) if m == "mahalanobis" else np.nan,
                   "auroc_by_threads": " ".join("t%d:%.4f" % (t, x) for t, x in zip(g.threads, g.old_auroc))})
    sp = pd.DataFrame(sp)
    sp.to_csv(rep / "old_spread.csv", index=False)

    st = []
    for (a, s, m), g in long.groupby(["arch", "seed", "method"], sort=False):
        st.append({"arch": a, "seed": s, "method": m, "anchor": a in C.ANCHORS, "published": g.published_auroc.iloc[0],
                   "stable": g.stable_auroc.mean(), "stable_range_over_repeats": g.stable_auroc.max() - g.stable_auroc.min(),
                   "n_repeats": len(g), "stable_minus_published": g.stable_auroc.mean() - g.published_auroc.iloc[0],
                   "old_mean": g.old_auroc.mean(), "old_range": g.old_auroc.max() - g.old_auroc.min(),
                   "vim_dim_old": g.vim_dim_old.iloc[0], "vim_dim_stable": g.vim_dim_stable.iloc[0],
                   "stable_neg_d2_total": g.stable_neg_d2.sum() if m == "mahalanobis" else np.nan})
    st = pd.DataFrame(st)
    st.to_csv(rep / "stable_vs_published.csv", index=False)

    vt = value_table(root, runs)
    wrows = [dict(variant="published", **w_numbers(cube(root, {})))]
    for v in VARIANTS:
        X = cube(root, vt[v])
        if v == NOVIM:
            X = np.delete(X, C.METHODS_ORDER.index("ViM"), axis=2)
        wrows.append(dict(variant=v, **w_numbers(X)))
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(rep / "w_per_variant.csv", index=False)
    old = wdf[wdf.variant.str.startswith("old_")]
    pub = wdf[wdf.variant == "published"].iloc[0]
    stab = wdf[wdf.variant == "stable"].iloc[0]
    ws = []
    for c in wdf.columns[1:]:
        ws.append({"statistic": c, "published": pub[c], "old_min": old[c].min(), "old_max": old[c].max(),
                   "old_range": old[c].max() - old[c].min(), "old_std": old[c].std(ddof=1), "stable": stab[c],
                   **{v: wdf[wdf.variant == v].iloc[0][c] for v in EXTRA}})
    pd.DataFrame(ws).to_csv(rep / "w_spread.csv", index=False)
    print("wrote %s (%d runs)" % (rep, len(runs)))


# ---------------------------------------------------------------------------------------------
# variant trees
# ---------------------------------------------------------------------------------------------
def _link_dir(src: Path, dst: Path, keep_real: set) -> None:
    """dst mirrors src with symlinks, except names in keep_real (created as real dirs)."""
    C.ensure_dir(dst)
    for p in src.iterdir():
        q = dst / p.name
        if p.name in keep_real:
            continue
        if not q.exists() and not q.is_symlink():
            os.symlink(p, q)


def _swap_csv(src: Path, dst: Path, vals: dict) -> None:
    df = pd.read_csv(src)
    for m, (au, fpr, _, _) in vals.items():
        sel = df.Method == m
        if sel.any():
            df.loc[sel, "AUROC"] = au
            if "FPR95" in df.columns:
                df.loc[sel, "FPR95"] = fpr
            for c in df.columns:
                if c.endswith("_CI_low") or c.endswith("_CI_high") or c == "Threshold":
                    df.loc[sel, c] = np.nan
    if dst.is_symlink():
        dst.unlink()
    df.to_csv(dst, index=False)


def variants(root: Path, out_root: Path) -> None:
    runs = load_runs(out_root)
    vt = value_table(root, runs)
    raw = out_root / "stability" / "raw"
    only = [x for x in os.environ.get("DST_STAB_VARIANTS", "").split(",") if x]
    for v in (only or VARIANTS):
        V = out_root / "stability" / "variants" / v
        if V.exists():
            shutil.rmtree(V)
        swap = vt[v]
        cells = sorted({(a, s) for a, s, _ in swap})
        # data, features
        C.ensure_dir(V / "outputs")
        os.symlink(root / "data", V / "data")
        os.symlink(root / "outputs/features", V / "outputs/features")
        # reports: everything linked except camelyon17/frac1/seed*, ranks file, rigor_pack
        R, RV = root / "outputs/reports", V / "outputs/reports"
        _link_dir(R, RV, {"camelyon17", "architecture_invariance_ranks.csv", "rigor_pack"})
        _link_dir(R / "camelyon17", RV / "camelyon17", {"frac1"})
        _link_dir(R / "camelyon17/frac1", RV / "camelyon17/frac1", {"seed%d" % s for s in C.SEEDS})
        for s in C.SEEDS:
            _link_dir(R / ("camelyon17/frac1/seed%d" % s), RV / ("camelyon17/frac1/seed%d" % s), set())
        for a, s in cells:
            f = "%s_score_comparison.csv" % C.file_stem(a)
            vals = {m: swap[(a, s, m)] for m in METHODS if (a, s, m) in swap}
            _swap_csv(R / ("camelyon17/frac1/seed%d" % s) / f, RV / ("camelyon17/frac1/seed%d" % s) / f, vals)
        rk = pd.read_csv(C.ranks_file(root))
        for (a, s, m), vals in swap.items():
            if s == 42:
                rk.loc[(rk.domain == "camelyon17") & (rk.backbone == a) & (rk.method == C.DISPLAY[m]), "auroc"] = vals[0]
        rk["rank"] = rk.groupby(["domain", "backbone"]).auroc.rank(ascending=False, method="average")
        rk.to_csv(RV / "architecture_invariance_ranks.csv", index=False)
        # inputs the rigor stages read from the rigor-pack report tree
        RP = C.ensure_dir(RV / "rigor_pack")
        for name in ("hospital2", "vit_seeds", "recipe_seeds", "inputs_status.csv"):
            if (R / "rigor_pack" / name).exists():
                os.symlink(R / "rigor_pack" / name, RP / name)
        # score caches
        S, SV = out_root / "scores/camelyon17", V / "outputs/rigor_pack/scores/camelyon17"
        for s in C.SEEDS:
            _link_dir(S / ("seed%d" % s), SV / ("seed%d" % s), set())
        for a, s in cells:
            dst = SV / ("seed%d" % s) / ("%s.npz" % C.file_stem(a))
            z = dict(np.load(S / ("seed%d" % s) / ("%s.npz" % C.file_stem(a)), allow_pickle=False))
            for m, (rep, pre) in spec(v).items():
                new = np.load(raw / ("%s_s%d_r%d.npz" % (C.file_stem(a), s, rep)))
                z["id_" + m] = new["id_%s_%s" % (pre, m)]
                z["ood_" + m] = new["ood_%s_%s" % (pre, m)]
            dst.unlink()
            np.savez(dst, **z)
        print("variant %s: %d cells swapped -> %s" % (v, len(cells), V), flush=True)


# ---------------------------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------------------------
def verdicts(root: Path, out_root: Path) -> None:
    import verdict_reader as VR

    rep = rep_dir(root)
    rows = []
    base = VR.read_all(C.default_reports(root))
    for h, (v, e) in base.items():
        rows.append({"variant": "as_run", "H": h, "verdict": v, "evidence": e})
    per = {}
    for v in VARIANTS:
        d = VR.read_all(out_root / "stability" / "variants" / v / "outputs/reports/rigor_pack", only=VR.DEPENDS_MAHA_VIM)
        per[v] = d
        for h, (vv, e) in d.items():
            rows.append({"variant": v, "H": h, "verdict": vv, "evidence": e})
    pd.DataFrame(rows).to_csv(rep / "verdicts_by_variant.csv", index=False)

    ba = []
    for h, (v0, e0) in base.items():
        if h not in VR.DEPENDS_MAHA_VIM:
            ba.append({"H": h, "depends_on_anchor_maha_vim": False, "as_run": v0, "old_10_repeats": "",
                       "stable": "", **{x: "" for x in EXTRA}, "final": v0})
            continue
        olds = [per["old_r%d" % r][h][0] for r in range(N_OLD)]
        st = per["stable"][h][0]
        uniq = sorted(set(olds))
        old_txt = uniq[0] + " (10/10)" if len(uniq) == 1 else "; ".join(
            "%s (%d/10)" % (u, olds.count(u)) for u in uniq)
        same = len(uniq) == 1 and uniq[0] == st and "missing" not in st and "error" not in st
        final = uniq[0] if same else "INCONCLUSIVE (numerical instability)"
        if h == "H14":
            final = "descriptive; see families" if same else "INCONCLUSIVE (numerical instability) for affected families"
        ba.append({"H": h, "depends_on_anchor_maha_vim": True, "as_run": v0, "old_10_repeats": old_txt,
                   "stable": st, **{x: per[x][h][0] for x in EXTRA}, "final": final})
    ba = pd.DataFrame(ba)
    ba.to_csv(rep / "verdicts_before_after.csv", index=False)
    lines = ["| H | depends on anchor Maha/ViM | as run (precommit pipeline) | old code, 10 repeats | "
             "stable (Maha LW64 + ViM 90%) | final (addendum rule) | extra: stable Maha + ViM 8 threads | "
             "extra: stable Maha, no ViM |", "|---|---|---|---|---|---|---|---|"]
    for _, r in ba.iterrows():
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r.H, "yes" if r.depends_on_anchor_maha_vim else "no", r.as_run, r.old_10_repeats or "-",
            r.stable or "-", r.final, r[EXTRA[0]] or "-", r[EXTRA[1]] or "-"))
    C.write_text(rep / "verdicts_before_after.md", lines)
    print("\n".join(lines))
