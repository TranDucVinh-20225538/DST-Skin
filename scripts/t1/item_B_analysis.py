#!/usr/bin/env python3
"""T1 item B: analysis of the kNN cells -> cells_knn.csv, perk.csv, collapse_lobo.csv, mts.csv, tests.csv,
collapse.png, perk.png, b_summary.json, verdict.json.

Decision unit (B-i, B-ii): Camelyon backbone (13; values averaged over seeds and folds), leave-one-backbone-out.
Single-descriptor maps are isotonic (equal capacity); x1, x2, x3 and rho_cos increasing (direction fixed by K4),
d/N, d and T_unseen with the direction chosen on the training units ('auto'). Paired LOBO error differences
D_b = |err_baseline| - |err_predictor| with a unit bootstrap (B = 10,000; default_rng(201) for B-i, 202 for B-ii).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
from item_A_analysis import boot_mean_ci, holm, lobo, r2_oos, write_csv  # noqa: E402

OUT = CM.RES / "B"
RAW = OUT / "raw"
KS = [1, 5, 10, 20, 50, 100, 200]
INC = {"x1": True, "x2": True, "x3": True, "rho_cos": True, "d_over_N": "auto", "d": "auto", "T_unseen": "auto"}


def iso_lobo(y, x, unit, increasing):
    y, x, unit = np.asarray(y, float), np.asarray(x, float), np.asarray(unit)
    pred = np.empty(len(y))
    for u in np.unique(unit):
        m = unit != u
        ir = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(x[m], y[m])
        pred[~m] = ir.predict(x[~m])
    return pred


def mean_lobo(y, unit):
    y, unit = np.asarray(y, float), np.asarray(unit)
    pred = np.empty(len(y))
    for u in np.unique(unit):
        pred[unit == u] = y[unit != u].mean()
    return pred


def load():
    cells = [json.load(open(p)) for p in sorted(glob.glob(str(RAW / "b_*.json")))]
    rows, perk = [], []
    for c in cells:
        for f in c["folds"]:
            mts = f["mts"]
            dm = dict(zip(mts["ks"], mts["delta_mts"]))
            r = dict(cell_id=c["cell_id"], dataset=c["dataset"], backbone=c["backbone"], family=c["family"], seed=c["seed"],
                     **{k: v for k, v in f.items() if k not in ("perk", "mts", "cell_id")},
                     delta_k50=f["perk"]["50"]["delta"] if "50" in f["perk"] else np.nan,
                     delta_mts_k50=dm.get(50, np.nan), r_nn_k50=f["perk"].get("50", {}).get("r_nn", np.nan),
                     mts_clipped_k50=bool(dict(zip(mts["ks"], np.logical_or(mts["bracket_clipped_low"], mts["bracket_clipped_high"]))).get(50, False)))
            rows.append(r)
            for k in KS:
                if str(k) not in f["perk"]:
                    continue
                p = f["perk"][str(k)]
                perk.append(dict(cell_id=c["cell_id"], dataset=c["dataset"], backbone=c["backbone"], fold=f["fold"], k=k,
                                 gsize_median=f["gsize_median"], gsize_min=f["gsize_min"], k_le_ng=k <= f["gsize_median"],
                                 x1=f["x1"], x2=f["x2"], x3=f["x3"], rho_cos=f["rho_cos"], d=f["d"],
                                 delta_mts=dm.get(k, np.nan), **p))
    return cells, rows, perk


def unit_tab(rows, key):
    tab = []
    for u in sorted({r[key] for r in rows}):
        rs = [r for r in rows if r[key] == u]
        t = dict(unit=u, dataset=rs[0]["dataset"], family=rs[0]["family"], n_cells=len({r["cell_id"] for r in rs}))
        for k in ("delta_k50", "delta_mts_k50", "x1", "x2", "x3", "rho_cos", "d", "d_over_N", "T_unseen", "pr", "twonn", "r_nn_k50"):
            t[k] = float(np.mean([r[k] for r in rs]))
        tab.append(t)
    return tab


def main():
    cells, rows, perk = load()
    write_csv(OUT / "cells_knn.csv", rows)
    write_csv(OUT / "perk.csv", perk)
    cam = [r for r in rows if r["dataset"] == "camelyon"]
    tab = unit_tab(cam, "backbone")
    y = np.array([t["delta_k50"] for t in tab])
    units = np.array([t["unit"] for t in tab])
    preds = {"B0": mean_lobo(y, units)}
    for k, inc in INC.items():
        preds[f"iso_{k}"] = iso_lobo(y, [t[k] for t in tab], units, inc)
    preds["lin_rho_cos"] = lobo(y, [t["rho_cos"] for t in tab])
    preds["MTS"] = np.array([t["delta_mts_k50"] for t in tab])
    preds["iso_r_nn_k50"] = iso_lobo(y, [t["r_nn_k50"] for t in tab], units, True)
    err = {k: np.abs(y - v) for k, v in preds.items()}
    col = []
    for k, v in preds.items():
        col.append(dict(block="camelyon_LOBO_13_k50", predictor=k, n_units=len(y), mae=float(err[k].mean()),
                        r2_oos_vs_B0=r2_oos(y, v, preds["B0"]), spearman=float(spearmanr(v, y).correlation)))
    # B-i: isotonic x1 vs B1 (isotonic rho_cos, equal capacity); B-ii: MTS vs B1
    D = {}
    for name, pred_key, base, seed in (("B_i", "iso_x1", "iso_rho_cos", 201), ("B_i_lin", "iso_x1", "lin_rho_cos", 201),
                                       ("B_ii", "MTS", "iso_rho_cos", 202), ("B_ii_lin", "MTS", "lin_rho_cos", 202)):
        db = err[base] - err[pred_key]
        lo, hi, p = boot_mean_ci(db, seed)
        D[name] = dict(predictor=pred_key, baseline=base, mean_D=float(db.mean()), ci_lo=lo, ci_hi=hi, p_boot_one_sided=p)
    r2_x1 = r2_oos(y, preds["iso_x1"], preds["B0"])
    b_i = bool(r2_x1 > 0 and D["B_i"]["ci_lo"] > 0)
    b_ii = bool(err["MTS"].mean() < err["iso_rho_cos"].mean() and D["B_ii"]["ci_lo"] > 0)
    # collapse across all cells x k with k <= n_g (report only): LOBO over backbones, unit = cell x fold x k point
    pts = [p for p in perk if p["k_le_ng"]]
    yk = np.array([p["delta"] for p in pts])
    ub = np.array([p["backbone"] for p in pts])
    p0 = mean_lobo(yk, ub)
    for k in ("x1", "x2", "x3", "rho_cos", "d"):
        pk = iso_lobo(yk, [p[k] for p in pts], ub, INC[k])
        col.append(dict(block="all_cells_k_le_ng_LOBO_backbone", predictor=f"iso_{k}", n_units=len(np.unique(ub)), n_points=len(yk),
                        mae=float(np.abs(yk - pk).mean()), r2_oos_vs_B0=r2_oos(yk, pk, p0)))
    write_csv(OUT / "collapse_lobo.csv", col)
    # MTS vs measured (all cells, report; Camelyon in B-ii)
    mrows = []
    for blk, rs in (("all_cells", rows), ("camelyon", cam)) + tuple((f"dataset_{ds}", [r for r in rows if r["dataset"] == ds]) for ds in sorted({r["dataset"] for r in rows})):
        a = np.array([r["delta_k50"] for r in rs])
        b = np.array([r["delta_mts_k50"] for r in rs])
        ok = np.isfinite(a) & np.isfinite(b)
        mrows.append(dict(block=blk, n_cell_folds=int(ok.sum()), mae=float(np.abs(a - b)[ok].mean()),
                          mean_delta_meas=float(a[ok].mean()), mean_delta_mts=float(b[ok].mean()),
                          spearman=float(spearmanr(a[ok], b[ok]).correlation),
                          frac_bracket_clipped=float(np.mean([r["mts_clipped_k50"] for r in rs]))))
    write_csv(OUT / "mts.csv", mrows)
    # per-k (report only): flatness for k <= n_g; drop beyond n_g on cells with median n_g < 100
    flat, drop = [], []
    for (cid, fo) in sorted({(p["cell_id"], p["fold"]) for p in perk}):
        ps = sorted([p for p in perk if p["cell_id"] == cid and p["fold"] == fo], key=lambda p: p["k"])
        inn = [p for p in ps if p["k_le_ng"]]
        if len(inn) >= 3:
            sl = np.polyfit(np.log([p["k"] for p in inn]), [p["delta"] for p in inn], 1)[0]
            flat.append(dict(cell_id=cid, fold=fo, slope_per_logk=float(sl), delta_range=float(np.ptp([p["delta"] for p in inn]))))
        if ps and ps[0]["gsize_median"] < 100:
            out_ = [p for p in ps if not p["k_le_ng"]]
            if inn and out_:
                drop.append(float(np.mean([p["delta"] for p in out_]) < np.mean([p["delta"] for p in inn])))
    perk_summary = dict(n_cell_folds_flat_test=len(flat),
                        median_abs_slope_per_logk=float(np.median([abs(f["slope_per_logk"]) for f in flat])) if flat else None,
                        median_slope_per_logk=float(np.median([f["slope_per_logk"] for f in flat])) if flat else None,
                        n_cell_folds_ng_lt_100=len(drop), frac_drop_beyond_ng=float(np.mean(drop)) if drop else None,
                        drop_test="testable" if drop else "not testable")
    tests = [dict(test="B-i: R2_oos(iso x1) > 0", value=r2_x1, pass_=r2_x1 > 0),
             dict(test="B-i: mean D_b vs B1 (iso rho_cos), lower 95% CI > 0", value=D["B_i"]["mean_D"], ci_lo=D["B_i"]["ci_lo"],
                  ci_hi=D["B_i"]["ci_hi"], p=D["B_i"]["p_boot_one_sided"], pass_=D["B_i"]["ci_lo"] > 0),
             dict(test="B-ii: MAE(MTS) < MAE(B1 iso rho_cos)", value=float(err["MTS"].mean() - err["iso_rho_cos"].mean()),
                  pass_=bool(err["MTS"].mean() < err["iso_rho_cos"].mean())),
             dict(test="B-ii: mean D_b MTS vs B1 (iso rho_cos), lower 95% CI > 0", value=D["B_ii"]["mean_D"], ci_lo=D["B_ii"]["ci_lo"],
                  ci_hi=D["B_ii"]["ci_hi"], p=D["B_ii"]["p_boot_one_sided"], pass_=D["B_ii"]["ci_lo"] > 0),
             dict(test="report: B-i vs linear B1", value=D["B_i_lin"]["mean_D"], ci_lo=D["B_i_lin"]["ci_lo"], ci_hi=D["B_i_lin"]["ci_hi"],
                  p=D["B_i_lin"]["p_boot_one_sided"], pass_="na"),
             dict(test="report: B-ii vs linear B1", value=D["B_ii_lin"]["mean_D"], ci_lo=D["B_ii_lin"]["ci_lo"], ci_hi=D["B_ii_lin"]["ci_hi"],
                  p=D["B_ii_lin"]["p_boot_one_sided"], pass_="na")]
    ps = [t.get("p") for t in tests]
    idx = [i for i, p in enumerate(ps) if p is not None]
    hp = holm([ps[i] for i in idx])
    for i, h in zip(idx, hp):
        tests[i]["p_holm"] = h
    write_csv(OUT / "tests.csv", tests)
    verdict = "GO" if b_i and b_ii else ("PARTIAL" if b_i or b_ii else "NO-GO")
    summ = dict(n_cells=len(cells), n_camelyon_backbones=len(tab), r2_oos_iso_x1=r2_x1, D=D, B_i=b_i, B_ii=b_ii, verdict=verdict,
                mae={k: float(v.mean()) for k, v in err.items()}, perk=perk_summary,
                max_abs_diff_k50_vs_r3=max(c["abs_diff_k50_vs_r3"] for c in cells if c["abs_diff_k50_vs_r3"] is not None),
                frac_mts_clipped_k50=float(np.mean([r["mts_clipped_k50"] for r in rows])), flat=flat)
    write_csv(OUT / "b_backbones.csv", [dict(t, **{f"pred_{k}": float(v[i]) for k, v in preds.items()}) for i, t in enumerate(tab)])
    CM.atomic_write_text(OUT / "b_summary.json", json.dumps(summ, indent=1))
    CM.atomic_write_text(OUT / "verdict.json", json.dumps(dict(item="B", verdict=verdict, B_i="pass" if b_i else "fail",
                                                               B_ii="pass" if b_ii else "fail"), indent=1))
    figures(rows, perk)
    print(json.dumps({k: v for k, v in summ.items() if k != "flat"}, indent=1))


def figures(rows, perk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for ds in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds]
        ax[0].scatter([r["x1"] for r in rs], [r["delta_k50"] for r in rs], s=12, label=ds)
        ax[1].scatter([r["delta_mts_k50"] for r in rs], [r["delta_k50"] for r in rs], s=12, label=ds)
    ax[0].set_xlabel("x1 = rho_cos sqrt(PR)")
    ax[0].set_ylabel("measured Delta (kNN, k = 50)")
    ax[0].set_xscale("log")
    ax[1].set_xlabel("Delta_MTS (k = 50)")
    lim = [min(ax[1].get_xlim()[0], ax[1].get_ylim()[0]), max(ax[1].get_xlim()[1], ax[1].get_ylim()[1])]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "collapse.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)
    for a, ds in zip(axs, sorted({p["dataset"] for p in perk})):
        for (cid, fo) in sorted({(p["cell_id"], p["fold"]) for p in perk if p["dataset"] == ds}):
            ps = sorted([p for p in perk if p["cell_id"] == cid and p["fold"] == fo], key=lambda p: p["k"])
            a.plot([p["k"] for p in ps], [p["delta"] for p in ps], lw=0.5, alpha=0.5)
        a.set_xscale("log")
        a.set_title(ds)
        a.set_xlabel("k")
    axs[0].set_ylabel("Delta(k)")
    fig.tight_layout()
    fig.savefig(OUT / "perk.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
