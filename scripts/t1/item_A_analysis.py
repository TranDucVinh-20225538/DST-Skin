#!/usr/bin/env python3
"""T1 item A: analysis of A1 (cross-backbone LOBO) and A2 (within-backbone designs); writes the A tables.

    item_A_analysis.py a1   -> cells.csv, a1_lobo.csv, a1_family.csv, a1_summary.json
    item_A_analysis.py a2   -> a2_designs.csv, a2_lobo.csv, a2_summary.json
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

OUT = CM.RES / "A"
RAW = OUT / "raw"
NB = 10000


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def lobo(y, z):
    """Leave-one-unit-out OLS y = a + b z (z None -> constant). Returns held-out predictions."""
    y = np.asarray(y, float)
    n = len(y)
    pred = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        if z is None:
            pred[i] = y[m].mean()
        else:
            zz = np.asarray(z, float)
            A = np.c_[np.ones(m.sum()), zz[m]]
            coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            pred[i] = coef[0] + coef[1] * zz[i]
    return pred


def lobo_points(y, z, unit):
    """LOBO over units with several points per unit (A2): OLS on all points of the other units (z: n x k or None)."""
    y = np.asarray(y, float)
    unit = np.asarray(unit)
    pred = np.empty(len(y))
    for u in np.unique(unit):
        m = unit != u
        if z is None:
            pred[~m] = y[m].mean()
        else:
            Z = np.asarray(z, float).reshape(len(y), -1)
            A = np.c_[np.ones(m.sum()), Z[m]]
            coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            pred[~m] = np.c_[np.ones((~m).sum()), Z[~m]] @ coef
    return pred


def boot_mean_ci(x, seed, nb=NB):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(nb, len(x)))
    bm = x[idx].mean(1)
    return float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975)), float(np.mean(bm <= 0))


def r2_oos(y, pred, pred0):
    y = np.asarray(y, float)
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - pred0) ** 2))


def perm_spearman(x, y, seed=102, nperm=10000):
    rho = spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(nperm)])
    return float(rho), float((1 + np.sum(null >= rho)) / (nperm + 1))


def load_a1():
    cells = [json.load(open(p)) for p in sorted(glob.glob(str(RAW / "a1_*.json")))]
    rows = []
    for c in cells:
        for f in c["folds"]:
            rows.append(dict(dataset=c["dataset"], backbone=c["backbone"], family=c["family"], seed=c["seed"],
                             delta_meas_cell=c["delta_meas"], delta_own_cell=c["delta_own"], **f))
    return cells, rows


def unit_table(rows, unit_key):
    """Average over seeds and folds within a unit (backbone): every predictor / descriptor and Delta_meas."""
    units = sorted({r[unit_key] for r in rows})
    tab = []
    for u in units:
        rs = [r for r in rows if r[unit_key] == u]
        cells = {r["cell_id"]: r["delta_meas_cell"] for r in rs}
        t = dict(unit=u, family=rs[0]["family"], dataset=rs[0]["dataset"], n_cells=len(cells),
                 delta_meas=float(np.mean(list(cells.values()))))
        for k in ("p_delta", "p_delta_mult", "p_delta_gauss", "rho_raw", "rho_w", "rho_anova", "d_over_N", "s_logo",
                  "dq_pred", "dq_meas", "r_sat", "lw_delta", "N", "d", "G", "s_tr"):
            t[k] = float(np.mean([r[k] for r in rs]))
        t["rho_w_sq"] = float(np.mean([r["rho_w"] ** 2 for r in rs]))
        t["rho_w_s_logo"] = float(np.mean([r["rho_w"] * r["s_logo"] for r in rs]))
        tab.append(t)
    return tab


BASELINES = {"B0": None, "B1": "rho_raw", "B1w": "rho_w", "B3": "d_over_N", "B4": "rho_w_sq", "B5": "rho_w_s_logo"}


def lobo_block(tab, label):
    y = np.array([t["delta_meas"] for t in tab])
    preds = {"P": lobo(y, [t["p_delta"] for t in tab]), "P_zero_param": np.array([t["p_delta"] for t in tab])}
    for b, k in BASELINES.items():
        preds[b] = lobo(y, None if k is None else [t[k] for t in tab])
    err = {k: np.abs(y - v) for k, v in preds.items()}
    out = []
    for k, v in preds.items():
        rho, p = perm_spearman(v, y) if len(y) >= 4 and k != "B0" else (float("nan"), float("nan"))
        out.append(dict(block=label, predictor=k, n_units=len(y), mae=float(err[k].mean()),
                        r2_oos_vs_B0=r2_oos(y, v, preds["B0"]), spearman=rho, spearman_perm_p=p))
    return preds, err, out


def a1():
    cells, rows = load_a1()
    write_csv(OUT / "cells.csv", rows)
    cam = [r for r in rows if r["dataset"] == "camelyon"]
    tab = unit_table(cam, "backbone")
    y = np.array([t["delta_meas"] for t in tab])
    preds, err, lobo_rows = lobo_block(tab, "camelyon_LOBO_13")
    for t, i in zip(tab, range(len(tab))):
        for k in preds:
            t[f"pred_{k}"] = float(preds[k][i])
            t[f"err_{k}"] = float(err[k][i])
    D = {}
    for b in ("B1", "B1w", "B3", "B4", "B5"):
        db = err[b] - err["P"]
        lo, hi, p = boot_mean_ci(db, 101)
        D[b] = dict(mean_D=float(db.mean()), ci_lo=lo, ci_hi=hi, p_boot_one_sided=p, D_b=db.tolist())
    # Delta-Q level (A-iv)
    dq_cells = {}
    for r in cam:
        dq_cells.setdefault(r["cell_id"], []).append((r["dq_pred"], r["dq_meas"]))
    rel_cell = [abs(np.mean([a for a, _ in v]) / np.mean([b for _, b in v]) - 1) for v in dq_cells.values()]
    rel_cellfold = [abs(r["dq_pred"] / r["dq_meas"] - 1) for r in cam]
    ydq = np.array([t["dq_meas"] for t in tab])
    pdq = lobo(ydq, [t["dq_pred"] for t in tab])
    pdq0 = lobo(ydq, None)
    r2dq = r2_oos(ydq, pdq, pdq0)
    rsat_cellfold = np.array([r["r_sat"] for r in cam])
    rsat_cell = np.array([np.mean([r["r_sat"] for r in cam if r["cell_id"] == c]) for c in dq_cells])
    frac_lt09 = dict(cell_fold=float(np.mean(rsat_cellfold < 0.9)), cell=float(np.mean(rsat_cell < 0.9)))
    b5_eligible = dict(cell_fold=frac_lt09["cell_fold"] >= 0.2, cell=frac_lt09["cell"] >= 0.2)
    frac_ge09 = dict(cell_fold=float(np.mean(rsat_cellfold >= 0.9)), cell=float(np.mean(rsat_cell >= 0.9)))
    # family / within-family (report only)
    fam_rows = []
    for train_f, test_f in (("CNN", "FM"), ("FM", "CNN")):
        tr = [t for t in tab if t["family"] == train_f]
        te = [t for t in tab if t["family"] == test_f]
        ytr, yte = np.array([t["delta_meas"] for t in tr]), np.array([t["delta_meas"] for t in te])
        for k, key in [("P", "p_delta")] + [(b, kk) for b, kk in BASELINES.items()]:
            if key is None:
                pr = np.full(len(te), ytr.mean())
            else:
                A = np.c_[np.ones(len(tr)), [t[key] for t in tr]]
                coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
                pr = coef[0] + coef[1] * np.array([t[key] for t in te])
            fam_rows.append(dict(block=f"train_{train_f}_test_{test_f}", predictor=k, n_test=len(te), mae=float(np.abs(yte - pr).mean())))
    for fam in ("FM", "CNN"):
        sub = [t for t in tab if t["family"] == fam]
        _, _, lr = lobo_block(sub, f"within_{fam}_LOBO_n{len(sub)}")
        fam_rows += lr
    # medbench leave-one-dataset-out (report only): unit = dataset x backbone
    med = [r for r in rows if r["dataset"] != "camelyon"]
    for r in med:
        r["unit_db"] = f"{r['dataset']}|{r['backbone']}"
    mtab = unit_table(med, "unit_db")
    for t in mtab:
        t["dataset"] = t["unit"].split("|")[0]
    dsets = sorted({t["dataset"] for t in mtab})
    for k, key in [("P", "p_delta"), ("P_zero_param", None)] + [(b, kk) for b, kk in BASELINES.items() if b != "B0"] + [("B0", None)]:
        errs = []
        for ds in dsets:
            tr = [t for t in mtab if t["dataset"] != ds]
            te = [t for t in mtab if t["dataset"] == ds]
            ytr, yte = np.array([t["delta_meas"] for t in tr]), np.array([t["delta_meas"] for t in te])
            if k == "P_zero_param":
                pr = np.array([t["p_delta"] for t in te])
            elif k == "B0":
                pr = np.full(len(te), ytr.mean())
            else:
                A = np.c_[np.ones(len(tr)), [t[key] for t in tr]]
                coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
                pr = coef[0] + coef[1] * np.array([t[key] for t in te])
            errs.append(np.abs(yte - pr))
        fam_rows.append(dict(block="medbench_leave_one_dataset_out_n4", predictor=k, n_test=int(sum(len(e) for e in errs)),
                             mae=float(np.concatenate(errs).mean())))
    write_csv(OUT / "a1_lobo.csv", lobo_rows + [dict(block="camelyon_LOBO_13", predictor=f"D_b_vs_{b}", n_units=13,
                                                    mean_D=v["mean_D"], ci_lo=v["ci_lo"], ci_hi=v["ci_hi"]) for b, v in D.items()])
    write_csv(OUT / "a1_family.csv", fam_rows)
    write_csv(OUT / "a1_backbones.csv", tab)
    summ = dict(n_camelyon_cells=len(dq_cells), n_backbones=len(tab), D=D,
                A_i=dict(pass_=bool(D["B1"]["ci_lo"] > 0 and D["B1w"]["ci_lo"] > 0)),
                r_sat_frac_lt_0_9=frac_lt09, r_sat_frac_ge_0_9=frac_ge09, B5_in_A_ii=b5_eligible,
                A_iv=dict(median_rel_err_cell=float(np.median(rel_cell)), median_rel_err_cell_fold=float(np.median(rel_cellfold)),
                          r2_oos_dq_vs_B0=r2dq,
                          pass_cell=bool(np.median(rel_cell) <= 0.35 and r2dq > 0),
                          pass_cell_fold=bool(np.median(rel_cellfold) <= 0.35 and r2dq > 0)),
                max_abs_diff_own_vs_r3=max(c["abs_diff_own_vs_r3"] for c in cells if c["abs_diff_own_vs_r3"] is not None))
    for key in ("cell_fold", "cell"):
        bl = ["B3", "B4"] + (["B5"] if b5_eligible[key] else [])
        summ[f"A_ii_{key}"] = dict(baselines=bl, pass_=bool(all(D[b]["ci_lo"] > 0 for b in bl)))
    CM.atomic_write_text(OUT / "a1_summary.json", json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "D"}, indent=1))
    print({b: (round(v["mean_D"], 5), round(v["ci_lo"], 5), round(v["ci_hi"], 5)) for b, v in D.items()})


def load_a2():
    rows = []
    for p in sorted(glob.glob(str(RAW / "a2_*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            r["design"] = f"G{r['G_prime']:02d}|n{r['n_prime']}"
            r["log_G"] = float(np.log(r["G_prime"]))
            r["log_n"] = float(np.log(r["N"] / r["G_prime"]))
            r["rho_w_sq"] = r["rho_w"] ** 2
            r["rho_w_s_logo"] = r["rho_w"] * r["s_logo"]
            rows.append(r)
    return rows


A2_BASELINES = {"B0": None, "B3": ["d_over_N"], "B4": ["rho_w_sq"], "B5": ["rho_w_s_logo"],
                "B6": ["rho_w", "log_G", "log_n", "d_over_N"]}


def a2():
    rows = load_a2()
    bbs = sorted({r["backbone"] for r in rows})
    complete, s5 = [], []
    for b in bbs:
        n_units = len({r["unit"] for r in rows if r["backbone"] == b})
        for f in (0, 1):
            a1 = json.load(open(RAW / f"a1_camelyon_{b}_s42.json"))
            ref = [x for x in a1["folds"] if x["fold"] == f][0]["delta_meas_fold"]
            full = [r["delta_meas"] for r in rows if r["backbone"] == b and r["fold"] == f and r["design"] == "G15|nall"]
            dev = max(abs(v - ref) for v in full) if full else float("nan")
            s5.append(dict(backbone=b, fold=f, a1_fold_delta=ref, n_full_design=len(full), max_abs_dev=dev, ok=bool(full) and dev <= 1e-6))
        if n_units == 600 and all(x["ok"] for x in s5 if x["backbone"] == b):
            complete.append(b)
    use = [r for r in rows if r["backbone"] in complete]
    y = np.array([r["delta_meas"] for r in use])
    unit = np.array([r["backbone"] for r in use])
    preds = {"P": lobo_points(y, [r["p_delta"] for r in use], unit), "P_zero_param": np.array([r["p_delta"] for r in use])}
    for b, ks in A2_BASELINES.items():
        preds[b] = lobo_points(y, None if ks is None else [[r[k] for k in ks] for r in use], unit)
    err = {k: np.abs(y - v) for k, v in preds.items()}
    eb = {k: np.array([v[unit == b].mean() for b in complete]) for k, v in err.items()}
    lobo_rows = [dict(block="A2_LOBO", predictor=k, n_backbones=len(complete), n_points=len(y), pooled_mae=float(err[k].mean()),
                      r2_oos_vs_B0=r2_oos(y, preds[k], preds["B0"])) for k in preds]
    D = {}
    for b in ("B3", "B4", "B5", "B6"):
        db = eb[b] - eb["P"]
        lo, hi, p = boot_mean_ci(db, 101)
        D[b] = dict(mean_D=float(db.mean()), ci_lo=lo, ci_hi=hi, p_boot_one_sided=p, D_b=db.tolist())
        lobo_rows.append(dict(block="A2_LOBO", predictor=f"D_b_vs_{b}", n_backbones=len(complete), mean_D=D[b]["mean_D"], ci_lo=lo, ci_hi=hi))
    designs = sorted({r["design"] for r in use})
    des_rows, sp = [], []
    for b in complete:
        mp, mm = [], []
        for d in designs:
            rs = [r for r in use if r["backbone"] == b and r["design"] == d]
            row = dict(backbone=b, design=d, G_prime=rs[0]["G_prime"], n_prime=rs[0]["n_prime"], n_points=len(rs))
            for k in ("delta_meas", "p_delta", "dq_pred", "dq_meas", "rho_w", "rho_raw", "s_logo", "r_sat", "lw_delta", "N", "d_over_N"):
                row[k] = float(np.mean([r[k] for r in rs]))
            des_rows.append(row)
            mp.append(row["p_delta"])
            mm.append(row["delta_meas"])
        sp.append(float(spearmanr(mp, mm).correlation))
    sp = np.array(sp)
    lo, hi, _ = boot_mean_ci(sp, 101)
    spear = dict(per_backbone=dict(zip(complete, sp.tolist())), mean=float(sp.mean()), ci_lo=lo, ci_hi=hi)
    a_iii_eval = len(complete) >= 8
    a_iii = dict(evaluable=a_iii_eval, n_complete=len(complete), D=D, spearman=spear,
                 pass_=(bool(D["B6"]["ci_lo"] > 0 and D["B5"]["ci_lo"] > 0 and spear["mean"] > 0 and spear["ci_lo"] > 0)
                        if a_iii_eval else None))
    write_csv(OUT / "a2_designs.csv", des_rows)
    write_csv(OUT / "a2_lobo.csv", lobo_rows)
    write_csv(OUT / "a2_s5.csv", s5)
    CM.atomic_write_text(OUT / "a2_summary.json", json.dumps(dict(complete=complete, A_iii=a_iii), indent=1))
    print(json.dumps(dict(complete=len(complete), s5_fail=[x for x in s5 if not x["ok"]], A_iii={k: v for k, v in a_iii.items() if k != "D"},
                          D={b: (round(v["mean_D"], 5), round(v["ci_lo"], 5), round(v["ci_hi"], 5)) for b, v in D.items()}), indent=1))


def a3():
    rows = []
    for p in sorted(glob.glob(str(RAW / "a3_*.json"))):
        c = json.load(open(p))
        rows += c["folds"]
    write_csv(OUT / "a3_audit.csv", rows)
    cells = sorted({r["cell_id"] for r in rows})
    flags = {k: sorted({r["cell_id"] for r in rows if r[k]}) for k in ("flag_r_sat_ge_0_9", "flag_delta_ge_0_5", "flag_N_lt_5d")}
    summ = dict(n_cells=len(cells), n_rows=len(rows), flags={k: dict(n_cells=len(v), cells=v) for k, v in flags.items()},
                median_rel_se_rho_w=float(np.median([r["se_rho_w"] / abs(r["rho_w"]) for r in rows if r["rho_w"] != 0])),
                median_rel_se_dq_pred=float(np.median([r["se_dq_pred"] / abs(r["dq_pred"]) for r in rows if r["dq_pred"] != 0])),
                median_n_w_over_n_bar=float(np.median([r["n_w_over_n_bar"] for r in rows])),
                max_abs_conversion_effect=float(max(abs(r["dq_pred"] / r["dq_pred_without_conversion"] - 1) for r in rows)))
    CM.atomic_write_text(OUT / "a3_summary.json", json.dumps(summ, indent=1))
    print(json.dumps({k: (v if k != "flags" else {kk: vv["n_cells"] for kk, vv in v.items()}) for k, v in summ.items()}, indent=1))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def final():
    s1 = json.load(open(OUT / "a1_summary.json"))
    s2 = json.load(open(OUT / "a2_summary.json"))
    a0 = {k: json.load(open(OUT / f"{k}.json")) for k in ("a0a", "a0b", "a0c")}
    tests = []
    for b in ("B1", "B1w", "B3", "B4", "B5"):
        v = s1["D"][b]
        tests.append(dict(test_id=f"A1_D_vs_{b}", item="A", quantity=f"mean D_b (e_{b} - e_P), Camelyon LOBO", n_units=13,
                          statistic=v["mean_D"], null="0", p_raw=v["p_boot_one_sided"], ci_lo=v["ci_lo"], ci_hi=v["ci_hi"], status="prereg"))
    for b in ("B5", "B6"):
        v = s2["A_iii"]["D"][b]
        tests.append(dict(test_id=f"A2_D_vs_{b}", item="A", quantity=f"mean D_b (e_{b} - e_P), A2 LOBO pooled over designs",
                          n_units=s2["A_iii"]["n_complete"], statistic=v["mean_D"], null="0", p_raw=v["p_boot_one_sided"],
                          ci_lo=v["ci_lo"], ci_hi=v["ci_hi"], status="prereg"))
    sp = s2["A_iii"]["spearman"]
    tests.append(dict(test_id="A2_within_backbone_spearman", item="A", quantity="mean within-backbone Spearman(P_delta, Delta_meas) over 30 designs",
                      n_units=s2["A_iii"]["n_complete"], statistic=sp["mean"], null="0", p_raw="", ci_lo=sp["ci_lo"], ci_hi=sp["ci_hi"], status="prereg"))
    tests.append(dict(test_id="A1_dq_median_rel_err", item="A", quantity="median |dQ_pred/dQ_meas - 1| over Camelyon cells (zero-parameter)",
                      n_units=s1["n_camelyon_cells"], statistic=s1["A_iv"]["median_rel_err_cell"], null="<= 0.35", p_raw="", ci_lo="", ci_hi="", status="prereg"))
    tests.append(dict(test_id="A1_dq_r2_oos", item="A", quantity="pooled R2_oos of LOBO-calibrated dQ_pred vs B0", n_units=13,
                      statistic=s1["A_iv"]["r2_oos_dq_vs_B0"], null="0", p_raw="", ci_lo="", ci_hi="", status="prereg"))
    pr = [t["p_raw"] for t in tests if t["p_raw"] != ""]
    adj = iter(holm(pr))
    for t in tests:
        t["p_holm"] = float(next(adj)) if t["p_raw"] != "" else ""
    a_i = s1["A_i"]["pass_"]
    a_ii = s1["A_ii_cell_fold"]["pass_"]
    a_ii_alt = s1["A_ii_cell"]["pass_"]
    a_iii = s2["A_iii"]["pass_"]
    a_iv = s1["A_iv"]["pass_cell"]
    a_iv_alt = s1["A_iv"]["pass_cell_fold"]
    status = {k: ("not evaluable" if v is None else ("pass" if v else "fail")) for k, v in
              dict(A_i=a_i, A_ii=a_ii, A_iii=a_iii, A_iv=a_iv).items()}
    for t in tests:
        t["pass"] = {"A1_D_vs_B1": s1["D"]["B1"]["ci_lo"] > 0, "A1_D_vs_B1w": s1["D"]["B1w"]["ci_lo"] > 0,
                     "A1_D_vs_B3": s1["D"]["B3"]["ci_lo"] > 0, "A1_D_vs_B4": s1["D"]["B4"]["ci_lo"] > 0,
                     "A1_D_vs_B5": s1["D"]["B5"]["ci_lo"] > 0,
                     "A2_D_vs_B5": s2["A_iii"]["D"]["B5"]["ci_lo"] > 0, "A2_D_vs_B6": s2["A_iii"]["D"]["B6"]["ci_lo"] > 0,
                     "A2_within_backbone_spearman": sp["mean"] > 0 and sp["ci_lo"] > 0,
                     "A1_dq_median_rel_err": s1["A_iv"]["median_rel_err_cell"] <= 0.35,
                     "A1_dq_r2_oos": s1["A_iv"]["r2_oos_dq_vs_B0"] > 0}[t["test_id"]]
        if not s2["A_iii"]["evaluable"] and t["test_id"].startswith("A2_"):
            t["pass"] = "na"
    write_csv(OUT / "tests.csv", tests)
    frac_ge = s1["r_sat_frac_ge_0_9"]["cell"]
    if a_i and a_ii and a_iii and a_iv:
        verdict = "GO"
    elif (a_i and (a_iii or a_iv)) or (a_iii and a_iv and not a_i):
        verdict = "PARTIAL"
    elif frac_ge >= 0.8 and a_iii is None:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO-GO"
    out = dict(verdict=verdict, status=status, a_ii_alt_reading_cell=a_ii_alt, a_iv_alt_reading_cell_fold=a_iv_alt,
               verdict_reading_dependent=bool(a_ii != a_ii_alt or a_iv != a_iv_alt), r_sat_frac_ge_0_9=s1["r_sat_frac_ge_0_9"],
               a0={k: v[f"pass_{k.upper()[0]}{k[1:]}"] for k, v in a0.items()})
    CM.atomic_write_text(OUT / "verdict.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tab = list(csv.DictReader(open(OUT / "a1_backbones.csv")))
    _, rows = load_a1()
    cam = [r for r in rows if r["dataset"] == "camelyon"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for ds, mk in (("camelyon", "o"), ("breakhis", "s"), ("dermamnist", "^"), ("isic2019", "v"), ("kermany", "D")):
        rs = [r for r in rows if r["dataset"] == ds]
        ax.scatter([-r["dq_meas"] for r in rs], [-r["dq_pred"] for r in rs], s=10, marker=mk, label=ds, alpha=0.7)
    lim = [1e-1, 1e5]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="-dQ measured (scorer units)", ylabel="-dQ predicted (P_dQ)", title="A1: dQ, cell x fold")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_dq.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter([float(t["delta_meas"]) for t in tab], [float(t["p_delta"]) for t in tab], c="C0", label="P_delta (zero-param)")
    ax.scatter([float(t["delta_meas"]) for t in tab], [float(t["pred_P"]) for t in tab], c="C1", marker="x", label="P_delta LOBO-calibrated")
    for t in tab:
        ax.annotate(t["unit"], (float(t["delta_meas"]), float(t["p_delta"])), fontsize=6)
    ax.plot([0, 0.25], [0, 0.25], "k--", lw=0.8)
    ax.set(xlabel="Delta measured (backbone mean)", ylabel="Delta predicted", title="A1: Camelyon, 13 backbones")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_delta.png", dpi=150)
    plt.close(fig)
    des = list(csv.DictReader(open(OUT / "a2_designs.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for b in sorted({r["backbone"] for r in des}):
        rs = [r for r in des if r["backbone"] == b and r["n_prime"] == "all"]
        rs.sort(key=lambda r: int(r["G_prime"]))
        axes[0].plot([int(r["G_prime"]) for r in rs], [float(r["delta_meas"]) for r in rs], "-o", ms=3, label=b)
        axes[1].plot([int(r["G_prime"]) for r in rs], [float(r["p_delta"]) for r in rs], "-o", ms=3)
    axes[0].set(xlabel="G' (n' = all)", ylabel="Delta measured", title="A2 measured")
    axes[1].set(xlabel="G' (n' = all)", ylabel="P_delta", title="A2 predicted (zero-param)")
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "a2_dose.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    {"a1": a1, "a2": a2, "a3": a3, "final": final, "figures": figures}[sys.argv[1]]()
