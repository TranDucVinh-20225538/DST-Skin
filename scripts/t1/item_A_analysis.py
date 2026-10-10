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


if __name__ == "__main__":
    {"a1": a1}[sys.argv[1]]()
