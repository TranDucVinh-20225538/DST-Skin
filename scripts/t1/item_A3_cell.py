#!/usr/bin/env python3
"""T1 item A3 (per cell): estimator audit.

Group-jackknife SE (delete-a-group if G <= 40, else delete-a-block over the 10 group folds used for s_logo) of
rho_w, rho_raw, rho_anova, s_logo and dQ_pred per paper_2fold fold; sensitivity of rho_w / dQ_pred to group-size
imbalance (n_w / n_bar plug-ins), to class-conditional vs class-agnostic centring (cells with labels_train only),
and to the A0c conversion factor 1/(1 - delta); regime flags r_sat >= 0.9, delta >= 0.5, N < 5d.

    item_A3_cell.py CELL [CELL ...]  -> results/t1/A/raw/a3_<cell>.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import scorer64 as S64  # noqa: E402
from item_A1_cell import logo_s  # noqa: E402

RAW = Path(__import__("os").environ.get("T1_RAW", str(CM.RES / "A" / "raw")))
REG = {r["cell_id"]: r for r in csv.DictReader(open(CM.RES / "I" / "registry.csv"))}


def estimators(Xf, gf, gid_seen, cell, fold):
    N = Xf.shape[0]
    lw = S64.LW(Xf)
    st, u, cnt = S64.group_stats(Xf, gf, lw.P)
    s_logo, _ = logo_s(Xf, gf, lw, cell, fold)
    ug, cg = np.unique(gid_seen, return_counts=True)
    n_of = dict(zip(u, cnt))
    keep = np.array([x in n_of for x in ug])
    ug, cg = ug[keep], cg[keep]
    n_g = np.array([n_of[x] for x in ug], dtype=float)
    dq = S64.dq_pred_groups(n_g, cg / cg.sum(), st["rho_w"], s_logo, N, lw.shrinkage)
    return dict(rho_w=st["rho_w"], rho_raw=st["rho_raw"], rho_anova=st["rho_anova"], s_logo=s_logo, dq_pred=dq), lw, st, u, cnt


def run_cell(cell):
    out = RAW / f"a3_{cell}.json"
    if out.exists() and out.stat().st_size > 0:
        print("done", cell)
        return
    t0 = time.time()
    torch = CM.set_float64_torch()
    from crossfit_ood import core as C
    from crossfit_ood.core import make_group_folds
    reg = REG[cell]
    seed = int(reg["seed"])
    d = CM.load_cell(cell)
    ytr_all = d.get("labels_train")
    xtr, gtr, xid, gid, xood, fm = CM.paper2fold_inputs(d, seed)
    tf, idf = C._apply_map(fm[0], gtr), C._apply_map(fm[0], gid)
    Xtr = S64.to_t(S64.l2n(xtr))
    del xtr, xid, xood, d
    a1 = json.load(open(RAW / f"a1_{cell}.json"))
    folds = []
    for f in (0, 1):
        fit = tf == f
        Xf = Xtr[torch.as_tensor(fit, device=S64.DEV)]
        gf = gtr[fit]
        gid_seen = gid[idf == f]
        full, lw, st, u, cnt = estimators(Xf, gf, gid_seen, cell, f)
        N, dim = Xf.shape
        G = len(u)
        if G <= 40:
            blocks, nb, scheme = np.searchsorted(u, gf), G, "delete-a-group"
        else:
            s = int(CM.seed_seq("A", dict(cell=cell, fold=f, what="logo_folds")).generate_state(1)[0])
            blocks, nb, scheme = make_group_folds(gf, n_splits=10, random_state=s), 10, "delete-a-block (10 group folds)"
        reps = []
        for b in range(nb):
            keep = blocks != b
            gk = gf[keep]
            if len(np.unique(gk)) < 2:
                continue
            est, *_ = estimators(Xf[torch.as_tensor(keep, device=S64.DEV)], gk, gid_seen[np.isin(gid_seen, gk)], cell, f)
            reps.append(est)
        se = {}
        for k in full:
            th = np.array([r[k] for r in reps])
            g = len(th)
            se[k] = float(np.sqrt((g - 1) / g * np.sum((th - th.mean()) ** 2)))
        a1f = a1["folds"][f]
        n_bar, n_w = N / G, float((cnt.astype(float) ** 2).sum() / N)
        rec = dict(cell_id=cell, fold=f, N=N, G=G, d=dim, jackknife=scheme, n_jackknife=len(reps),
                   **{k: full[k] for k in full}, **{f"se_{k}": v for k, v in se.items()},
                   n_w_over_n_bar=n_w / n_bar, dq_pred_nw=a1f["dq_pred_nw"], dq_pred_nbar=a1f["dq_pred_nbar"],
                   dq_pred_without_conversion=full["dq_pred"] * (1 - lw.shrinkage), conversion_factor=1 / (1 - lw.shrinkage),
                   s_logo_over_s_tr=a1f["s_logo_over_s_tr"], lw_delta=lw.shrinkage, r_sat=a1f["r_sat"],
                   flag_r_sat_ge_0_9=bool(a1f["r_sat"] >= 0.9), flag_delta_ge_0_5=bool(lw.shrinkage >= 0.5), flag_N_lt_5d=bool(N < 5 * dim))
        if ytr_all is not None:
            y = np.asarray(ytr_all)[fit]
            Xr = Xf.clone()
            for c in np.unique(y):
                m = torch.as_tensor(y == c, device=S64.DEV)
                Xr[m] -= Xr[m].mean(0)
            lwc = S64.LW(Xr)
            stc, uc, cntc = S64.group_stats(Xr, gf, lwc.P)
            s_c, _ = logo_s(Xr, gf, lwc, cell, f)
            ug, cg = np.unique(gid_seen, return_counts=True)
            n_of = dict(zip(uc, cntc))
            n_g = np.array([n_of[x] for x in ug], dtype=float)
            rec.update(classcond_rho_w=stc["rho_w"], classcond_s_logo=s_c,
                       classcond_dq_pred=S64.dq_pred_groups(n_g, cg / cg.sum(), stc["rho_w"], s_c, N, lwc.shrinkage))
        else:
            rec.update(classcond_rho_w="NA (no labels_train)", classcond_s_logo="NA", classcond_dq_pred="NA")
        folds.append(rec)
        del Xf
        torch.cuda.empty_cache()
    CM.atomic_write_text(out, json.dumps(dict(cell_id=cell, folds=folds, seconds=time.time() - t0), indent=1))
    print(cell, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run_cell(c)
