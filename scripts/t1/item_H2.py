#!/usr/bin/env python3
"""T1 item H2: placebo descriptors for item A. The A1 predictor (rho_w, s_logo, dQ_pred, P_Delta per paper_2fold fold)
is recomputed with permuted fit-group labels: Camelyon within hospital {0, 3, 4} (results/t1/I/camelyon_manifest.csv),
other cells within the cell; 20 permutations per cell from default_rng(801) spawned per cell. The label vector is
permuted, so group sizes and the seen-eval weights pi_g are unchanged; Delta_meas is the real one.

    run_cell is called by item_H_run.py (units h2|<cell>)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import scorer64 as S64  # noqa: E402
from item_A1_cell import logo_s  # noqa: E402

RAW = Path(os.environ.get("T1_RAW", str(CM.RES / "H" / "raw")))
REG = {r["cell_id"]: r for r in csv.DictReader(open(CM.RES / "I" / "registry.csv"))}
HOSP = {r["slide"]: r["hospital"] for r in csv.DictReader(open(CM.RES / "I" / "camelyon_manifest.csv"))}
NPERM = 20


def cell_rng(cell):
    cells = sorted(REG)
    return np.random.default_rng(np.random.SeedSequence(801).spawn(len(cells))[cells.index(cell)])


def run_cell(uid, cell):
    from crossfit_ood import core as C
    reg = REG[cell]
    d = CM.load_cell(cell)
    xtr, gtr, xid, gid, xood, fm = CM.paper2fold_inputs(d, int(reg["seed"]))
    tf, idf = C._apply_map(fm[0], gtr), C._apply_map(fm[0], gid)
    Xtr, Xid, Xood = S64.to_t(S64.l2n(xtr)), S64.to_t(S64.l2n(xid)), S64.to_t(S64.l2n(xood))
    del xtr, xid, xood, d
    rng = cell_rng(cell)
    a1 = json.load(open(CM.RES / "A" / "raw" / f"a1_{cell}.json"))
    folds = []
    for f in (0, 1):
        fit = tf == f
        Xf = Xtr[torch.as_tensor(fit, device=S64.DEV)]
        gf = gtr[fit]
        N = len(gf)
        lw = S64.LW(Xf)
        q_id = lw.q(Xid).cpu().numpy()
        q_ood = lw.q(Xood).cpu().numpy()
        seen, unseen = idf == f, (idf >= 0) & (idf != f)
        q_u = q_id[unseen]
        a_unseen = S64.auroc_id_pos(-q_u, -q_ood)
        strata = np.array([HOSP.get(str(g), "all") for g in gf]) if reg["dataset"] == "camelyon" else np.full(N, "all")
        if reg["dataset"] == "camelyon":
            assert not (strata == "all").any(), "Camelyon slide without hospital"
        u0, cnt0 = np.unique(gf, return_counts=True)
        n_of = dict(zip(u0, cnt0))
        ug, cg = np.unique(gid[seen], return_counts=True)
        n_g = np.array([n_of[x] for x in ug], float)
        pi_g = cg / cg.sum()
        perms = []
        for k in range(NPERM):
            gp = gf.copy()
            for s in np.unique(strata):
                idx = np.flatnonzero(strata == s)
                gp[idx] = gf[idx][rng.permutation(len(idx))]
            st, _, _ = S64.group_stats(Xf, gp, lw.P)
            s_logo, _ = logo_s(Xf, gp, lw, f"h2perm{k}|{cell}", f)
            dq = S64.dq_pred_groups(n_g, pi_g, st["rho_w"], s_logo, N, lw.shrinkage)
            p_delta = S64.auroc_id_pos(-(q_u - abs(dq)), -q_ood) - a_unseen
            perms.append(dict(perm=k, rho_w=st["rho_w"], s_logo=s_logo, dq_pred=dq, p_delta=p_delta))
        real = a1["folds"][f]
        folds.append(dict(fold=f, real_rho_w=real["rho_w"], real_dq_pred=real["dq_pred"], real_p_delta=real["p_delta"],
                          placebo_rho_w_mean=float(np.mean([p["rho_w"] for p in perms])),
                          placebo_dq_pred_mean=float(np.mean([p["dq_pred"] for p in perms])),
                          placebo_p_delta_mean=float(np.mean([p["p_delta"] for p in perms])), perms=perms))
        del Xf, lw
        torch.cuda.empty_cache()
    return dict(unit=uid, cell=cell, dataset=reg["dataset"], backbone=reg["backbone"], family=reg["family"],
                delta_meas=a1["delta_meas"], real_p_delta=a1["p_delta"],
                placebo_p_delta=float(np.mean([f["placebo_p_delta_mean"] for f in folds])), folds=folds)
