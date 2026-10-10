#!/usr/bin/env python3
"""T1 item H1: null controls H-N1 (group-free Gaussian twin), H-N2 (one image per group), H-N3 (identical split).

    hn1 / hn2 / hn3 are called by item_H_run.py (units hn1|<cell>, hn3|<cell>, hn2|<cell>)
Scorer: Track A mahalanobis_l2 (LW on L2-normalised float64 features, scorer64). Delta CIs: refit-aware weighted
delete-a-block jackknife of crossfit-ood (Busing weights; package _blocks / _jk_var_weighted; <= 50 ID blocks of groups,
<= 50 OOD blocks of samples), V = V_id + V_ood, CI = Delta +- 1.96 sqrt(V); block rng default_rng(SeedSequence of the
unit, H item code 8). Per-fold Deltas are independent (H-N1, H-N3): Delta = mean of the two folds, V = (V_0 + V_1) / 4.
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
NB = 50


def jk_var(theta, th_del, m):
    from crossfit_ood.core import _jk_var_weighted
    return float(_jk_var_weighted(np.array([theta]), np.asarray(th_del, float)[:, None], np.asarray(m, float))[0])


def blocks(units, rng):
    from crossfit_ood.core import _blocks
    return _blocks(np.asarray(units), NB, rng)


def auc(q_id, q_ood):
    return S64.auroc_id_pos(-np.asarray(q_id), -np.asarray(q_ood))


def two_arm_jk(Xfit, gfit, XA, gA, XB, gB, Xood, rng):
    """theta = AUROC(A vs OOD) - AUROC(B vs OOD) under one LW model fitted on (Xfit, gfit). Jackknife: ID units = all
    groups of fit, A and B (a deleted group loses its fit and eval samples; refit); OOD blocks of samples."""
    def theta_of(kf, ka, kb):
        lw = S64.LW(Xfit[torch.as_tensor(kf, device=S64.DEV)])
        qa = lw.q(XA[torch.as_tensor(ka, device=S64.DEV)]).cpu().numpy()
        qb = lw.q(XB[torch.as_tensor(kb, device=S64.DEV)]).cpu().numpy()
        qo = lw.q(Xood).cpu().numpy()
        return qa, qb, qo
    allk = (np.ones(len(gfit), bool), np.ones(len(gA), bool), np.ones(len(gB), bool))
    qa, qb, qo = theta_of(*allk)
    a_a, a_b = auc(qa, qo), auc(qb, qo)
    th = a_a - a_b
    units = np.unique(np.concatenate([gfit, gA, gB]))
    idb = blocks(units, rng)
    th_id = []
    for b in idb:
        ka, kb, kf = ~np.isin(gA, b), ~np.isin(gB, b), ~np.isin(gfit, b)
        if ka.sum() == 0 or kb.sum() == 0:
            th_id.append(th)
            continue
        qa2, qb2, qo2 = theta_of(kf, ka, kb)
        th_id.append(auc(qa2, qo2) - auc(qb2, qo2))
    oob = blocks(np.arange(len(qo)), rng)
    th_ood = []
    for b in oob:
        keep = np.ones(len(qo), bool)
        keep[b] = False
        th_ood.append(auc(qa, qo[keep]) - auc(qb, qo[keep]))
    v = jk_var(th, th_id, [len(b) for b in idb]) + jk_var(th, th_ood, [len(b) for b in oob])
    return dict(theta=th, auroc_A=a_a, auroc_B=a_b, var=v, n_id_blocks=len(idb), n_ood_blocks=len(oob))


def load(cell):
    reg = REG[cell]
    d = CM.load_cell(cell)
    xtr, gtr, xid, gid, xood, fm = CM.paper2fold_inputs(d, int(reg["seed"]))
    from crossfit_ood import core as C
    tf, idf = C._apply_map(fm[0], gtr), C._apply_map(fm[0], gid)
    return reg, xtr, gtr, xid, gid, xood, tf, idf


def hn1(uid, cell):
    """Group-free Gaussian twin per fold: fit / seen / unseen ~ N(mu_f, Sigma_LW,f) of the real normalised fit set,
    real group labels and sizes, rho = 0; OOD = real OOD standardised per coordinate to the fit moments; full A1
    per-fold pipeline (rho_w, s_logo, dQ_pred, dQ_meas, Delta_meas) on the l2-normalised simulated data."""
    reg, xtr, gtr, xid, gid, xood, tf, idf = load(cell)
    Xtr, Xid, Xood = S64.to_t(S64.l2n(xtr)), S64.to_t(S64.l2n(xid)), S64.to_t(S64.l2n(xood))
    del xtr, xid, xood
    folds = []
    for f in (0, 1):
        ss = CM.seed_seq("H", dict(unit=uid, fold=f), 0)
        gen = torch.Generator(device=S64.DEV).manual_seed(int(ss.generate_state(1, np.uint64)[0] & np.uint64(2 ** 63 - 1)))
        rng = np.random.default_rng(ss)
        fit = tf == f
        Xf = Xtr[torch.as_tensor(fit, device=S64.DEV)]
        gf = gtr[fit]
        lw0 = S64.LW(Xf)
        dim = Xf.shape[1]
        cov = (1 - lw0.shrinkage) * lw0.S + lw0.shrinkage * lw0.mu * torch.eye(dim, dtype=torch.float64, device=S64.DEV)
        Lc = torch.linalg.cholesky(cov)
        mu = lw0.loc
        sd_f = Xf.std(0)
        seen, unseen = idf == f, (idf >= 0) & (idf != f)

        def sim(n):
            return S64.to_t(S64.l2n((mu + torch.randn(n, dim, generator=gen, dtype=torch.float64, device=S64.DEV) @ Lc.T).cpu().numpy()))
        Sf, Ss, Su = sim(len(gf)), sim(int(seen.sum())), sim(int(unseen.sum()))
        Xo = (Xood - Xood.mean(0)) / Xood.std(0).clamp_min(1e-12) * sd_f + mu
        So = S64.to_t(S64.l2n(Xo.cpu().numpy()))
        del Xf, lw0, cov, Lc, Xo
        lw = S64.LW(Sf)
        st, u, cnt = S64.group_stats(Sf, gf, lw.P)
        s_logo, scheme = logo_s(Sf, gf, lw, uid, f)
        q_s, q_u, q_o = lw.q(Ss).cpu().numpy(), lw.q(Su).cpu().numpy(), lw.q(So).cpu().numpy()
        gs = gid[seen]
        ug, cg = np.unique(gs, return_counts=True)
        n_of = dict(zip(u, cnt))
        n_g = np.array([n_of[x] for x in ug], float)
        dq = S64.dq_pred_groups(n_g, cg / cg.sum(), st["rho_w"], s_logo, len(gf), lw.shrinkage)
        jk = two_arm_jk(Sf, gf, Ss, gs, Su, gid[unseen], So, rng)
        folds.append(dict(fold=f, N=len(gf), d=dim, rho_w=st["rho_w"], rho_raw=st["rho_raw"], rho_anova=st["rho_anova"],
                          s_logo=s_logo, s_logo_scheme=scheme, dq_pred=dq, dq_meas=float(q_s.mean() - q_u.mean()),
                          delta_meas=jk["theta"], var=jk["var"], auroc_seen=jk["auroc_A"], auroc_unseen=jk["auroc_B"]))
        del Sf, Ss, Su, So, lw
        torch.cuda.empty_cache()
    return folds


def hn3(uid, cell):
    """Identical-split control on the real cell: per fold, the unseen-eval groups are split at random into two halves
    (A, B) and Delta_f = AUROC(A vs OOD) - AUROC(B vs OOD) under the fold-f model."""
    reg, xtr, gtr, xid, gid, xood, tf, idf = load(cell)
    Xtr, Xid, Xood = S64.to_t(S64.l2n(xtr)), S64.to_t(S64.l2n(xid)), S64.to_t(S64.l2n(xood))
    del xtr, xid, xood
    folds = []
    for f in (0, 1):
        rng = np.random.default_rng(CM.seed_seq("H", dict(unit=uid, fold=f), 0))
        fit = tf == f
        unseen = np.flatnonzero((idf >= 0) & (idf != f))
        ug = np.unique(gid[unseen])
        if len(ug) < 2:
            folds.append(dict(fold=f, status="not evaluable: < 2 unseen groups"))
            continue
        half = set(rng.permutation(ug)[:len(ug) // 2].tolist())
        inA = np.array([g in half for g in gid[unseen]])
        A, B = unseen[inA], unseen[~inA]
        jk = two_arm_jk(Xtr[torch.as_tensor(fit, device=S64.DEV)], gtr[fit], Xid[torch.as_tensor(A, device=S64.DEV)], gid[A],
                        Xid[torch.as_tensor(B, device=S64.DEV)], gid[B], Xood, rng)
        folds.append(dict(fold=f, n_A=len(A), n_B=len(B), n_groups_A=len(half), n_groups_B=len(ug) - len(half),
                          delta_meas=jk["theta"], var=jk["var"], auroc_A=jk["auroc_A"], auroc_B=jk["auroc_B"]))
        torch.cuda.empty_cache()
    return folds


def hn2(uid, cell):
    """One image per group (pooled train + ID eval, one image per group, default_rng of the unit): a random sample split
    (fit / eval halves) vs a group split (package _random_fold_map over the groups); Delta = AUROC_random - AUROC_group,
    each arm = AUROC(eval half vs OOD | fit on the other half), averaged over the two halves' roles."""
    from crossfit_ood import core as C
    reg, xtr, gtr, xid, gid, xood, tf, idf = load(cell)
    rng = np.random.default_rng(CM.seed_seq("H", dict(unit=uid), 0))
    X = np.concatenate([xtr, xid])
    g = np.concatenate([gtr.astype(str), gid.astype(str)])
    perm = rng.permutation(len(g))
    _, first = np.unique(g[perm], return_index=True)
    keep = np.sort(perm[first])
    X, g = S64.to_t(S64.l2n(X[keep])), g[keep]
    Xo = S64.to_t(S64.l2n(xood))
    del xtr, xid, xood
    n = len(g)
    r_split = np.zeros(n, int)
    r_split[rng.permutation(n)[: n // 2]] = 1
    gmap = C._random_fold_map(np.unique(g), 2, int(rng.integers(2 ** 31)))
    g_split = C._apply_map(gmap, g)
    res = []
    for name, sp in (("random", r_split), ("group", g_split)):
        for f in (0, 1):
            res.append((name, f, sp))
    # theta = mean_f AUROC_random(f) - mean_f AUROC_group(f); jackknife over groups (= images) and OOD samples
    def theta(keep_id, keep_ood):
        out = {}
        for name, sp in (("random", r_split), ("group", g_split)):
            a = []
            for f in (0, 1):
                fit, ev = (sp == f) & keep_id, (sp != f) & keep_id
                lw = S64.LW(X[torch.as_tensor(fit, device=S64.DEV)])
                a.append(auc(lw.q(X[torch.as_tensor(ev, device=S64.DEV)]).cpu().numpy(),
                             lw.q(Xo[torch.as_tensor(keep_ood, device=S64.DEV)]).cpu().numpy()))
            out[name] = float(np.mean(a))
        return out
    full = theta(np.ones(n, bool), np.ones(len(Xo), bool))
    th = full["random"] - full["group"]
    idb = blocks(np.unique(g), rng)
    th_id = []
    for b in idb:
        t = theta(~np.isin(g, b), np.ones(len(Xo), bool))
        th_id.append(t["random"] - t["group"])
    oob = blocks(np.arange(len(Xo)), rng)
    th_ood = []
    for b in oob:
        k = np.ones(len(Xo), bool)
        k[b] = False
        t = theta(np.ones(n, bool), k)
        th_ood.append(t["random"] - t["group"])
    v = jk_var(th, th_id, [len(b) for b in idb]) + jk_var(th, th_ood, [len(b) for b in oob])
    torch.cuda.empty_cache()
    return [dict(fold="pooled", n_images=n, auroc_random=full["random"], auroc_group=full["group"], delta_meas=th, var=v)]


def summarise(folds):
    ok = [f for f in folds if "delta_meas" in f]
    if len(ok) == 0:
        return dict(status="not evaluable")
    d = float(np.mean([f["delta_meas"] for f in ok]))
    se = float(np.sqrt(sum(f["var"] for f in ok)) / len(ok))
    lo, hi = d - 1.96 * se, d + 1.96 * se
    return dict(status="ok", delta=d, se=se, ci_lo=lo, ci_hi=hi, flag_s9=bool(abs(d) > 0.02 and (lo > 0 or hi < 0)))


def units():
    cells = sorted(REG)
    big = sorted({c for c in cells if REG[c]["dataset"] in
                  {ds for ds in {REG[x]["dataset"] for x in cells}
                   if min(int(REG[x]["n_groups_fit_f0"]) + int(REG[x]["n_groups_fit_f1"]) for x in cells if REG[x]["dataset"] == ds) >= 500}})
    return [f"hn1|{c}" for c in cells] + [f"hn3|{c}" for c in cells] + [f"hn2|{c}" for c in big]
