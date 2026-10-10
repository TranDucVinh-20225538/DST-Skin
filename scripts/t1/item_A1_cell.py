#!/usr/bin/env python3
"""T1 item A1 (per cell): fit-set descriptors, s_logo, P_dQ, P_Delta and measured quantities per paper_2fold fold.

    item_A1_cell.py CELL [CELL ...]  -> results/t1/A/raw/a1_<cell>.json  (skips cells already done)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import scorer64 as S64  # noqa: E402

RAW = CM.RES / "A" / "raw"
REG = {r["cell_id"]: r for r in csv.DictReader(open(CM.RES / "I" / "registry.csv"))}
R3 = {(r["cell"], r["scorer"]): r for r in csv.DictReader(open(CM.REPO / "results/r3/1/paper_ci_v2.csv"))
      if r["seen"] == "strict"}


def logo_s(X, g, lw_full, cell, fold):
    """s_logo = (1 - delta_full) * mean over left-out-group points of Q under the refit without that group.
    G refits if G <= 40, else K = 10 group folds (package make_group_folds, seed from SeedSequence)."""
    from crossfit_ood.core import make_group_folds
    u = np.unique(g)
    if len(u) <= 40:
        blocks = np.searchsorted(u, g)
        nb = len(u)
        scheme = f"LOGO ({nb} refits)"
    else:
        seed = int(CM.seed_seq("A", dict(cell=cell, fold=fold, what="logo_folds")).generate_state(1)[0])
        blocks = make_group_folds(g, n_splits=10, random_state=seed)
        nb = 10
        scheme = f"K=10 group folds (seed {seed})"
    qsum, qn = 0.0, 0
    for b in range(nb):
        out = blocks == b
        if not out.any():
            continue
        out_t = S64.torch.as_tensor(out, device=X.device)
        lw = S64.LW(X[~out_t])
        q = lw.q(X[out_t])
        qsum += float(q.sum())
        qn += int(q.numel())
        del lw
    return (1 - lw_full.shrinkage) * qsum / qn, scheme


def run_cell(cell):
    out = RAW / f"a1_{cell}.json"
    if out.exists() and out.stat().st_size > 0:
        print("done", cell)
        return
    t0 = time.time()
    torch = CM.set_float64_torch()
    reg = REG[cell]
    seed = int(reg["seed"])
    d = CM.load_cell(cell)
    xtr, gtr, xid, gid, xood, fm = CM.paper2fold_inputs(d, seed)
    from crossfit_ood import core as C
    tf, idf = C._apply_map(fm[0], gtr), C._apply_map(fm[0], gid)
    Xtr, Xid, Xood = S64.to_t(S64.l2n(xtr)), S64.to_t(S64.l2n(xid)), S64.to_t(S64.l2n(xood))
    del xtr, xid, xood, d
    folds = []
    for f in (0, 1):
        fit = tf == f
        Xf = Xtr[S64.torch.as_tensor(fit, device=S64.DEV)]
        gf = gtr[fit]
        N, dim = Xf.shape
        lw = S64.LW(Xf)
        delta, mu = lw.shrinkage, lw.mu
        st, u, cnt = S64.group_stats(Xf, gf, lw.P)
        G = st["G"]
        n_bar = N / G
        n_w = float((cnt.astype(float) ** 2).sum() / N)
        s_logo, scheme = logo_s(Xf, gf, lw, cell, f)
        s_tr = lw.s_tr()
        q_id = lw.q(Xid).cpu().numpy()
        q_ood = lw.q(Xood).cpu().numpy()
        seen = idf == f
        unseen = (idf >= 0) & (idf != f)
        q_s, q_u = q_id[seen], q_id[unseen]
        a_seen = S64.auroc_id_pos(-q_s, -q_ood)
        a_unseen = S64.auroc_id_pos(-q_u, -q_ood)
        dq_meas = float(q_s.mean() - q_u.mean())
        s_unseen = (1 - delta) * float(q_u.mean())
        # pi_g over seen-eval points; n_g = fit points of that group
        gs = gid[seen]
        ug, cg = np.unique(gs, return_counts=True)
        n_of = dict(zip(u, cnt))
        n_g = np.array([n_of[x] for x in ug], dtype=float)
        pi_g = cg / cg.sum()
        rho = st["rho_w"]
        dq = S64.dq_pred_groups(n_g, pi_g, rho, s_logo, N, delta)
        variants = dict(
            dq_pred_nw=S64.dq_pred_plugin(n_w, rho, s_logo, N, delta),
            dq_pred_nbar=S64.dq_pred_plugin(n_bar, rho, s_logo, N, delta),
            dq_pred_str=S64.dq_pred_groups(n_g, pi_g, rho, s_tr, N, delta),
            dq_pred_rhoraw=S64.dq_pred_groups(n_g, pi_g, st["rho_raw"], s_logo, N, delta),
            dq_pred_rhoanova=S64.dq_pred_groups(n_g, pi_g, st["rho_anova"], s_logo, N, delta))
        dq_norm = dq * (1 - delta)
        r_sat = abs(dq_norm) / (rho * s_logo) if rho * s_logo != 0 else float("nan")
        shift = abs(dq)
        p_delta = S64.auroc_id_pos(-(q_u - shift), -q_ood) - a_unseen
        s_scorer = s_logo / (1 - delta)
        p_delta_mult = S64.auroc_id_pos(-(q_u * (1 + dq / s_scorer)), -q_ood) - a_unseen
        sd = np.sqrt(q_ood.var(ddof=1) + q_u.var(ddof=1))
        p_delta_gauss = float(norm.cdf((q_ood.mean() - (q_u.mean() + dq)) / sd) - norm.cdf((q_ood.mean() - q_u.mean()) / sd))
        folds.append(dict(
            cell_id=cell, fold=f, N=N, G=G, d=dim, n_bar=n_bar, n_w=n_w, gsize_min=int(cnt.min()), gsize_median=float(np.median(cnt)),
            gsize_max=int(cnt.max()), id_acc=reg["id_acc"], lw_delta=delta, lw_mu=mu, **{k: st[k] for k in ("rho_w", "rho_raw", "rho_anova", "icc_whitened_r3", "n0")},
            s_logo=s_logo, s_logo_scheme=scheme, s_tr=s_tr, s_logo_over_s_tr=s_logo / s_tr, s_unseen=s_unseen,
            r_sat=r_sat, dq_pred=dq, **variants, dq_meas=dq_meas,
            auroc_seen=a_seen, auroc_unseen=a_unseen, delta_meas_fold=a_seen - a_unseen,
            p_delta=p_delta, p_delta_mult=p_delta_mult, p_delta_gauss=p_delta_gauss,
            n_seen=int(seen.sum()), n_unseen=int(unseen.sum()), n_ood=int(len(q_ood)), n_seen_groups=len(ug),
            d_over_N=dim / N))
        del Xf, lw
        S64.torch.cuda.empty_cache()
    r3 = R3.get((cell, "mahalanobis_l2"))
    delta_own = float(np.mean([x["auroc_seen"] for x in folds]) - np.mean([x["auroc_unseen"] for x in folds]))
    res = dict(cell_id=cell, dataset=reg["dataset"], backbone=reg["backbone"], family=reg["family"], seed=seed,
               scorer="mahalanobis_l2 (Track A: LW on L2-normalised float64 features, class-agnostic)",
               delta_meas=float(r3["delta"]) if r3 else None, delta_own=delta_own,
               abs_diff_own_vs_r3=abs(delta_own - float(r3["delta"])) if r3 else None,
               p_delta=float(np.mean([x["p_delta"] for x in folds])), folds=folds, seconds=time.time() - t0,
               precommit=CM.precommit_hash())
    CM.atomic_write_text(out, json.dumps(res, indent=1))
    print(cell, f"{time.time() - t0:.0f}s", "delta_own", delta_own, "r3", res["delta_meas"], "p_delta", res["p_delta"], flush=True)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for c in sys.argv[1:]:
        run_cell(c)
