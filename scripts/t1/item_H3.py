#!/usr/bin/env python3
"""T1 item H3: coverage of the group-jackknife CIs of rho_w and dQ_pred (Track A estimators of A1 / A3) in toy data
with real-matched designs. x = sqrt(rho) u_g + sqrt(1 - rho) e (isotropic, d dims), L2-normalised as in Track A;
1,000 datasets per design cell; truth = the simulated rho and dQ_true = mean Q(seen) - mean Q(unseen) over 50,000 fresh
draws per side for the realised fit set (seen: group g with probability pi_g = n_g / N, its realised u_g; unseen: new u).

Estimators exactly as A1 (LW on the normalised fit set, rho_w with the LW precision, s_logo = LOGO over groups if G <= 40,
else over K = 10 group folds, dQ_pred = K2 per-group closed form with pi_g = n_g / N). Jackknife units as A3 (groups if
G <= 40, else the same 10 group folds; nested LOGO over the remaining units), Busing-weighted delete-m_j variance
(m_j = groups in unit j), CI = estimate +- 1.96 SE. Every refit is computed from per-unit sufficient statistics
(n, sum x, sum x x', sum_g s_g s_g'/n_g, sum ||x||^2, sum ||x||^4, sum ||x||^2 x), which gives the same LW fit, rho_w and
LOGO Q sums as refitting on the retained rows.
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

DEV = S64.DEV
NDATA = int(os.environ.get("T1_H3_NDATA", "1000"))
NFRESH = 50000


def design_ng(p):
    from crossfit_ood import core as C
    reg = {r["cell_id"]: r for r in csv.DictReader(open(CM.RES / "I" / "registry.csv"))}[p["n_g_from"]]
    d = CM.load_cell(p["n_g_from"], ["groups_train", "groups_id_eval", "fold_train"])
    gtr, gid = np.asarray(d["groups_train"]), np.asarray(d["groups_id_eval"])
    fm = C._fold_ids_to_maps(np.asarray(d["fold_train"]).astype(int), gtr) if "fold_train" in d else \
        C._new_fold_maps("paper_2fold", gtr, gid[np.isin(gid, gtr)], 2, 1, int(reg["seed"]))
    tf = C._apply_map(fm[0], gtr)
    _, cnt = np.unique(gtr[tf == p["fold"]], return_counts=True)
    return cnt.astype(int)


class Stats:
    """Per-unit sufficient statistics of a normalised fit set X (rows) with group codes g and unit codes un."""

    def __init__(self, X, g, un, nu):
        d = X.shape[1]
        self.nu, self.d = nu, d
        gt = torch.as_tensor(g, device=DEV)
        ut = torch.as_tensor(un, device=DEV)
        G = int(g.max()) + 1
        ng = torch.bincount(gt, minlength=G).double()
        sg = torch.zeros(G, d, dtype=torch.float64, device=DEV).index_add_(0, gt, X)
        a = (X * X).sum(1)
        self.n = torch.bincount(ut, minlength=nu).double()
        self.s = torch.zeros(nu, d, dtype=torch.float64, device=DEV).index_add_(0, ut, X)
        self.A1 = torch.zeros(nu, dtype=torch.float64, device=DEV).index_add_(0, ut, a)
        self.A2 = torch.zeros(nu, dtype=torch.float64, device=DEV).index_add_(0, ut, a * a)
        self.Ax = torch.zeros(nu, d, dtype=torch.float64, device=DEV).index_add_(0, ut, a[:, None] * X)
        unit_of_g = torch.zeros(G, dtype=torch.long, device=DEV)
        unit_of_g[gt] = ut
        self.S = torch.empty(nu, d, d, dtype=torch.float64, device=DEV)
        self.B = torch.empty(nu, d, d, dtype=torch.float64, device=DEV)
        self.Gu = torch.bincount(unit_of_g, minlength=nu).double()
        for u in range(nu):
            Xu = X[ut == u]
            self.S[u] = Xu.T @ Xu
            Mu = sg[unit_of_g == u] / torch.sqrt(ng[unit_of_g == u])[:, None]
            self.B[u] = Mu.T @ Mu

    def fit(self, w):
        """LW fit and rho_w on the units with weight w (0/1 tensor (nu,) or (k, nu) batch). Returns dict of batched tensors."""
        W = w if w.dim() == 2 else w[None]
        N = W @ self.n
        s = W @ self.s
        m = s / N[:, None]
        S_sum = torch.einsum("ku,uij->kij", W, self.S)
        Sn = S_sum / N[:, None, None] - m[:, :, None] * m[:, None, :]
        d = self.d
        emp_tr = torch.diagonal(Sn, dim1=1, dim2=2)
        mu = emp_tr.sum(1) / d
        delta_ = (Sn * Sn).sum((1, 2))
        c = (m * m).sum(1)
        beta_ = (W @ self.A2) + 4 * torch.einsum("ki,kij,kj->k", m, S_sum, m) + N * c * c \
            - 4 * (m * (W @ self.Ax)).sum(1) + 2 * c * (W @ self.A1) - 4 * c * (m * s).sum(1)
        beta = 1.0 / (d * N) * (beta_ / N - delta_)
        delta = (delta_ - 2.0 * mu * emp_tr.sum(1) + d * mu ** 2) / d
        beta = torch.minimum(beta, delta)
        shrink = torch.where(beta == 0, torch.zeros_like(beta), beta / delta)
        eye = torch.eye(d, dtype=torch.float64, device=DEV)
        P = torch.linalg.inv((1 - shrink)[:, None, None] * Sn + (shrink * mu)[:, None, None] * eye)
        Gk = W @ self.Gu
        SST = S_sum - N[:, None, None] * m[:, :, None] * m[:, None, :]
        SSB = torch.einsum("ku,uij->kij", W, self.B) - N[:, None, None] * m[:, :, None] * m[:, None, :]
        Wm = (SST - SSB) / (N - Gk)[:, None, None]
        Tm = SST / (N - 1)[:, None, None]
        rho_w = 1.0 - (P * Wm).sum((1, 2)) / (P * Tm).sum((1, 2))
        return dict(N=N, loc=m, P=P, shrink=shrink, rho_w=rho_w)

    def qsum(self, fit, u):
        """Sum of Q over the rows of unit u under each fit in the batch."""
        P, loc = fit["P"], fit["loc"]
        return (torch.einsum("kij,ij->k", P, self.S[u]) - 2 * torch.einsum("ki,kij,j->k", loc, P, self.s[u])
                + self.n[u] * torch.einsum("ki,kij,kj->k", loc, P, loc))

    def estimate(self, keep):
        """rho_w, s_logo, delta of the fit on the units in `keep` (list); s_logo by LOGO over those units (batched)."""
        w = torch.zeros(self.nu, dtype=torch.float64, device=DEV)
        w[keep] = 1
        full = self.fit(w)
        Wl = w.repeat(len(keep), 1)
        for i, u in enumerate(keep):
            Wl[i, u] = 0
        lo = self.fit(Wl)
        qs = torch.stack([self.qsum({"P": lo["P"][i:i + 1], "loc": lo["loc"][i:i + 1]}, u)[0] for i, u in enumerate(keep)])
        s_logo = (1 - full["shrink"][0]) * qs.sum() / self.n[keep].sum()
        return float(full["rho_w"][0]), float(s_logo), float(full["shrink"][0]), full


def _dq(n_g, pi, rho, s, N, delta):
    """dQ_pred; NaN when the LW shrinkage is 1 (the 1/(1 - delta) factor is undefined)."""
    return float("nan") if delta >= 1 else S64.dq_pred_groups(n_g, pi, rho, s, N, delta)


def one_dataset(p, n_g, seed_ss, unit_of_group):
    gen = torch.Generator(device=DEV).manual_seed(int(seed_ss.generate_state(1, np.uint64)[0] & np.uint64(2 ** 63 - 1)))
    d, rho = p["d"], p["rho"]
    G = len(n_g)
    g = np.repeat(np.arange(G), n_g)
    U = torch.randn(G, d, generator=gen, dtype=torch.float64, device=DEV)
    gt = torch.as_tensor(g, device=DEV)
    X = np.sqrt(rho) * U[gt] + np.sqrt(1 - rho) * torch.randn(len(g), d, generator=gen, dtype=torch.float64, device=DEV)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    un = unit_of_group[g]
    nu = int(unit_of_group.max()) + 1
    st = Stats(X, g, un, nu)
    Gm = st.Gu.cpu().numpy()
    N = len(g)
    pi = n_g / N
    rho_w, s_logo, delta, full = st.estimate(list(range(nu)))
    dq = _dq(n_g, pi, rho_w, s_logo, N, delta)
    reps = []
    for j in range(nu):
        keep = [u for u in range(nu) if u != j]
        r2, s2, d2, _ = st.estimate(keep)
        kg = unit_of_group != j
        reps.append((r2, _dq(n_g[kg], n_g[kg] / n_g[kg].sum(), r2, s2, int(n_g[kg].sum()), d2)))
    from crossfit_ood.core import _jk_var_weighted
    th = np.array([[rho_w, dq]])
    v = _jk_var_weighted(th, np.array(reps), Gm)
    se = np.sqrt(v)
    # truth: dQ_true for the realised fit (P, loc of the full fit)
    P, loc = full["P"][0], full["loc"][0]
    gs = torch.as_tensor(np.random.default_rng(seed_ss.spawn(1)[0]).choice(G, NFRESH, p=pi), device=DEV)
    Xs = np.sqrt(rho) * U[gs] + np.sqrt(1 - rho) * torch.randn(NFRESH, d, generator=gen, dtype=torch.float64, device=DEV)
    Xu = np.sqrt(rho) * torch.randn(NFRESH, d, generator=gen, dtype=torch.float64, device=DEV) + \
        np.sqrt(1 - rho) * torch.randn(NFRESH, d, generator=gen, dtype=torch.float64, device=DEV)
    q = lambda Z: (((Z / (Z.norm(dim=1, keepdim=True) + 1e-8) - loc) @ P) * (Z / (Z.norm(dim=1, keepdim=True) + 1e-8) - loc)).sum(1)  # noqa: E731
    dq_true = float(q(Xs).mean() - q(Xu).mean())
    del X, st, U, Xs, Xu
    return dict(rho_w=rho_w, se_rho_w=float(se[0]), dq_pred=dq, se_dq=float(se[1]), dq_true=dq_true,
                cover_rho=bool(abs(rho_w - rho) <= 1.96 * se[0]), cover_dq=bool(abs(dq - dq_true) <= 1.96 * se[1]), dq_undefined=bool(not np.isfinite(dq) or not np.isfinite(se[1])))


def run_unit(uid, p, deadline=None):
    from crossfit_ood.core import make_group_folds
    t0 = time.time()
    n_g = design_ng(p)
    G = len(n_g)
    if G <= 40:
        unit_of_group, scheme = np.arange(G), f"delete-a-group ({G})"
    else:
        s = int(CM.seed_seq("H", dict(unit=uid, what="folds")).generate_state(1)[0])
        unit_of_group = make_group_folds(np.arange(G), n_splits=10, random_state=s)
        scheme = "delete-a-block (10 group folds)"
    rows = []
    for k in range(NDATA):
        if deadline is not None and time.time() > deadline:
            break
        rows.append(one_dataset(p, n_g, CM.seed_seq("H", uid, k), unit_of_group))
        if k % 100 == 0:
            torch.cuda.empty_cache()
    cr = np.mean([r["cover_rho"] for r in rows]) if rows else float("nan")
    cd = np.mean([r["cover_dq"] for r in rows]) if rows else float("nan")
    return dict(G=G, N=int(n_g.sum()), d=p["d"], rho=p["rho"], jackknife=scheme, n_datasets=len(rows),
                coverage_rho_w=float(cr), coverage_dq_pred=float(cd),
                se_coverage_rho_w=float(np.sqrt(cr * (1 - cr) / len(rows))), se_coverage_dq_pred=float(np.sqrt(cd * (1 - cd) / len(rows))),
                mean_rho_w=float(np.mean([r["rho_w"] for r in rows])), mean_dq_pred=float(np.nanmean([r["dq_pred"] for r in rows])),
                mean_dq_true=float(np.mean([r["dq_true"] for r in rows])),
                n_dq_undefined=int(sum(r["dq_undefined"] for r in rows)),
                median_se_rho_w=float(np.median([r["se_rho_w"] for r in rows])), median_se_dq=float(np.nanmedian([r["se_dq"] for r in rows])),
                pass_093=bool(cr >= 0.93 and cd >= 0.93) if len(rows) == NDATA else None,
                status="ok" if len(rows) == NDATA else f"not run: H cap reached ({len(rows)}/{NDATA} datasets)",
                seconds=time.time() - t0)
