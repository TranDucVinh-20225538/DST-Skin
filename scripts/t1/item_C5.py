#!/usr/bin/env python3
"""T1 item C5 (stage 3): misspecification variants over the 1,500-config sub-designs of C1 (unit params in addendum C).

Each unit runs sim_c.run_config (Mahalanobis family only; adaptive replicates as C1) on the variant model and stores the
closed-form K2 prediction of the isotropic Gaussian model with the same (d, G, mean n, rho, lambda), plus the real-feature
diagnostics of the replicate-0 fit set for the failure-boundary regression:
  (a) heavy tails: Student-t nu on u or on e;
  (b) proportional anisotropy B = rho Sigma, spiked (r spikes of size kappa) or power law (alpha); ridge in raw coordinates
      and, as a second run on the same seeds, in whitened coordinates;
  (c) non-proportional: Sigma power law alpha = 1; u_g only in the top-k or bottom-k eigendirections (k = max(1, d // 10)),
      scaled by sqrt(lambda_j d / k) so that the whitened group signal per selected direction, and hence the accuracy of an
      optimal linear group probe, is the same for top and bottom;
  (d) K class means separated by sep (sigma units): class-agnostic (run_config) and class-conditional Mahalanobis
      (labels known; class means and the tied within-class covariance from the fit set; Q at the point's own class);
  (e) G in {6, 12, 24}, n_g ~ lognormal with CV 1 and mean n (rounded, >= 2).
"""
from __future__ import annotations

import numpy as np
import torch

import common as CM
import scorer64 as S64
import sim_c

DEV = S64.DEV


def power_law(d, alpha):
    sp = np.arange(1, d + 1, dtype=float) ** -float(alpha)
    return sp / sp.mean()


def spiked(d, r, kappa):
    sp = np.ones(d)
    sp[:int(r)] = float(kappa)
    return sp / sp.mean()


def variant_cfg(uid, p):
    d, rho, G, n = int(p["d"]), float(p["rho"]), int(p["G"]), int(p["n"])
    v = p["variant"]
    cfg = dict(d=d, rho=rho, n_g=[n] * G)
    if v == "a":
        cfg["nu_u" if p["v_target"] == "u" else "nu_e"] = float(p["v_nu"])
    elif v == "b":
        cfg["spectrum"] = (spiked(d, p["v_r"], p["v_kappa"]) if p["v_kind"] == "spiked" else power_law(d, p["v_alpha"])).tolist()
    elif v == "c":
        sp = power_law(d, 1.0)
        k = max(1, d // 10)
        sel = np.arange(k) if p["v_where"] == "top" else np.arange(d - k, d)
        us = np.zeros(d)
        us[sel] = np.sqrt(d / k)
        cfg["spectrum"] = sp.tolist()
        cfg["u_scale"] = us.tolist()
    elif v == "d":
        cfg.update(K=int(p["v_K"]), sep=float(p["v_sep"]))
    elif v == "e":
        G = int(p["v_G"])
        rng = np.random.default_rng(CM.seed_seq("C", dict(unit=uid, what="n_g")))
        s2 = np.log(2.0)
        cfg["n_g"] = np.maximum(2, np.rint(n * np.exp(np.sqrt(s2) * rng.standard_normal(G) - s2 / 2))).astype(int).tolist()
    return cfg


def diagnostics(X, cls=None):
    """Diagnostics computable on real features: log condition number of the LW estimate T, PR/d, mean excess kurtosis of
    the whitened coordinates, LW shrinkage, N/d, class-separation index (tr S_between-class / tr S_within-class)."""
    N, d = X.shape
    lw = S64.LW(X)
    T = (1 - lw.shrinkage) * lw.S + lw.shrinkage * lw.mu * torch.eye(d, dtype=torch.float64, device=DEV)
    ev = torch.linalg.eigvalsh(T)
    evS = torch.linalg.eigvalsh(lw.S).clamp_min(0)
    pr = float(evS.sum() ** 2 / (evS ** 2).sum())
    L = torch.linalg.cholesky(T)
    W = torch.linalg.solve_triangular(L, (X - lw.loc).T, upper=False).T
    W = (W - W.mean(0)) / W.std(0)
    kurt = float(((W ** 4).mean(0) - 3).mean())
    csi = 0.0
    if cls is not None and len(np.unique(cls)) > 1:
        ct = torch.as_tensor(cls, device=DEV)
        mu = X.mean(0)
        sb, sw = 0.0, 0.0
        for c in np.unique(cls):
            Xc = X[ct == int(c)]
            mc = Xc.mean(0)
            sb += Xc.shape[0] * float(((mc - mu) ** 2).sum())
            sw += float(((Xc - mc) ** 2).sum())
        csi = sb / sw
    return dict(diag_logcond=float(torch.log(ev.max() / ev.min())), diag_pr_over_d=pr / d, diag_kurtosis=kurt,
                diag_delta_lw=float(lw.shrinkage), diag_N_over_d=N / d, diag_class_sep=csi)


class _RecRNG:
    """numpy Generator proxy recording the outputs of integers() (class and group indices of the draws)."""

    def __init__(self, rng):
        self._rng, self.calls = rng, []

    def integers(self, *a, **k):
        r = self._rng.integers(*a, **k)
        self.calls.append(r)
        return r

    def __getattr__(self, name):
        return getattr(self._rng, name)


def _class_model(cfg, dr):
    """Variant-d model with class labels: same draws as sim_c.Model, with the class indices captured."""
    dr.np = _RecRNG(dr.np)
    mod = sim_c.Model(cfg, dr)
    return mod, np.asarray(dr.np.calls[-1])


def cc_replicate(cfg, ss, m=sim_c.M):
    """Class-conditional Mahalanobis (tied within-class covariance + lambda I, own-class Q) for every lambda: mean Q seen and unseen."""
    dr = sim_c.Draw(ss)
    mod, cls_g = _class_model(cfg, dr)
    X = mod.fit()
    N, d = X.shape
    gidx = np.repeat(np.arange(len(mod.n_g)), mod.n_g)
    cx = torch.as_tensor(cls_g[gidx], device=DEV)
    K = mod.means.shape[0]
    mus = torch.stack([X[cx == c].mean(0) if bool((cx == c).any()) else torch.zeros(d, dtype=torch.float64, device=DEV) for c in range(K)])
    R = X - mus[cx]
    Sw = R.T @ R / N
    # seen: classes of the drawn groups; unseen: classes of the fresh draws
    Zs = mod.seen(m)
    cs = torch.as_tensor(cls_g[np.asarray(dr.np.calls[-1])], device=DEV)
    Zu = mod.fresh(m)
    cu = torch.as_tensor(np.asarray(dr.np.calls[-1]), device=DEV)
    eye = torch.eye(d, dtype=torch.float64, device=DEV)
    out = {}
    for lam in sim_c.LAMS:
        if lam == 0 and N < 2 * d:
            continue
        L = torch.linalg.cholesky(Sw + lam * eye)
        q = lambda Z, c: (torch.linalg.solve_triangular(L, (Z - mus[c]).T, upper=False) ** 2).sum(0)  # noqa: E731
        out[f"cc_l{lam:g}"] = (float(q(Zs, cs).mean()), float(q(Zu, cu).mean()))
    return out, X, cls_g[gidx]


def run_c5(uid, p):
    cfg = variant_cfg(uid, p)
    seed_fn = lambda r: CM.seed_seq("C", uid, r)  # noqa: E731
    agg, _ = sim_c.run_config(cfg, seed_fn, scorers=("maha",))
    res = dict(variant=p["variant"], n_g_used=cfg["n_g"] if p["variant"] == "e" else None, **agg)
    if p["variant"] == "b":
        aw, _ = sim_c.run_config(dict(cfg, whiten_ridge=True), seed_fn, scorers=("maha",))
        res.update({f"wh|{k}": v for k, v in aw.items()})
    cls = None
    if p["variant"] == "d":
        reps = []
        for r in range(agg["R"]):
            o, X0, c0 = cc_replicate(cfg, seed_fn(r))
            if r == 0:
                cls = c0
            reps.append(o)
            del X0
        for name in reps[0]:
            dq = np.array([o[name][0] - o[name][1] for o in reps])
            res[f"{name}|dq_sim"] = float(dq.mean())
            res[f"{name}|dq_sim|se"] = float(dq.std(ddof=1) / np.sqrt(len(dq))) if len(dq) > 1 else float("nan")
            res[f"{name}|s_sim"] = float(np.mean([o[name][1] for o in reps]))
    dr = sim_c.Draw(seed_fn(0))
    X = sim_c.Model(cfg, dr).fit()
    dg = diagnostics(X, cls)
    dg["diag_G"] = len(cfg["n_g"])
    res.update(dg)
    del X
    return res
