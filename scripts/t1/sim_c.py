"""T1 item C: common float64 GPU simulator of the random-effects model (order, "Common simulator spec").

Per config and replicate: group effects u_g and fit data X (variant-dependent), scorers fit on X only, m = 4,000 seen
(new draws from uniformly chosen fit groups), 4,000 unseen (fresh groups), 4,000 OOD per OOD setting
(O1 isotropic scale a^2 = 1 + c sqrt(2/d), c in {1,2,4}; O2 mean shift |m|^2 = c' sqrt(2d), c' in {.5,1,2}; O3 null = fresh
unseen-type draw). Orientation of every reported AUROC: higher = more ID (Delta = AUROC_seen - AUROC_unseen).
Adaptive replicates: stop when SE(dQ_sim) <= 2 % |dQ_th| for every uncentred S_lambda with a closed form and SE(Delta) <= 0.004
for every scorer x OOD setting, with R_min = 8, R_max = 64.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import norm

import sim_re
import scorer64 as S64

DEV = S64.DEV
M = 4000
LAMS = (0.0, 0.01, 0.1, 1.0)
O1_C = (1.0, 2.0, 4.0)
O2_C = (0.5, 1.0, 2.0)
R_MIN, R_MAX = 8, 64
CHUNK = 2048


def vim_dim(d):
    from crossfit_ood.scorers import _default_vim_dim
    return _default_vim_dim(d)


def ood_names():
    return [f"O1c{c:g}" for c in O1_C] + [f"O2c{c:g}" for c in O2_C] + ["O3"]


class Draw:
    """Random draws for one replicate: torch generator for large normals, numpy generator for small things."""

    def __init__(self, ss: np.random.SeedSequence):
        a, b = ss.spawn(2)
        self.np = np.random.default_rng(a)
        self.g = torch.Generator(device=DEV)
        self.g.manual_seed(int(b.generate_state(1, dtype=np.uint64)[0] % (2 ** 63)))

    def normal(self, *shape):
        return torch.randn(*shape, dtype=torch.float64, device=DEV, generator=self.g)

    def t_scale(self, n, nu):
        """Per-row factor sqrt((nu-2)/chi2_nu) so that a multivariate-t row has the covariance of the Gaussian row."""
        if nu is None or not np.isfinite(nu):
            return None
        return torch.as_tensor(np.sqrt((nu - 2.0) / self.np.chisquare(nu, n)), device=DEV)


def _rows(z, scale):
    return z if scale is None else z * scale[:, None]


class Model:
    """Variant of M(G, n, rho, d): x = Sigma^{1/2} (u_g + e), u_g ~ rho * law_u, e ~ (1 - rho) * law_e (total cov Sigma).
    cfg keys: d, rho, n_g (array of group sizes), nu_u, nu_e (None = Gaussian), spectrum (None or diag of Sigma, mean 1)."""

    def __init__(self, cfg, dr: Draw):
        self.d, self.rho = int(cfg["d"]), float(cfg["rho"])
        self.n_g = np.asarray(cfg["n_g"], dtype=int)
        self.nu_u, self.nu_e = cfg.get("nu_u"), cfg.get("nu_e")
        sp = cfg.get("spectrum")
        self.sq = None if sp is None else torch.as_tensor(np.sqrt(np.asarray(sp, float)), device=DEV)
        self.dr = dr
        G = len(self.n_g)
        self.u = np.sqrt(self.rho) * _rows(dr.normal(G, self.d), dr.t_scale(G, self.nu_u))

    def _e(self, n):
        return np.sqrt(1 - self.rho) * _rows(self.dr.normal(n, self.d), self.dr.t_scale(n, self.nu_e))

    def _out(self, z):
        return z if self.sq is None else z * self.sq[None, :]

    def fit(self):
        idx = torch.as_tensor(np.repeat(np.arange(len(self.n_g)), self.n_g), device=DEV)
        return self._out(self.u[idx] + self._e(len(idx)))

    def seen(self, m):
        gi = torch.as_tensor(self.dr.np.integers(0, len(self.n_g), m), device=DEV)
        return self._out(self.u[gi] + self._e(m))

    def fresh(self, m):
        uf = np.sqrt(self.rho) * _rows(self.dr.normal(m, self.d), self.dr.t_scale(m, self.nu_u))
        return self._out(uf + self._e(m))

    def oods(self, m):
        out = {}
        for c in O1_C:
            out[f"O1c{c:g}"] = np.sqrt(1 + c * np.sqrt(2.0 / self.d)) * self.fresh(m)
        for c in O2_C:
            v = self.dr.normal(self.d)
            v = v / torch.linalg.norm(v) * np.sqrt(c * np.sqrt(2.0 * self.d))
            out[f"O2c{c:g}"] = self.fresh(m) + self._out(v[None, :])
        out["O3"] = self.fresh(m)
        return out


# ---------------------------------------------------------------- scorers (anomaly scores: higher = more anomalous)
def maha_family(X, whiten=None, lams=LAMS, lw=True):
    """Q functions for S_lambda = X'X/N + lam I (uncentred, 'u') and its centred version ('c'), lambda = 0 only if N >= 2d;
    and the Ledoit-Wolf scorer (centred). whiten: optional per-coordinate scale applied before fitting (whitened ridge)."""
    N, d = X.shape
    Xw = X if whiten is None else X / whiten[None, :]
    mu = Xw.mean(0)
    G0 = Xw.T @ Xw / N
    Gc = G0 - torch.outer(mu, mu)
    eye = torch.eye(d, dtype=torch.float64, device=DEV)
    fam, skipped = {}, []
    for lam in lams:
        if lam == 0 and N < 2 * d:
            skipped.append(lam)
            continue
        for tag, Sm, loc in (("u", G0, None), ("c", Gc, mu)):
            L = torch.linalg.cholesky(Sm + lam * eye)
            fam[f"maha_{tag}_l{lam:g}"] = (L, loc)
    if lw:
        lwf = S64.LW(Xw)
        fam["maha_lw"] = (torch.linalg.cholesky((1 - lwf.shrinkage) * lwf.S + lwf.shrinkage * lwf.mu * eye), lwf.loc)
    return fam, skipped, whiten


def maha_q(entry, Z, whiten=None):
    L, loc = entry
    Zw = Z if whiten is None else Z / whiten[None, :]
    if loc is not None:
        Zw = Zw - loc
    Y = torch.linalg.solve_triangular(L, Zw.T, upper=False)
    return (Y * Y).sum(0)


def knn_dist(X, Z, ks):
    """Euclidean distance to the k-th neighbour (kNN-k, Sun et al. form) for every k in ks."""
    xn = (X * X).sum(1)
    kmax = max(ks)
    out = []
    for i in range(0, Z.shape[0], CHUNK):
        z = Z[i:i + CHUNK]
        d2 = ((z * z).sum(1, keepdim=True) - 2.0 * z @ X.T + xn[None, :]).clamp_min(0.0)
        v = torch.sqrt(torch.topk(d2, kmax, dim=1, largest=False).values)
        out.append(torch.stack([v[:, k - 1] for k in ks], 1))
    return torch.cat(out)


def knn_cos_mean(Xu, Z, k):
    """Track A form: mean cosine distance to the k nearest fit points (unit vectors)."""
    Zu = Z / torch.linalg.norm(Z, dim=1, keepdim=True)
    out = []
    for i in range(0, Zu.shape[0], CHUNK):
        s = Zu[i:i + CHUNK] @ Xu.T
        out.append((1.0 - torch.topk(s, k, dim=1).values).mean(1))
    return torch.cat(out)


def vim_fit(X):
    mu = X.mean(0)
    Xc = X - mu
    C = Xc.T @ Xc / X.shape[0]
    _, V = torch.linalg.eigh(C)
    p = vim_dim(X.shape[1])
    return mu, V[:, -p:], p


def vim_res(fit, Z):
    mu, V, _ = fit
    Zc = Z - mu
    return torch.linalg.norm(Zc - (Zc @ V) @ V.T, dim=1)


def auroc(s_id_anom, s_ood_anom):
    """AUROC with orientation higher = more ID, from anomaly scores: P(s_ood > s_id) + 1/2 P(tie)."""
    return S64.auroc_id_pos(-np.asarray(s_id_anom), -np.asarray(s_ood_anom))


def auroc_mix_direct(s_seen, s_unseen, s_ood, omega):
    """Same mixture AUROC computed directly from the weighted pooled ID sample (not by linearity)."""
    s_id = np.r_[s_seen, s_unseen]
    w = np.r_[np.full(len(s_seen), omega / len(s_seen)), np.full(len(s_unseen), (1 - omega) / len(s_unseen))]
    so = np.sort(-np.asarray(s_ood))
    x = -s_id
    lo = np.searchsorted(so, x, side="left")
    hi = np.searchsorted(so, x, side="right")
    return float(np.sum(w * (lo + 0.5 * (hi - lo))) / len(so))


# ---------------------------------------------------------------- one replicate
def replicate(cfg, ss, scorers=("maha", "knn", "cos", "vim"), ks=(1, 5, 20), m=M):
    t0 = time.time()
    dr = Draw(ss)
    mod = Model(cfg, dr)
    X = mod.fit()
    N, d = X.shape
    Zs, Zu = mod.seen(m), mod.fresh(m)
    Zo = mod.oods(m)
    res = dict(N=N)
    scores = {}
    if "maha" in scorers:
        wh = mod.sq if cfg.get("whiten_ridge") else None
        fam, skipped, wh = maha_family(X, whiten=wh)
        res["lam_skipped"] = skipped
        for name, ent in fam.items():
            qs, qu = maha_q(ent, Zs, wh), maha_q(ent, Zu, wh)
            res[f"{name}|q_seen"] = float(qs.mean())
            res[f"{name}|q_unseen"] = float(qu.mean())
            scores[name] = (qs.cpu().numpy(), qu.cpu().numpy(), {o: maha_q(ent, z, wh).cpu().numpy() for o, z in Zo.items()})
        del fam
    kk = [k for k in ks if k <= N]
    if "knn" in scorers and kk:
        ds, du = knn_dist(X, Zs, kk).cpu().numpy(), knn_dist(X, Zu, kk).cpu().numpy()
        do = {o: knn_dist(X, z, kk).cpu().numpy() for o, z in Zo.items()}
        for j, k in enumerate(kk):
            scores[f"knn{k}"] = (ds[:, j], du[:, j], {o: v[:, j] for o, v in do.items()})
    if "cos" in scorers:
        Xu = X / torch.linalg.norm(X, dim=1, keepdim=True)
        k = min(50, N)
        scores["knncos50"] = (knn_cos_mean(Xu, Zs, k).cpu().numpy(), knn_cos_mean(Xu, Zu, k).cpu().numpy(),
                              {o: knn_cos_mean(Xu, z, k).cpu().numpy() for o, z in Zo.items()})
        del Xu
    if "vim" in scorers and vim_dim(d) < d:
        vf = vim_fit(X)
        scores["vim"] = (vim_res(vf, Zs).cpu().numpy(), vim_res(vf, Zu).cpu().numpy(),
                         {o: vim_res(vf, z).cpu().numpy() for o, z in Zo.items()})
    k1 = 0.0
    for name, (ss_, su, so) in scores.items():
        for o, sd in so.items():
            a_s, a_u = auroc(ss_, sd), auroc(su, sd)
            res[f"{name}|{o}|auroc_seen"] = a_s
            res[f"{name}|{o}|auroc_unseen"] = a_u
            res[f"{name}|{o}|delta"] = a_s - a_u
        sd = so["O1c2"]
        d1 = res[f"{name}|O1c2|delta"]
        for om in (0.0, 0.25, 0.5, 1.0):
            dom = auroc_mix_direct(ss_, su, sd, om) - auroc(su, sd)
            k1 = max(k1, abs(dom - om * d1))
    res["k1_max_abs_dev"] = k1
    # additive / Gaussian maps for the uncentred S_lambda (C2) with the closed-form dQ_th
    if "maha" in scorers and cfg.get("closed_form", True):
        n_eff = float(np.mean(mod.n_g))
        for lam in LAMS:
            name = f"maha_u_l{lam:g}"
            if name not in scores or (lam == 0 and N <= d):
                continue
            _, dq = sim_re.k2_theory(d, len(mod.n_g), n_eff, mod.rho, lam)
            _, su, so = scores[name]
            for o in [x for x in so if x != "O3"]:
                q_o = so[o]
                a0 = auroc(su, q_o)
                res[f"{name}|{o}|pred_add"] = auroc(su - abs(dq), q_o) - a0
                sdv = np.sqrt(q_o.var(ddof=1) + su.var(ddof=1))
                res[f"{name}|{o}|pred_gauss"] = float(norm.cdf((q_o.mean() - (su.mean() + dq)) / sdv) - norm.cdf((q_o.mean() - su.mean()) / sdv))
    res["seconds"] = time.time() - t0
    del X, Zs, Zu, Zo
    return res


def theory(cfg):
    """Closed-form s and dQ (K2) for every uncentred S_lambda (balanced design n = mean n_g; Gaussian isotropic K2)."""
    d, rho = int(cfg["d"]), float(cfg["rho"])
    n_g = np.asarray(cfg["n_g"])
    G, n = len(n_g), float(n_g.mean())
    out = {}
    for lam in LAMS:
        if lam == 0 and G * n <= d:
            continue
        s, dq = sim_re.k2_theory(d, G, n, rho, lam)
        out[f"maha_u_l{lam:g}"] = dict(s_th=s, dq_th=dq)
    return out


def run_config(cfg, seed_fn, scorers=("maha", "knn", "cos", "vim"), ks=(1, 5, 20), r_min=R_MIN, r_max=R_MAX,
               se_dq_rel=0.02, se_delta=0.004):
    """Adaptive replicates. seed_fn(r) -> SeedSequence of replicate r. Returns (aggregate dict, list of per-replicate dicts)."""
    th = theory(cfg) if cfg.get("closed_form", True) else {}
    reps = []
    while True:
        reps.append(replicate(cfg, seed_fn(len(reps)), scorers=scorers, ks=ks))
        R = len(reps)
        if R < r_min:
            continue
        ok = True
        for name, t in th.items():
            key_s, key_u = f"{name}|q_seen", f"{name}|q_unseen"
            if key_s not in reps[0]:
                continue
            dq = np.array([r[key_s] - r[key_u] for r in reps])
            if dq.std(ddof=1) / np.sqrt(R) > se_dq_rel * abs(t["dq_th"]):
                ok = False
                break
        if ok:
            for key in [k for k in reps[0] if k.endswith("|delta")]:
                v = np.array([r[key] for r in reps])
                if v.std(ddof=1) / np.sqrt(R) > se_delta:
                    ok = False
                    break
        if ok or R >= r_max:
            break
    agg = dict(R=len(reps), seconds_per_rep=float(np.mean([r["seconds"] for r in reps])),
               k1_max_abs_dev=float(max(r["k1_max_abs_dev"] for r in reps)), N=reps[0]["N"])
    keys = [k for k in reps[0] if isinstance(reps[0][k], float) and k not in ("seconds", "k1_max_abs_dev")]
    for k in keys:
        v = np.array([r[k] for r in reps])
        agg[k] = float(v.mean())
        agg[k + "|se"] = float(v.std(ddof=1) / np.sqrt(len(v)))
    for name, t in th.items():
        if f"{name}|q_seen" in reps[0]:
            dq = np.array([r[f"{name}|q_seen"] - r[f"{name}|q_unseen"] for r in reps])
            agg[f"{name}|dq_sim"] = float(dq.mean())
            agg[f"{name}|dq_sim|se"] = float(dq.std(ddof=1) / np.sqrt(len(dq)))
            agg[f"{name}|s_sim"] = agg[f"{name}|q_unseen"]
            agg[f"{name}|dq_th"] = t["dq_th"]
            agg[f"{name}|s_th"] = t["s_th"]
    agg["lam_skipped"] = reps[0].get("lam_skipped", [])
    return agg, reps
