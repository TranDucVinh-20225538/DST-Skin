"""Gaussian random-effects toy M(G, n, rho, d): closed form K2 and a float64 GPU simulator (T1 items A0a, C)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def k2_delta(d, G, n, rho, lam):
    """Deterministic equivalent delta solving
    delta = d / (N lam + G w1/(1+w1 delta) + (N-G)(1-rho)/(1+(1-rho) delta)), N = G n, w1 = 1+(n-1) rho.
    s = N delta = tr S_lam^{-1} with S_lam = X'X/N + lam I."""
    N = G * n
    w1, w2 = 1 + (n - 1) * rho, 1 - rho

    def f(de):
        return de - d / (N * lam + G * w1 / (1 + w1 * de) + (N - G) * w2 / (1 + w2 * de))

    hi = 1.0
    while f(hi) < 0:
        hi *= 2.0
    return brentq(f, 1e-300, hi, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=1000)


def k2_theory(d, G, n, rho, lam):
    """Returns (s = E Q_unseen, dQ = E Q_seen - E Q_unseen = - n rho^2 s^2 / (N + w1 s))."""
    N = G * n
    w1 = 1 + (n - 1) * rho
    s = N * k2_delta(d, G, n, rho, lam)
    return s, -n * rho ** 2 * s ** 2 / (N + w1 * s)


def sim_maha_dq(d, G, n, rho, lam, ss: np.random.SeedSequence, m_seen=2000, m_unseen=2000, g_unseen=400,
                device="cuda"):
    """One replicate of the s2a design: known mean 0, S = X'X/N + lam I, Q = x' S^{-1} x.
    seen = new draws from uniformly chosen fit groups; unseen = draws from g_unseen fresh groups.
    Draws with numpy (SeedSequence), linear algebra in float64 torch on `device`. Returns (mean Q_seen, mean Q_unseen)."""
    import torch
    rng = np.random.default_rng(ss)
    u = rng.standard_normal((G, d)) * np.sqrt(rho)
    X = np.repeat(u, n, axis=0) + rng.standard_normal((G * n, d)) * np.sqrt(1 - rho)
    gi = rng.integers(0, G, m_seen)
    Xin = u[gi] + rng.standard_normal((m_seen, d)) * np.sqrt(1 - rho)
    uo = rng.standard_normal((g_unseen, d)) * np.sqrt(rho)
    go = rng.integers(0, g_unseen, m_unseen)
    Xout = uo[go] + rng.standard_normal((m_unseen, d)) * np.sqrt(1 - rho)
    t = lambda a: torch.from_numpy(a).to(device=device, dtype=torch.float64)  # noqa: E731
    Xt = t(X)
    N = X.shape[0]
    S = Xt.T @ Xt / N + lam * torch.eye(d, dtype=torch.float64, device=device)
    L = torch.linalg.cholesky(S)

    def q(Z):
        Y = torch.linalg.solve_triangular(L, t(Z).T, upper=False)
        return (Y * Y).sum(0)

    return float(q(Xin).mean()), float(q(Xout).mean())
