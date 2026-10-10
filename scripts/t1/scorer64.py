"""float64 GPU implementation of the Track A `mahalanobis_l2` scorer and the K2 descriptors (T1 items A, G, H).

Scorer: L2-normalise x / (||x|| + 1e-8); Ledoit-Wolf (sklearn formula, centring at the mean, divisor N);
Q = (x - mu)' P (x - mu); Track A score = -sqrt(Q) (higher = more ID), so AUROCs equal those of -Q.
"""
from __future__ import annotations

import numpy as np
import torch

DEV = __import__("os").environ.get("T1_DEV", "cuda")


def l2n(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def to_t(x):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=DEV)


class LW:
    """Ledoit-Wolf fit on a float64 GPU tensor X (already L2-normalised)."""

    def __init__(self, X: torch.Tensor):
        n, p = X.shape
        self.n, self.p = n, p
        self.loc = X.mean(0)
        Xc = X - self.loc
        self.S = Xc.T @ Xc / n
        X2 = Xc * Xc
        emp_tr = X2.sum(0) / n
        mu = emp_tr.sum() / p
        beta_ = (X2.T @ X2).sum()
        delta_ = (self.S * self.S).sum()
        beta = 1.0 / (p * n) * (beta_ / n - delta_)
        delta = (delta_ - 2.0 * mu * emp_tr.sum() + p * mu ** 2) / p
        beta = torch.minimum(beta, delta)
        self.shrinkage = 0.0 if float(beta) == 0 else float(beta / delta)
        self.mu = float(mu)
        eye = torch.eye(p, dtype=torch.float64, device=X.device)
        self.P = torch.linalg.inv((1 - self.shrinkage) * self.S + self.shrinkage * self.mu * eye)
        del Xc, X2

    def q(self, Z: torch.Tensor, chunk=65536) -> torch.Tensor:
        out = []
        for i in range(0, Z.shape[0], chunk):
            Zc = Z[i:i + chunk] - self.loc
            out.append(((Zc @ self.P) * Zc).sum(1).clamp_min(0.0))
        return torch.cat(out) if out else torch.empty(0, dtype=torch.float64, device=Z.device)

    def s_tr(self) -> float:
        """tr (S_n + lam_eff I)^{-1} in normalised units = (1 - delta) mu tr P."""
        return float((1 - self.shrinkage) * self.mu * torch.trace(self.P))


def auroc_id_pos(s_id, s_ood):
    """P(s_ood < s_id) + 0.5 P(tie); higher score = more ID (package convention)."""
    s_id, s_ood = np.asarray(s_id, dtype=np.float64), np.sort(np.asarray(s_ood, dtype=np.float64))
    lo = np.searchsorted(s_ood, s_id, side="left")
    hi = np.searchsorted(s_ood, s_id, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(s_id) * len(s_ood)))


def group_stats(X: torch.Tensor, g: np.ndarray, P: torch.Tensor):
    """rho_w, rho_raw, rho_anova, R3 whitened ICC and group sizes for fit set X (normalised features) with groups g."""
    N, d = X.shape
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    G = len(u)
    inv_t = torch.as_tensor(inv, device=X.device)
    cnt_t = torch.as_tensor(cnt, dtype=torch.float64, device=X.device)
    m = X.mean(0)
    sums = torch.zeros((G, d), dtype=torch.float64, device=X.device).index_add_(0, inv_t, X)
    M = sums / cnt_t[:, None] - m
    Xc = X - m
    SST = Xc.T @ Xc
    SSB = (M * cnt_t[:, None]).T @ M
    SSW = SST - SSB
    W = SSW / (N - G)
    T = SST / (N - 1)
    rho_w = 1.0 - float((P * W).sum() / (P * T).sum())
    rho_raw = 1.0 - float(torch.trace(W) / torch.trace(T))
    n0 = (N - float((cnt ** 2).sum()) / N) / (G - 1)
    msb = torch.diagonal(SSB) / (G - 1)
    msw = torch.diagonal(SSW) / (N - G)
    den = msb + (n0 - 1) * msw
    ok = den > 0
    rho_anova = float(((msb - msw)[ok] / den[ok]).mean())
    icc_wh = float((P * (SSB / N)).sum() / (P * (SST / N)).sum())
    return dict(G=G, rho_w=rho_w, rho_raw=rho_raw, rho_anova=rho_anova, icc_whitened_r3=icc_wh, n0=n0), u, cnt


def dq_pred_groups(n_g, pi_g, rho, s, N, delta):
    """K2 per-group closed form in scorer units: (1/(1-delta)) sum_g pi_g [- n_g rho^2 s^2 / (N + w1_g s)]."""
    n_g, pi_g = np.asarray(n_g, float), np.asarray(pi_g, float)
    w1 = 1 + (n_g - 1) * rho
    return float(np.sum(pi_g * (-n_g * rho ** 2 * s ** 2 / (N + w1 * s)))) / (1 - delta)


def dq_pred_plugin(n, rho, s, N, delta):
    w1 = 1 + (n - 1) * rho
    return float(-n * rho ** 2 * s ** 2 / (N + w1 * s)) / (1 - delta)
