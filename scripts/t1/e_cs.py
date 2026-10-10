"""T1 item E: time-uniform confidence sequences (float64 torch, vectorised over replicates).

Betting CS (implemented from Waudby-Smith & Ramdas, arXiv 2010.09686; `confseq` is not installed): hedged capital process
K_t(m) = 1/2 prod_i (1 + l+_i(m)(x_i - m)) + 1/2 prod_i (1 - l-_i(m)(x_i - m)) with predictable plug-in empirical-Bernstein
bets l_t = sqrt(2 log(2/alpha) / (sig2_{t-1} t log(1+t))), l+_t(m) = min(l_t, c/m), l-_t(m) = min(l_t, c/(1-m)), c = 1/2,
mu_t = (1/2 + sum_{i<=t} x_i)/(t+1), sig2_t = (1/4 + sum_{i<=t} (x_i - mu_i)^2)/(t+1); CS_t = {m in grid : K_t(m) < 1/alpha},
grid = 1,001 points on [0, 1]. Decisions (miscoverage, width, certificates) are read at the decision times passed in
(group boundaries). Miscoverage of a value theta uses K_t(theta) exactly; widths use the running intersection of the grid CS.
"""
from __future__ import annotations

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
C_TRUNC = 0.5


def prpl_lambda(x, alpha):
    """x: (R, T) in [0,1]. Returns predictable bets l_t (R, T) (uses x_1..x_{t-1} only)."""
    R, T = x.shape
    t = torch.arange(1, T + 1, dtype=torch.float64, device=x.device)
    mu = (0.5 + torch.cumsum(x, 1)) / (t + 1)
    sig2 = (0.25 + torch.cumsum((x - mu) ** 2, 1)) / (t + 1)
    sig2_prev = torch.cat([torch.full((R, 1), 0.25, dtype=torch.float64, device=x.device), sig2[:, :-1]], 1)
    return torch.sqrt(2 * np.log(2 / alpha) / (sig2_prev * t * torch.log1p(t)))


def logK_at(x, lam, m):
    """log K_t(m) for every t (R, T); m: scalar or (R,) tensor."""
    m = torch.as_tensor(m, dtype=torch.float64, device=x.device)
    if m.dim() == 0:
        m = m.expand(x.shape[0])
    m = m[:, None]
    lp = torch.minimum(lam, C_TRUNC / m.clamp_min(1e-12))
    lm = torch.minimum(lam, C_TRUNC / (1 - m).clamp_min(1e-12))
    a = torch.cumsum(torch.log1p(lp * (x - m)), 1)
    b = torch.cumsum(torch.log1p(-lm * (x - m)), 1)
    return torch.logaddexp(a, b) - np.log(2.0)


def ever_miscover(x, lam, theta, dec_idx, alpha):
    """1 if K_t(theta) >= 1/alpha at some decision time (indices dec_idx into the stream)."""
    lk = logK_at(x, lam, theta)[:, dec_idx]
    return (lk >= np.log(1 / alpha)).any(1)


def grid_cs(x, lam, dec_idx, alpha, chunk=None):
    """Running-intersection grid CS at the decision times. Returns lo, hi (R, n_dec) endpoints on the grid."""
    R, T = x.shape
    grid = GRID.to(x.device)
    thr = np.log(1 / alpha)
    lo = torch.zeros(R, dtype=torch.long, device=x.device)
    hi = torch.full((R,), 1000, dtype=torch.long, device=x.device)
    carry_a = torch.zeros(R, 1001, dtype=torch.float64, device=x.device)
    carry_b = torch.zeros(R, 1001, dtype=torch.float64, device=x.device)
    dec = torch.as_tensor(dec_idx, device=x.device)
    out_lo = torch.empty(R, len(dec_idx), dtype=torch.long, device=x.device)
    out_hi = torch.empty_like(out_lo)
    di = 0
    s = 0
    while s < T:
        w0, w1 = int(lo.min()), int(hi.max()) + 1
        step = chunk or max(8, int(1.5e8 // (R * (w1 - w0))))
        e = min(T, s + step)
        m = grid[w0:w1]
        xs, ls = x[:, s:e, None], lam[:, s:e, None]
        lp = torch.minimum(ls, C_TRUNC / m.clamp_min(1e-12))
        lm = torch.minimum(ls, C_TRUNC / (1 - m).clamp_min(1e-12))
        a = carry_a[:, None, w0:w1] + torch.cumsum(torch.log1p(lp * (xs - m)), 1)
        b = carry_b[:, None, w0:w1] + torch.cumsum(torch.log1p(-lm * (xs - m)), 1)
        carry_a[:, w0:w1], carry_b[:, w0:w1] = a[:, -1], b[:, -1]
        sel = (dec >= s) & (dec < e)
        idxs = (dec[sel] - s).tolist()
        for j in idxs:
            lk = torch.logaddexp(a[:, j], b[:, j]) - np.log(2.0)
            ok = lk < thr
            gi = torch.arange(w0, w1, device=x.device)[None, :]
            act = ok & (gi >= lo[:, None]) & (gi <= hi[:, None])
            anyact = act.any(1)
            first = torch.where(anyact, act.float().argmax(1) + w0, lo)
            last = torch.where(anyact, (w1 - 1) - act.flip(1).float().argmax(1), lo)
            # an empty CS keeps a degenerate interval at its previous lower end
            hi = torch.where(anyact, last, lo)
            lo = first
            out_lo[:, di], out_hi[:, di] = lo, hi
            di += 1
        del a, b, lp, lm
        s = e
    return out_lo, out_hi


def grid_cs_mask(x, lam, dec_mask, n_dec, alpha):
    """As grid_cs, with replicate-specific decision times: dec_mask (R, T) bool marks the decision times (group ends);
    returns lo, hi (R, n_dec) at the k-th decision of each replicate (k < number of its decisions)."""
    R, T = x.shape
    grid = GRID.to(x.device)
    thr = np.log(1 / alpha)
    active = torch.ones(R, 1001, dtype=torch.bool, device=x.device)
    carry_a = torch.zeros(R, 1001, dtype=torch.float64, device=x.device)
    carry_b = torch.zeros(R, 1001, dtype=torch.float64, device=x.device)
    out_lo = torch.zeros(R, n_dec, dtype=torch.long, device=x.device)
    out_hi = torch.zeros(R, n_dec, dtype=torch.long, device=x.device)
    kdec = torch.zeros(R, dtype=torch.long, device=x.device)
    prev_lo = torch.zeros(R, dtype=torch.long, device=x.device)
    s = 0
    while s < T:
        idx = active.any(0).nonzero()
        w0, w1 = (int(idx.min()), int(idx.max()) + 1) if len(idx) else (0, 1)
        step = max(8, int(1.2e8 // (R * (w1 - w0))))
        e = min(T, s + step)
        m = grid[w0:w1]
        xs, ls = x[:, s:e, None], lam[:, s:e, None]
        lp = torch.minimum(ls, C_TRUNC / m.clamp_min(1e-12))
        lm = torch.minimum(ls, C_TRUNC / (1 - m).clamp_min(1e-12))
        a = carry_a[:, None, w0:w1] + torch.cumsum(torch.log1p(lp * (xs - m)), 1)
        b = carry_b[:, None, w0:w1] + torch.cumsum(torch.log1p(-lm * (xs - m)), 1)
        carry_a[:, w0:w1], carry_b[:, w0:w1] = a[:, -1], b[:, -1]
        dm = dec_mask[:, s:e]
        bad = (torch.logaddexp(a, b) - np.log(2.0) >= thr) & dm[:, :, None]
        excl = torch.cummax(bad.to(torch.uint8), 1).values.bool()
        act = active[:, None, w0:w1] & ~excl
        anyact = act.any(2)
        first = torch.where(anyact, act.to(torch.uint8).argmax(2) + w0, torch.full_like(anyact, -1, dtype=torch.long))
        # an empty CS keeps a degenerate interval at its last lower end
        frozen = torch.maximum(torch.cummax(first, 1).values, prev_lo[:, None])
        first = torch.where(anyact, first, frozen)
        last = torch.where(anyact, (w1 - 1) - act.flip(2).to(torch.uint8).argmax(2), frozen)
        prev_lo = first[:, -1]
        kpos = kdec[:, None] + torch.cumsum(dm.long(), 1) - 1
        ri, ci = dm.nonzero(as_tuple=True)
        k = kpos[ri, ci].clamp_max(n_dec - 1)
        out_lo[ri, k] = first[ri, ci]
        out_hi[ri, k] = last[ri, ci]
        kdec += dm.long().sum(1)
        active[:, w0:w1] = act[:, -1]
        del a, b, lp, lm, bad, excl, act
        s = e
    return out_lo, out_hi


def mix_radius(t, sd, alpha, t0):
    """Normal-mixture asymptotic CS radius (bundle s1b_cs.py)."""
    r2 = (-2 * np.log(alpha) + np.log(-2 * np.log(alpha) + 1)) / t0
    return sd * torch.sqrt(2 * (t * r2 + 1) / (t ** 2 * r2) * torch.log(torch.sqrt(t * r2 + 1) / alpha))


def asym_cs(gm, alpha, t0=30):
    """N3: group-level asymptotic CS on group means (bundle s1b), returns (mean, radius) (R, G)."""
    R, G = gm.shape
    t = torch.arange(1, G + 1, dtype=torch.float64, device=gm.device)
    mu = torch.cumsum(gm, 1) / t
    v = torch.cumsum(gm ** 2, 1) / t - mu ** 2
    sd = torch.sqrt(v.clamp_min(1e-12) * t / torch.clamp(t - 1, min=1))
    sd[:, 0] = 1.0
    return mu, mix_radius(t, sd, alpha, t0)
