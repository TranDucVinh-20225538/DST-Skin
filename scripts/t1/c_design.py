"""T1 item C: enumeration of every C unit (C0, C1, C3, C4, C5) with its parameters (deterministic; seed 301 where the order says so).

Unit IDs (first column of the raw output):
  c0|s2a|d<d>_G<G>_n<n>_rho<rho>_lam<lam>       s2a reproduction (the three A0a cells), >= 64 replicates
  c0|s2b|d<d>_G<G>_n<n>_rho<rho>_lam<lam>       s2b Maha Delta_sim lines (16), >= 64 replicates
  c0|s2d|d<d>_rho<rho>_G<G>                      s2d exact-TV table (d = 1: 12 rows, d = 2: 15 rows), >= 64 draws
  c0|s2f|d<d>_rho<rho>                           s2f kNN-1 null-OOD leak (6 values), G = 100, n = 5, >= 64 replicates
  c0|time|d<d>_G<G>_n<n>                         seconds per replicate of the common simulator by (d, N) (budget)
  c1|<i:04d>                                     C1 config i of 6,000 (Latin hypercube over the discrete levels, seed 301)
  c3|d<d>_rho<rho>_G<G>_n<n>                     C3 kNN collapse grid (378 configs)
  c4tv|d<d>_rho<rho>_G<G>                        C4 exact-TV grid (75 configs), >= 40 draws
  c4sc|<law>|d<d>_rho<rho>_G<G>                  C4 scorer grid, law in {gauss, t5, aniso} (300 configs), >= 64 replicates
  c5|<variant>|<i:04d>                           C5 variant a..e on the 1,500-config sub-design of C1 (seed 301)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

C1_D = [16, 32, 64, 128, 256, 512, 1024, 2048]
C1_RHO = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7]
C1_G = [6, 12, 24, 48, 100, 200, 400]
C1_N = [5, 10, 25, 50, 100, 500]
TRAIN_D, INTERP_D = {16, 32, 128, 512}, {64, 256, 1024}

S2A = [(128, 40, 10, 0.6, 0.0), (128, 100, 5, 0.3, 0.1), (32, 40, 10, 0.6, 0.0)]
S2B = [(d, G, n, rho, lam) for (d, G, n) in ((64, 100, 5), (128, 100, 5), (128, 200, 10), (256, 200, 10))
       for rho in (0.2, 0.4) for lam in (0.0, 0.1)]
S2D = [(1, rho, G) for rho in (0.5, 0.9, 0.99) for G in (3, 10, 30, 100)] + \
      [(2, rho, G) for rho in (0.5, 0.9, 0.99) for G in (3, 10, 30, 100, 300)]
S2F = [(16, 0.3), (256, 0.1), (16, 0.5), (64, 0.3), (64, 0.5), (256, 0.3)]
TIME = [(d, G, n) for d in (16, 64, 256, 1024, 2048) for (G, n) in ((12, 100), (100, 100), (200, 500), (400, 500))]

C3_D = [16, 32, 64, 128, 256, 512, 1024]
C3_RHO = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
C3_G = [50, 100, 200]
C3_N = [5, 10, 25]
C4_RHO = [0.3, 0.5, 0.7, 0.9, 0.99]
C4_G = [3, 10, 30, 100, 300]
C5_VARIANTS = "abcde"


def c1_block(d, rho, G):
    if d == 2048 or G == 6 or rho == 0.7:
        return "TEST-extrap"
    return "TRAIN" if d in TRAIN_D else "TEST-interp"


def c1_configs():
    u = qmc.LatinHypercube(d=4, seed=np.random.default_rng(301)).random(6000)
    levels = [C1_D, C1_RHO, C1_G, C1_N]
    out = {}
    for i, row in enumerate(u):
        d, rho, G, n = (lv[min(int(x * len(lv)), len(lv) - 1)] for x, lv in zip(row, levels))
        out[f"c1|{i:04d}"] = dict(d=d, rho=rho, G=G, n=n, N=G * n, block=c1_block(d, rho, G),
                                  in_R0=bool(d >= 64 and G >= 12 and G * n >= 2 * d))
    return out


def c5_configs():
    """Each variant runs on the same 1,500-config sub-design of C1 (rng(301) choice without replacement); the
    variant's sub-levels are assigned to the 1,500 configs by an rng(301)-permuted balanced cycle (one level per config)."""
    c1 = c1_configs()
    ids = sorted(c1)
    rng = np.random.default_rng(301)
    sub = sorted(rng.choice(len(ids), 1500, replace=False))
    levels = {
        "a": [dict(nu=nu, target=t) for nu in (3, 5, 10, 30) for t in ("e", "u")],
        "b": [dict(kind="spiked", r=r, kappa=k) for r in (1, 5, 20) for k in (5, 20, 100)]
             + [dict(kind="powerlaw", alpha=a) for a in (0.5, 1.0, 1.5)],
        "c": [dict(where="top"), dict(where="bottom")],
        "d": [dict(K=K, sep=m) for K in (2, 5, 10) for m in (0, 1, 3, 6)],
        "e": [dict(G=G, n_dist="lognormal_cv1") for G in (6, 12, 24)],
    }
    out = {}
    for v in C5_VARIANTS:
        lv = levels[v]
        assign = rng.permutation(np.arange(1500) % len(lv))
        for j, idx in enumerate(sub):
            base = dict(c1[ids[idx]])
            base.update(variant=v, c1_id=ids[idx], **{f"v_{k}": val for k, val in lv[assign[j]].items()})
            if v == "e":
                base["G"] = lv[assign[j]]["G"]
                base["N"] = None
                base["block"] = c1_block(base["d"], base["rho"], base["G"])
            out[f"c5|{v}|{idx:04d}"] = base
    return out


def all_units():
    u = {}
    for (d, G, n, rho, lam) in S2A:
        u[f"c0|s2a|d{d}_G{G}_n{n}_rho{rho:g}_lam{lam:g}"] = dict(kind="s2a", d=d, G=G, n=n, rho=rho, lam=lam)
    for (d, G, n, rho, lam) in S2B:
        u[f"c0|s2b|d{d}_G{G}_n{n}_rho{rho:g}_lam{lam:g}"] = dict(kind="s2b", d=d, G=G, n=n, rho=rho, lam=lam)
    for (d, rho, G) in S2D:
        u[f"c0|s2d|d{d}_rho{rho:g}_G{G}"] = dict(kind="s2d", d=d, rho=rho, G=G)
    for (d, rho) in S2F:
        u[f"c0|s2f|d{d}_rho{rho:g}"] = dict(kind="s2f", d=d, rho=rho, G=100, n=5)
    for (d, G, n) in TIME:
        u[f"c0|time|d{d}_G{G}_n{n}"] = dict(kind="time", d=d, G=G, n=n)
    u.update(c1_configs())
    for d in C3_D:
        for rho in C3_RHO:
            for G in C3_G:
                for n in C3_N:
                    u[f"c3|d{d}_rho{rho:g}_G{G}_n{n}"] = dict(d=d, rho=rho, G=G, n=n)
    for d in (1, 2, 3):
        for rho in C4_RHO:
            for G in C4_G:
                u[f"c4tv|d{d}_rho{rho:g}_G{G}"] = dict(d=d, rho=rho, G=G)
    for law in ("gauss", "t5", "aniso"):
        for d in (2, 4, 8, 16):
            for rho in C4_RHO:
                for G in C4_G:
                    u[f"c4sc|{law}|d{d}_rho{rho:g}_G{G}"] = dict(law=law, d=d, rho=rho, G=G, n=5)
    u.update(c5_configs())
    return u
