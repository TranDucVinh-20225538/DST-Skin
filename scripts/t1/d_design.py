"""T1 item D: enumeration of the D units (D1 configs, D2 random configs, D3 / D4 real cells) and the D2 search space.

Unit IDs:
  d1|<i:04d>        D1 config i of 1,000 (Latin hypercube over the D1 levels and the two laws, seed 401)
  d2r|<i:04d>       D2 random config i of 3,000 (uniform in the encoded extended space, seed 401)
  d3|<cell_id>      D3 real cell (both paper_2fold folds)          [stage 3]
  d4|<cell_id>      D4 null on real features (both folds)          [stage 3]
CMA-ES evaluations of D2 (population 32, 40 generations, seed 401) are adaptive and therefore not part of the work list.
"""
from __future__ import annotations

import csv

import numpy as np
from scipy.stats import qmc

D1_D = [32, 128, 512, 1024, 2048]
D1_RHO = [0.03, 0.1, 0.2, 0.3, 0.5]
D1_G = [12, 24, 100, 400]
D1_N = [5, 25, 100]
D1_LAW = ["gauss", "t5"]
D1_TRAIN_D, D1_TEST_D = {32, 512, 2048}, {128, 1024}

# encoded extended space for D2: every coordinate in [0, 1]
D2_DIMS = ["log_d", "log_rho", "log_G", "log_n", "inv_nu", "alpha_pow", "log_kappa", "spike_frac", "K", "sep", "cv_n", "log_urank_frac"]


def d1_configs():
    u = qmc.LatinHypercube(d=5, seed=np.random.default_rng(401)).random(1000)
    levels = [D1_D, D1_RHO, D1_G, D1_N, D1_LAW]
    out = {}
    for i, row in enumerate(u):
        d, rho, G, n, law = (lv[min(int(x * len(lv)), len(lv) - 1)] for x, lv in zip(row, levels))
        out[f"d1|{i:04d}"] = dict(d=d, rho=rho, G=G, n=n, law=law, block="TRAIN" if d in D1_TRAIN_D else "TEST")
    return out


def decode(x):
    """Encoded vector in [0,1]^12 -> extended-space configuration (order: D2 (i))."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    d = int(round(np.exp(np.log(16) + x[0] * (np.log(2048) - np.log(16)))))
    rho = float(np.exp(np.log(0.005) + x[1] * (np.log(0.7) - np.log(0.005))))
    G = int(round(np.exp(np.log(6) + x[2] * (np.log(500) - np.log(6)))))
    n = int(round(np.exp(np.log(3) + x[3] * (np.log(300) - np.log(3)))))
    inv_nu = x[4] / 3.0
    nu = None if inv_nu < 1e-3 else float(1.0 / inv_nu)
    alpha = 1.5 * x[5]
    kappa = float(np.exp(x[6] * np.log(100.0)))
    r_spike = int(max(1, round(x[7] * min(20, d))))
    K = int(1 + round(x[8] * 9))
    sep = 6.0 * x[9]
    cv = 1.5 * x[10]
    u_rank = int(max(1, round(np.exp(x[11] * np.log(d)))))
    return dict(d=d, rho=rho, G=G, n=n, nu=nu, alpha=alpha, kappa=kappa, r_spike=r_spike, K=K, sep=sep, cv_n=cv, u_rank=u_rank)


def spectrum(cfg):
    d = cfg["d"]
    lam = np.arange(1, d + 1, dtype=float) ** -cfg["alpha"]
    lam[:cfg["r_spike"]] *= cfg["kappa"]
    return lam / lam.mean()


def d2_random():
    rng = np.random.default_rng(401)
    X = rng.random((3000, len(D2_DIMS)))
    return {f"d2r|{i:04d}": dict(x=X[i].tolist(), **decode(X[i])) for i in range(3000)}


def all_units(registry_csv):
    u = dict(d1_configs())
    u.update(d2_random())
    cells = [r["cell_id"] for r in csv.DictReader(open(registry_csv))]
    for c in cells:
        u[f"d3|{c}"] = dict(cell=c)
        u[f"d4|{c}"] = dict(cell=c)
    return u
