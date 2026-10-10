#!/usr/bin/env python3
"""T1 item C0: reproduction of the bundle toy cells in the new float64 GPU code, and seconds per replicate by (d, N).

Each bundle design is re-implemented as written in the bundle script (`sims/s2a_maha.py`, `s2b_auroc.py`, `s2d_cap.py`,
`s2f_null.py`); draws from SeedSequence([20261010, 3, stable_hash(unit), rep]). Pass per row: |new - bundle| <= 3 combined SE,
combined SE = sqrt(SE_new^2 + SE_bundle^2); SE_bundle is the bundle's printed SE where it printed one for the compared
quantity (s2a), otherwise the new per-replicate SD / sqrt(bundle replicate count) (s2b: 16, s2d: 40 / 12, s2f: 8).

    item_C0.py   -> results/t1/C/c0.json, results/t1/C/c0_rows.csv, results/t1/C/c0_seconds_per_rep.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import c_design  # noqa: E402
import scorer64 as S64  # noqa: E402
import sim_c  # noqa: E402
import sim_re  # noqa: E402

OUT = CM.RES / "C"
DEV = S64.DEV
R = 64

S2A_REF = {(128, 40, 10, 0.6, 0.0): (-170.27, 0.66, -168.93), (128, 100, 5, 0.3, 0.1): (-12.73, 0.23, -12.99),
           (32, 40, 10, 0.6, 0.0): (-10.89, 0.20, -10.43)}
S2B_REF = {(64, 100, 5, 0.2, 0.0): 0.019, (64, 100, 5, 0.2, 0.1): 0.017, (64, 100, 5, 0.4, 0.0): 0.064, (64, 100, 5, 0.4, 0.1): 0.059,
           (128, 100, 5, 0.2, 0.0): 0.044, (128, 100, 5, 0.2, 0.1): 0.040, (128, 100, 5, 0.4, 0.0): 0.113, (128, 100, 5, 0.4, 0.1): 0.102,
           (128, 200, 10, 0.2, 0.0): 0.025, (128, 200, 10, 0.2, 0.1): 0.023, (128, 200, 10, 0.4, 0.0): 0.073, (128, 200, 10, 0.4, 0.1): 0.068,
           (256, 200, 10, 0.2, 0.0): 0.050, (256, 200, 10, 0.2, 0.1): 0.046, (256, 200, 10, 0.4, 0.0): 0.106, (256, 200, 10, 0.4, 0.1): 0.100}
S2D_REF = {(1, 0.5, 3): 0.172, (1, 0.5, 10): 0.096, (1, 0.5, 30): 0.058, (1, 0.5, 100): 0.033,
           (1, 0.9, 3): 0.377, (1, 0.9, 10): 0.208, (1, 0.9, 30): 0.130, (1, 0.9, 100): 0.071,
           (1, 0.99, 3): 0.685, (1, 0.99, 10): 0.417, (1, 0.99, 30): 0.262, (1, 0.99, 100): 0.139,
           (2, 0.5, 3): 0.237, (2, 0.5, 10): 0.172, (2, 0.5, 30): 0.095, (2, 0.5, 100): 0.051, (2, 0.5, 300): 0.030,
           (2, 0.9, 3): 0.637, (2, 0.9, 10): 0.438, (2, 0.9, 30): 0.284, (2, 0.9, 100): 0.154, (2, 0.9, 300): 0.092,
           (2, 0.99, 3): 0.936, (2, 0.99, 10): 0.841, (2, 0.99, 30): 0.673, (2, 0.99, 100): 0.460, (2, 0.99, 300): 0.286}
S2F_REF = {(16, 0.3): 0.025, (256, 0.1): 0.042, (16, 0.5): 0.111, (64, 0.3): 0.164, (64, 0.5): 0.476, (256, 0.3): 0.489}


def t(a):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=DEV)


def auroc_pos(s_pos, s_neg):
    """bundle common.auroc: P(s_pos > s_neg) + 0.5 ties (pos = OOD, higher = more anomalous)."""
    return S64.auroc_id_pos(np.asarray(s_pos), np.asarray(s_neg))


def gen_groups(rng, G, n, d, rho):
    u = rng.standard_normal((G, d)) * np.sqrt(rho)
    X = np.repeat(u, n, axis=0) + rng.standard_normal((G * n, d)) * np.sqrt(1 - rho)
    return X, u


def draw_from_groups(rng, u, gidx, rho):
    return u[gidx] + rng.standard_normal((len(gidx), u.shape[1])) * np.sqrt(1 - rho)


def s2b_one(d, G, n, rho, lam, a2, ss):
    """s2b_auroc.one, Mahalanobis part (the uo2 / placeholder draws are kept so the draw sequence matches the script)."""
    rng = np.random.default_rng(ss)
    X, u = gen_groups(rng, G, n, d, rho)
    N = G * n
    mu = X.mean(0)
    Xc = t(X - mu)
    S = Xc.T @ Xc / N + lam * torch.eye(d, dtype=torch.float64, device=DEV)
    P = torch.linalg.inv(S)
    m = 3000
    Xin = draw_from_groups(rng, u, rng.integers(0, G, m), rho) - mu
    uo = rng.standard_normal((600, d)) * np.sqrt(rho)
    Xout = draw_from_groups(rng, uo, rng.integers(0, 600, m), rho) - mu
    uo2 = rng.standard_normal((600, d)) * np.sqrt(rho * a2)
    _ = draw_from_groups(rng, uo2 / np.sqrt(a2) * np.sqrt(a2), rng.integers(0, 600, m), rho)
    Xood = np.sqrt(a2) * draw_from_groups(rng, uo, rng.integers(0, 600, m), rho) - mu

    def q(Z):
        Zt = t(Z)
        return ((Zt @ P) * Zt).sum(1).cpu().numpy()

    qi, qo, qd = q(Xin), q(Xout), q(Xood)
    return auroc_pos(qd, qi), auroc_pos(qd, qo)


def s2d_tv(rng, d, G, rho):
    u = rng.standard_normal((G, d)) * np.sqrt(rho)
    s = np.sqrt(1 - rho)
    if d == 1:
        grid = t(np.linspace(-9, 9, 6001))
        ut = t(u[:, 0])
        pin = (torch.exp(-(grid[:, None] - ut[None, :]) ** 2 / (2 * s * s)) / (np.sqrt(2 * np.pi) * s)).mean(1)
        pout = torch.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
        return float(0.5 * (pin - pout).abs().sum() * (grid[1] - grid[0]))
    g = t(np.linspace(-7, 7, 421))
    X, Y = torch.meshgrid(g, g, indexing="ij")
    P = torch.zeros_like(X)
    for a in u:
        P += torch.exp(-((X - a[0]) ** 2 + (Y - a[1]) ** 2) / (2 * s * s)) / (2 * np.pi * s * s)
    P /= G
    Q = torch.exp(-(X ** 2 + Y ** 2) / 2) / (2 * np.pi)
    return float(0.5 * (P - Q).abs().sum() * (g[1] - g[0]) ** 2)


def s2f_one(d, G, n, rho, ss):
    rng = np.random.default_rng(ss)
    X, u = gen_groups(rng, G, n, d, rho)
    m = 1500
    E = draw_from_groups(rng, u, rng.integers(0, G, m), rho)
    uo = rng.standard_normal((800, d)) * np.sqrt(rho)
    F = draw_from_groups(rng, uo, rng.integers(0, 800, m), rho)
    Xt = t(X)
    xn = (Xt * Xt).sum(1)

    def k(Z):
        Zt = t(Z)
        d2 = ((Zt * Zt).sum(1, keepdim=True) - 2 * Zt @ Xt.T + xn[None, :]).clamp_min(0.0)
        return torch.sqrt(d2.min(1).values).cpu().numpy()

    return auroc_pos(k(F), k(E)) - 0.5


def row(unit, kind, new, se_new, ref, se_ref, se_ref_source, reps, extra=None):
    comb = float(np.hypot(se_new, se_ref))
    r = dict(unit=unit, kind=kind, new=new, se_new=se_new, bundle=ref, se_bundle=se_ref, se_bundle_source=se_ref_source,
             combined_se=comb, z=(new - ref) / comb if comb > 0 else float("nan"), replicates=reps,
             pass_=bool(abs(new - ref) <= 3 * comb))
    if extra:
        r.update(extra)
    return r


def main():
    CM.set_float64_torch()
    OUT.mkdir(parents=True, exist_ok=True)
    units = c_design.all_units()
    rows, times = [], []
    t_start = time.time()
    for uid, p in units.items():
        if not uid.startswith("c0|"):
            continue
        k = p["kind"]
        t0 = time.time()
        if k == "s2a":
            key = (p["d"], p["G"], p["n"], p["rho"], p["lam"])
            reps = np.array([sim_re.sim_maha_dq(*key, CM.seed_seq("C", uid, r)) for r in range(R)])
            se_in, se_out = reps.std(0) / np.sqrt(R)
            dq_new = float(reps[:, 0].mean() - reps[:, 1].mean())
            ref, ref_se, th_b = S2A_REF[key]
            _, th_new = sim_re.k2_theory(*key)
            rows.append(row(uid, k, dq_new, float(np.hypot(se_in, se_out)), ref, ref_se, "bundle printed", R,
                            dict(theory_new=th_new, theory_bundle=th_b, theory_rel_diff=abs(th_new / th_b - 1))))
        elif k == "s2b":
            key = (p["d"], p["G"], p["n"], p["rho"], p["lam"])
            a2 = 1 + 3.0 / np.sqrt(p["d"])
            reps = np.array([s2b_one(*key, a2, CM.seed_seq("C", uid, r)) for r in range(R)])
            dl = reps[:, 0] - reps[:, 1]
            sd = float(dl.std(ddof=1))
            rows.append(row(uid, k, float(dl.mean()), sd / np.sqrt(R), S2B_REF[key], sd / np.sqrt(16), "new SD / sqrt(16)", R))
        elif k == "s2d":
            key = (p["d"], p["rho"], p["G"])
            rng = np.random.default_rng(CM.seed_seq("C", uid, 0))
            tv = np.array([s2d_tv(rng, p["d"], p["G"], p["rho"]) for _ in range(R)])
            nb = 40 if p["d"] == 1 else 12
            sd = float(tv.std(ddof=1))
            rows.append(row(uid, k, float(tv.mean()), sd / np.sqrt(R), S2D_REF[key], sd / np.sqrt(nb), f"new SD / sqrt({nb})", R))
        elif k == "s2f":
            key = (p["d"], p["rho"])
            v = np.array([s2f_one(p["d"], p["G"], p["n"], p["rho"], CM.seed_seq("C", uid, r)) for r in range(R)])
            sd = float(v.std(ddof=1))
            rows.append(row(uid, k, float(v.mean()), sd / np.sqrt(R), S2F_REF[key], sd / np.sqrt(8), "new SD / sqrt(8)", R,
                            dict(rho_sqrt_d=p["rho"] * np.sqrt(p["d"]))))
        elif k == "time":
            cfg = dict(d=p["d"], rho=0.3, n_g=[p["n"]] * p["G"])
            sim_c.replicate(cfg, CM.seed_seq("C", uid, 0))
            torch.cuda.synchronize()
            t1 = time.time()
            r = sim_c.replicate(cfg, CM.seed_seq("C", uid, 1))
            torch.cuda.synchronize()
            times.append(dict(unit=uid, d=p["d"], G=p["G"], n=p["n"], N=p["G"] * p["n"], seconds_per_rep=time.time() - t1,
                              seconds_internal=r["seconds"]))
        print(uid, f"{time.time() - t0:.1f}s", rows[-1]["pass_"] if k != "time" else times[-1]["seconds_per_rep"], flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "c0_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "c0_seconds_per_rep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(times[0]))
        w.writeheader()
        w.writerows(times)
    th_ok = all(r["theory_rel_diff"] <= 0.01 for r in rows if r["kind"] == "s2a")
    summ = dict(n_rows=len(rows), n_pass=sum(r["pass_"] for r in rows), failed=[r["unit"] for r in rows if not r["pass_"]],
                s2a_theory_within_1pct=th_ok, pass_C0=bool(all(r["pass_"] for r in rows) and th_ok),
                seconds=time.time() - t_start, gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
                precommit=CM.precommit_hash())
    CM.atomic_write_text(OUT / "c0.json", json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
