#!/usr/bin/env python3
"""T1 item F1 (CPU): re-implementation of the Gaussian reference of the quadratic-scorer sign law (bundle s3, s3b, s3c),
new draws from SeedSequence([20261010, 6, stable_hash(config), rep]). Writes results/t1/F/f1_ref.csv and f1.json.

Tolerances (order F1): sign accuracy within +-0.02 of the bundle value (per d x scorer); corr(AUROC, Phi(z)) is
reported against 0.995 (the bundle's own d = 8 Euclid value is 0.994, so it is not a gate); balanced means within +-0.005 of the bundle (s3c mean AUROC at d = 8/32/128 = .489/.491/.495,
CF-predicted .486/.490/.495; s3b mean AUROC of the blind scorer on its own balanced shifts and the two-block Maha AUROC).
"""
from __future__ import annotations

import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

OUT = CM.RES / "F"
REF_S3 = {8: dict(maha=(0.963, 0.995), euclid=(0.975, 0.994), vimres=(0.983, 0.995)),
          32: dict(maha=(0.988, 0.998), euclid=(0.983, 0.998), vimres=(0.979, 0.998)),
          128: dict(maha=(0.990, 1.000), euclid=(0.979, 1.000), vimres=(1.000, 1.000))}
N_S3 = {8: 240, 32: 240, 128: 96}
REF_S3C = {8: (0.489, 0.486), 32: (0.491, 0.490), 128: (0.495, 0.495)}
REF_S3B_BAL = {32: dict(maha=0.486, euclid=0.494, vimres=0.488), 128: dict(maha=0.491, euclid=0.496, vimres=0.493)}
REF_S3B_TWO = {(8, 1.3): 0.4959, (8, 1.6): 0.4826, (32, 1.3): 0.4964, (32, 1.6): 0.4894, (128, 1.3): 0.4990,
               (128, 1.6): 0.4955, (512, 1.3): 0.4994, (512, 1.6): 0.4974}
REF_S3B_TWO_CONTRAST = {(8, 1.3): 0.6603, (8, 1.6): 0.7950, (32, 1.3): 0.7997, (32, 1.6): 0.9508, (128, 1.3): 0.9539,
                        (128, 1.6): 0.9995, (512, 1.3): 0.9996, (512, 1.6): 1.0000}


def auroc(s_pos, s_neg):
    n, m = len(s_pos), len(s_neg)
    r = rankdata(np.r_[s_pos, s_neg])
    return (r[:n].sum() - n * (n + 1) / 2) / (n * m)


def qf(X, A):
    return np.einsum("ij,jk,ik->i", X, A, X)


def rng_of(cfg, rep):
    return np.random.default_rng(CM.seed_seq("F", cfg, rep))


def id_model(rng, d):
    lam = np.arange(1, d + 1.0) ** -1.0
    lam /= lam.mean()
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sig = (Q * lam) @ Q.T
    L = Q * np.sqrt(lam)
    ev, V = np.linalg.eigh(Sig)
    kv = max(2, d // 4)
    Vr = V[:, :d - kv]
    return Sig, L, {"maha": np.linalg.inv(Sig), "euclid": np.eye(d), "vimres": Vr @ Vr.T}


def s3_one(args):
    d, rep = args
    rng = rng_of(dict(f1="s3", d=d), rep)
    Sig, L, As = id_model(rng, d)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eta = rng.standard_normal() * 0.8
    c = np.exp(eta * rng.standard_normal(d) * 0.7 + rng.uniform(-0.6, 0.2))
    Cw = (U * c) @ U.T
    mw = rng.standard_normal(d)
    mw = mw / np.linalg.norm(mw) * rng.uniform(0, 1.5) * np.sqrt(d) * 0.25
    n = 6000
    X0 = rng.standard_normal((n, d)) @ L.T
    X1 = (mw + (rng.standard_normal((n, d)) * np.sqrt(c)) @ U.T) @ L.T
    M0, M1, C1, m1 = Sig, L @ (Cw + np.outer(mw, mw)) @ L.T, L @ Cw @ L.T, L @ mw
    out = {}
    for k, A in As.items():
        a = auroc(qf(X1, A), qf(X0, A))
        gap = np.trace(A @ (M1 - M0))
        AS, AC = A @ Sig, A @ C1
        s2 = 2 * np.trace(AS @ AS) + 2 * np.trace(AC @ AC) + 4 * m1 @ A @ C1 @ A @ m1
        out[k] = (a, gap / np.sqrt(s2))
    return out


def cum(A, C, m, j):
    t = np.trace(np.linalg.matrix_power(A @ C, j))
    q = m @ A @ np.linalg.matrix_power(C @ A, j - 1) @ m
    return 2 ** (j - 1) * math.factorial(j - 1) * (t + j * q)


def s3c_one(args):
    d, rep = args
    rng = rng_of(dict(f1="s3c", d=d), rep)
    n = 40000
    lam = np.arange(1, d + 1.0) ** -1.0
    lam /= lam.mean()
    Qm, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sig = (Qm * lam) @ Qm.T
    L = Qm * np.sqrt(lam)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    c = np.exp(rng.standard_normal() * 0.8 * rng.standard_normal(d) * 0.7 + rng.uniform(-0.3, 0.3))
    Cw = (U * c) @ U.T
    mw = rng.standard_normal(d)
    mw *= rng.uniform(0, 0.4) * np.sqrt(d) / np.linalg.norm(mw)
    A = np.linalg.inv(Sig)
    C1, m1, z0 = L @ Cw @ L.T, L @ mw, np.zeros(d)
    sc = (d - m1 @ A @ m1) / np.trace(A @ C1)
    if sc <= 0.2:
        return None
    C1 = C1 * sc
    X0 = rng.standard_normal((n, d)) @ L.T
    X1 = m1 + (rng.standard_normal((n, d)) * np.sqrt(c * sc)) @ U.T @ L.T
    a = auroc(qf(X1, A), qf(X0, A))
    k1 = cum(A, C1, m1, 1) - cum(A, Sig, z0, 1)
    k2 = cum(A, C1, m1, 2) + cum(A, Sig, z0, 2)
    k3 = cum(A, C1, m1, 3) - cum(A, Sig, z0, 3)
    s = np.sqrt(k2)
    z = k1 / s
    return a, norm.cdf(z), norm.cdf(z) + (k3 / (6 * s ** 3)) * (z ** 2 - 1) * norm.pdf(z)


def s3b_bal(args):
    d, rep, target = args
    rng = rng_of(dict(f1="s3b_bal", d=d, target=target), rep)
    Sig, L, As = id_model(rng, d)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    c0 = np.exp(0.7 * rng.standard_normal(d))
    Cw0 = (U * c0) @ U.T
    mw = rng.standard_normal(d)
    mw *= 0.3 * np.sqrt(d) / np.linalg.norm(mw)
    A = As[target]
    t = (np.trace(A @ Sig) - (L @ mw) @ A @ (L @ mw)) / np.trace(A @ L @ Cw0 @ L.T)
    if t <= 0.05:
        return None
    n = 8000
    X0 = rng.standard_normal((n, d)) @ L.T
    X1 = (mw + (rng.standard_normal((n, d)) * np.sqrt(c0 * t)) @ U.T) @ L.T
    return {k: auroc(qf(X1, B), qf(X0, B)) for k, B in As.items()}


def s3b_two(args):
    d, a2, rep = args
    rng = rng_of(dict(f1="s3b_two", d=d, a2=a2), rep)
    n = 20000
    c = np.r_[np.full(d // 2, a2), np.full(d - d // 2, 2 - a2)]
    X0 = rng.standard_normal((n, d))
    X1 = rng.standard_normal((n, d)) * np.sqrt(c)
    h = lambda X: (X[:, :d // 2] ** 2).sum(1) - (X[:, d // 2:] ** 2).sum(1)  # noqa: E731
    return auroc((X1 ** 2).sum(1), (X0 ** 2).sum(1)), auroc(h(X1), h(X0))


def main(nproc):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(nproc) as p:
        for d in (8, 32, 128):
            R = p.map(s3_one, [(d, r) for r in range(N_S3[d])])
            for k in ("maha", "euclid", "vimres"):
                a = np.array([r[k][0] for r in R])
                z = np.array([r[k][1] for r in R])
                acc = float(np.mean(np.sign(a - 0.5) == np.sign(z)))
                cor = float(np.corrcoef(a, norm.cdf(z))[0, 1])
                ref_acc, ref_cor = REF_S3[d][k]
                rows.append(dict(block="s3_sign_law", d=d, scorer=k, n=len(R), value=acc, reference=ref_acc, tol=0.02,
                                 pass_=abs(acc - ref_acc) <= 0.02, corr_auroc_phi_z=cor, corr_pass=cor >= 0.995,
                                 mae_auroc_phi_z=float(np.mean(np.abs(a - norm.cdf(z)))),
                                 frac_inverted=float(np.mean(a < 0.48)), frac_blind=float(np.mean(np.abs(a - 0.5) <= 0.02))))
        for d in (8, 32, 128):
            R = np.array([r for r in p.map(s3c_one, [(d, r) for r in range(48)]) if r is not None])
            a, f, cf = R.T
            rows.append(dict(block="s3c_balanced_mean_auroc", d=d, scorer="maha", n=len(R), value=float(a.mean()),
                             reference=REF_S3C[d][0], tol=0.005, pass_=abs(a.mean() - REF_S3C[d][0]) <= 0.005))
            rows.append(dict(block="s3c_cf_pred_mean", d=d, scorer="maha", n=len(R), value=float(cf.mean()),
                             reference=REF_S3C[d][1], tol=0.005, pass_=abs(cf.mean() - REF_S3C[d][1]) <= 0.005,
                             sign_agreement_cf=float(np.mean(np.sign(a - 0.5) == np.sign(cf - 0.5)))))
        for d in (32, 128):
            for tg in ("maha", "euclid", "vimres"):
                R = [r for r in p.map(s3b_bal, [(d, r, tg) for r in range(60)]) if r]
                v = float(np.mean([r[tg] for r in R]))
                rows.append(dict(block="s3b_balanced_own_scorer", d=d, scorer=tg, n=len(R), value=v,
                                 reference=REF_S3B_BAL[d][tg], tol=0.005, pass_=abs(v - REF_S3B_BAL[d][tg]) <= 0.005))
        for d in (8, 32, 128, 512):
            for a2 in (1.3, 1.6):
                R = np.array(p.map(s3b_two, [(d, a2, r) for r in range(12)])).mean(0)
                rows.append(dict(block="s3b_twoblock_maha", d=d, scorer="maha", a2=a2, n=12, value=float(R[0]),
                                 reference=REF_S3B_TWO[(d, a2)], tol=0.005, pass_=abs(R[0] - REF_S3B_TWO[(d, a2)]) <= 0.005))
                rows.append(dict(block="s3b_twoblock_contrast (report only)", d=d, scorer="half_contrast", a2=a2, n=12,
                                 value=float(R[1]), reference=REF_S3B_TWO_CONTRAST[(d, a2)], pass_="na"))
    from item_A_analysis import write_csv
    write_csv(OUT / "f1_ref.csv", rows)
    gate = [r for r in rows if r["pass_"] != "na"]
    ok = all(bool(r["pass_"]) for r in gate)
    res = dict(n_checks=len(gate), n_pass=sum(bool(r["pass_"]) for r in gate),
               corr_all_ge_0_995=all(r.get("corr_pass", True) for r in rows), pass_F1=bool(ok),
               failed=[r for r in gate if not r["pass_"]],
               corr_below_0_995=[(r["d"], r["scorer"], r["corr_auroc_phi_z"]) for r in rows if r.get("corr_pass") is False],
               precommit=CM.precommit_hash())
    CM.atomic_write_text(OUT / "f1.json", json.dumps(res, indent=1, default=lambda o: o.item()))
    print(json.dumps({k: v for k, v in res.items() if k != "failed"}, default=lambda o: o.item()), [(r["block"], r["d"], r["scorer"], r["value"], r["reference"]) for r in res["failed"]])


def diag_bundle_seeds(nproc):
    """Implementation check only (never the F-i gate): the same code with the bundle's seeds default_rng(0..n-1)."""
    global rng_of
    rng_of = lambda cfg, rep: np.random.default_rng(rep)  # noqa: E731
    rows = []
    with Pool(nproc) as p:
        for d in (8, 32, 128):
            R = p.map(s3_one, [(d, r) for r in range(N_S3[d])])
            for k in ("maha", "euclid", "vimres"):
                a = np.array([r[k][0] for r in R])
                z = np.array([r[k][1] for r in R])
                rows.append(dict(block="s3_sign_law", d=d, scorer=k, value=float(np.mean(np.sign(a - 0.5) == np.sign(z))),
                                 reference=REF_S3[d][k][0]))
        for d in (8, 32, 128):
            R = np.array([r for r in p.map(s3c_one, [(d, r) for r in range(48)]) if r is not None])
            rows.append(dict(block="s3c_balanced_mean_auroc", d=d, scorer="maha", value=float(R[:, 0].mean()), reference=REF_S3C[d][0]))
            rows.append(dict(block="s3c_cf_pred_mean", d=d, scorer="maha", value=float(R[:, 2].mean()), reference=REF_S3C[d][1]))
    from item_A_analysis import write_csv
    write_csv(OUT / "f1_diag_bundle_seeds.csv", rows)
    print(rows)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "diag":
        diag_bundle_seeds(int(sys.argv[1]))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
