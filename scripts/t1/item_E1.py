#!/usr/bin/env python3
"""T1 item E1: simulated streams, 10,000 replicates per cell, alpha = 0.05, horizon 2,000 groups; methods N1, N2, N3, G1.

    item_E1.py unittest                 -> results/t1/E/unit_test.json
    item_E1.py run LIST TASK NTASKS     -> results/t1/E/raw/e1_<LIST>_<task>.jsonl   (LIST a = regime (a), b = regime (b))
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta as beta_dist, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import e_cs  # noqa: E402

OUT = CM.RES / "E"
RAW = Path(os.environ.get("T1_RAW", str(OUT / "raw")))
DEV = e_cs.DEV
ALPHA, HORIZON, REPS = 0.05, 2000, int(os.environ.get("T1_E_REPS", "10000"))
EPS = (0.05, 0.10)
KAPPA_CONT = 10.0
N3_SKIP = 5


def theta_of(outcome):
    return 0.05 if outcome == "flag0.05" else 0.3


def m_params(outcome, rho):
    """Beta(a, b) of the group means m_g: ICC of the observed outcome = rho (flags: Var m / Var y; continuous score
    y | m ~ Beta(m kappa, (1-m) kappa), kappa = 10, ICC of y = 1/(1 + a'/(kappa+1)) with a' = a + b)."""
    th = theta_of(outcome)
    if rho == 0:
        return None
    ab = 1 / rho - 1 if outcome.startswith("flag") else (KAPPA_CONT + 1) * (1 / rho - 1)
    return th * ab, (1 - th) * ab


def lognormal_sizes(rng_or_z, n, size=None):
    s2 = np.log(2.0)  # CV = 1
    z = rng_or_z.standard_normal(size) if size is not None else rng_or_z
    return np.maximum(1, np.round(n * np.exp(np.sqrt(s2) * z - s2 / 2))).astype(int)


def draw_groups(rng, p, G):
    """(m_g, n_g) for G groups under the cell's size law / mechanism."""
    th = theta_of(p["outcome"])
    ab = m_params(p["outcome"], p["rho"])
    n, s = p["n"], p["size"]
    if s in ("b1", "b2"):
        k = 0.6 if s == "b1" else -0.6
        z1 = rng.standard_normal(G)
        z2 = k * z1 + np.sqrt(1 - k * k) * rng.standard_normal(G)
        m = beta_dist.ppf(norm.cdf(z1), *ab)
        return m, lognormal_sizes(z2, n)
    m = np.full(G, th) if ab is None else rng.beta(*ab, G)
    if s == "const":
        ng = np.full(G, n)
    elif s == "poisson":
        ng = 1 + rng.poisson(n - 1, G)
    elif s == "lognormal":
        ng = lognormal_sizes(rng, n, G)
    elif s == "b3":
        ng = np.clip(np.round(n * m / th), 1, 10 * n).astype(int)
    return m, ng


def draw_samples(rng, outcome, m, ng):
    mm = np.repeat(m, ng)
    if outcome.startswith("flag"):
        return (rng.random(len(mm)) < mm).astype(np.float64)
    return rng.beta(np.maximum(mm * KAPPA_CONT, 1e-12), np.maximum((1 - mm) * KAPPA_CONT, 1e-12))


def true_estimands(p):
    th = theta_of(p["outcome"])
    if p["regime"] == "a":
        return th, th, 0.0
    rng = np.random.default_rng(501)
    nm, nn = 0.0, 0.0
    num, den = [], []
    for _ in range(10):
        m, ng = draw_groups(rng, p, 1_000_000)
        num.append(np.sum(ng * m))
        den.append(np.sum(ng))
        nm += num[-1]
        nn += den[-1]
    m, ng = draw_groups(rng, p, 200_000)
    r = ng * m - (nm / nn) * ng
    se = float(np.sqrt(np.var(r, ddof=1) / 1e7) / np.mean(ng))
    return th, float(nm / nn), se


def certify(radius, center, theta, start=0):
    """First decision index >= start with radius <= eps; returns dict per eps (cert flag, groups, false-cert flag)."""
    out = {}
    for eps in EPS:
        okm = radius <= eps
        okm[:, :start] = False
        has = okm.any(1)
        j = okm.to(torch.uint8).argmax(1)
        c = center.gather(1, j[:, None])[:, 0]
        out[eps] = (has, (j + 1).double(), {k: (has & ((c - v).abs() > eps)) for k, v in theta.items()})
    return out


def run_batch(uid, p, reps, theta):
    rows = []
    xs, decs, gms = [], [], []
    stats = []
    for r in reps:
        rng = np.random.default_rng(CM.seed_seq("E", uid, r))
        m, ng = draw_groups(rng, p, HORIZON)
        y = draw_samples(rng, p["outcome"], m, ng)
        ends = np.cumsum(ng) - 1
        gsum = np.add.reduceat(y, np.r_[0, ends[:-1] + 1])
        gsq = np.add.reduceat(y * y, np.r_[0, ends[:-1] + 1])
        xs.append(y)
        decs.append(ends)
        gms.append(gsum / ng)
        stats.append((ng, gsum, gsq))
    T = max(len(v) for v in xs)
    R = len(reps)
    x = torch.zeros(R, T, dtype=torch.float64)
    dm = torch.zeros(R, T, dtype=torch.bool)
    for i, (v, e) in enumerate(zip(xs, decs)):
        x[i, :len(v)] = torch.from_numpy(v)
        dm[i, torch.from_numpy(e)] = True
    x, dm = x.to(DEV), dm.to(DEV)
    gm = torch.as_tensor(np.stack(gms), device=DEV)
    th = {k: torch.tensor(v, dtype=torch.float64, device=DEV) for k, v in theta.items()}
    res = {}
    # N1 (sample-level betting CS) and N2 (N1 radius x sqrt(1 + (n-1) rho_hat_t))
    lam = e_cs.prpl_lambda(x, ALPHA)
    thr = np.log(1 / ALPHA)
    for k, v in theta.items():
        lk = e_cs.logK_at(x, lam, v)
        res[f"N1|miscover|{k}"] = ((lk >= thr) & dm).any(1)
        del lk
    lo, hi = e_cs.grid_cs_mask(x, lam, dm, HORIZON, ALPHA)
    c1 = (lo + hi).double() / 2000.0
    r1 = (hi - lo).double() / 2000.0
    ng = torch.as_tensor(np.stack([s[0] for s in stats]), dtype=torch.float64, device=DEV)
    gsum = torch.as_tensor(np.stack([s[1] for s in stats]), device=DEV)
    gsq = torch.as_tensor(np.stack([s[2] for s in stats]), device=DEV)
    t = torch.arange(1, HORIZON + 1, dtype=torch.float64, device=DEV)
    Nt = torch.cumsum(ng, 1)
    St = torch.cumsum(gsum, 1)
    Bt = torch.cumsum(gsum ** 2 / ng, 1) - St ** 2 / Nt
    Wt = torch.cumsum(gsq - gsum ** 2 / ng, 1)
    msb = Bt / (t - 1).clamp_min(1)
    msw = Wt / (Nt - t).clamp_min(1)
    n0 = (Nt - torch.cumsum(ng ** 2, 1) / Nt) / (t - 1).clamp_min(1)
    rho_hat = ((msb - msw) / (msb + (n0 - 1) * msw).clamp_min(1e-300)).clamp(0, 1)
    rho_hat[:, 0] = 0.0
    r2 = r1 * torch.sqrt(1 + (p["n"] - 1) * rho_hat)
    for k, v in th.items():
        res[f"N2|miscover|{k}"] = ((c1 - v).abs() > r2).any(1)
    # N3 (group-level asymptotic CS, bundle s1b) and G1 (group-level betting CS)
    mu3, r3 = e_cs.asym_cs(gm, ALPHA)
    for k, v in th.items():
        res[f"N3|miscover|{k}"] = ((mu3 - v).abs() > r3)[:, N3_SKIP:].any(1)
    lamg = e_cs.prpl_lambda(gm, ALPHA)
    for k, v in theta.items():
        res[f"G1|miscover|{k}"] = (e_cs.logK_at(gm, lamg, v) >= thr).any(1)
    log_, hig = e_cs.grid_cs(gm, lamg, np.arange(HORIZON), ALPHA)
    cg = (log_ + hig).double() / 2000.0
    rg = (hig - log_).double() / 2000.0
    for name, rad, cen, start in (("N1", r1, c1, 0), ("N2", r2, c1, 0), ("N3", r3, mu3, N3_SKIP), ("G1", rg, cg, 0)):
        cert = certify(rad, cen, th, start)
        for eps, (has, gcert, fc) in cert.items():
            res[f"{name}|cert|eps{eps:g}"] = has
            res[f"{name}|groups_to_cert|eps{eps:g}"] = torch.where(has, gcert, torch.full_like(gcert, float("nan")))
            for k, f in fc.items():
                res[f"{name}|false_cert|eps{eps:g}|{k}"] = f
        res[f"{name}|width_at_100"] = 2 * rad[:, 99]
    out = {k: v.double().cpu().numpy() for k, v in res.items()}
    del x, dm, lam, gm
    torch.cuda.empty_cache()
    return out


def run_cell(uid, p, batch=None):
    t0 = time.time()
    th_g, th_s, th_s_se = true_estimands(p)
    if p["regime"] == "b" and th_s_se >= 1e-4:
        return dict(status="not run: theta_size MC SE >= 1e-4", theta_group=th_g, theta_size=th_s, theta_size_se=th_s_se)
    theta = {"theta_group": th_g, "theta_size": th_s}
    batch = batch or max(10, min(1000, int(2e7 // (p["n"] * HORIZON * (3 if p["size"] in ("lognormal", "b1", "b2", "b3") else 1)))))
    acc = {}
    for b0 in range(0, REPS, batch):
        o = run_batch(uid, p, list(range(b0, min(REPS, b0 + batch))), theta)
        for k, v in o.items():
            acc.setdefault(k, []).append(v)
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    res = dict(theta_group=th_g, theta_size=th_s, theta_size_se=th_s_se, gap_over_eps={f"{e:g}": abs(th_s - th_g) / e for e in EPS},
               deff=1 + (p["n"] - 1) * p["rho"], reps=REPS, seconds=time.time() - t0,
               targeted={"N1": "theta_size", "N2": "theta_size", "N3": "theta_group", "G1": "theta_group"})
    for k, v in acc.items():
        ok = ~np.isnan(v)
        res[k] = float(np.nanmean(v)) if ok.any() else float("nan")
        res[k + "|se"] = float(np.nanstd(v, ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan")
    for name in ("N1", "N2", "N3", "G1"):
        for eps in EPS:
            c = acc[f"{name}|cert|eps{eps:g}"]
            for k in theta:
                f = acc[f"{name}|false_cert|eps{eps:g}|{k}"]
                res[f"{name}|false_cert_given_cert|eps{eps:g}|{k}"] = float(f[c > 0].mean()) if (c > 0).any() else float("nan")
    return res


def unittest():
    rng = np.random.default_rng(CM.seed_seq("E", "unit_test", 0))
    x = torch.as_tensor(rng.random((10000, 2000)) < 0.3, dtype=torch.float64, device=DEV)
    lam = e_cs.prpl_lambda(x, ALPHA)
    mis = float(e_cs.ever_miscover(x, lam, 0.3, np.arange(2000), ALPHA).double().mean())
    se = float(np.sqrt(ALPHA * (1 - ALPHA) / 10000))
    res = dict(test="iid Bernoulli(0.3), 10,000 replicates, horizon 2,000, every time a decision time", alpha=ALPHA,
               ever_miscover=mis, bound=ALPHA + 3 * se, pass_=bool(mis <= ALPHA + 3 * se),
               implementation="own implementation of the hedged-capital betting CS with predictable plug-in empirical-Bernstein bets "
                              "(Waudby-Smith & Ramdas, arXiv 2010.09686), 1,001-point grid; confseq not installed",
               precommit=CM.precommit_hash())
    CM.atomic_write_text(OUT / "unit_test.json", json.dumps(res, indent=1))
    print(res)
    return res["pass_"]


CAP_S = 6 * 3600.0


def spent_seconds():
    tot = 0.0
    for f in RAW.glob("*.jsonl"):
        for line in open(f):
            try:
                tot += float(json.loads(line).get("seconds", 0.0))
            except ValueError:
                pass
    return tot


def run_list(lst, task, ntasks):
    CM.set_float64_torch()
    RAW.mkdir(parents=True, exist_ok=True)
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_E.json"))
    j = 0 if lst == "a" else 1
    order = [u for u in add["work_lists"][j]["execution_order"] if u.startswith(f"e1|{lst}|")]
    params = add["unit_params"]
    mine = [u for i, u in enumerate(order) if i % ntasks == task]
    out = RAW / f"e1_{lst}_{task}.jsonl"
    have = set()
    if out.exists():
        with open(out, "rb+") as fh:
            data = fh.read()
            if data and not data.endswith(b"\n"):
                fh.truncate(data.rfind(b"\n") + 1)
        have = {json.loads(line)["unit"] for line in open(out)}
    pre = CM.precommit_hash()
    with open(out, "a") as fh:
        for uid in [u for u in mine if u not in have][:int(os.environ.get("T1_MAXUNITS", "100000000"))]:
            if spent_seconds() >= CAP_S:
                print("E cap reached", flush=True)
                return
            res = run_cell(uid, params[uid])
            fh.write(json.dumps(dict(unit=uid, params=params[uid], precommit=pre, **res)) + "\n")
            fh.flush()
            print(uid, f"{res.get('seconds', 0):.0f}s", res.get("N1|miscover|theta_size"), res.get("G1|miscover|theta_group"), flush=True)
    if "T1_MAXUNITS" not in os.environ:
        Path(str(out) + ".done").write_text("ok\n")


if __name__ == "__main__":
    if sys.argv[1] == "unittest":
        sys.exit(0 if unittest() else 1)
    run_list(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
