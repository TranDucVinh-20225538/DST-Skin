#!/usr/bin/env python3
"""T1 item E, GPU part: unit test, E1 simulated streams and E4 real-ICC-matched streams (10,000 replicates per cell,
alpha = 0.05, horizon 2,000 groups). E1 methods N1, N2, N3, G1; E4 methods N1, N2, G1.

    item_E1.py unittest                 -> results/t1/E/unit_test.json
    item_E1.py run LIST TASK NTASKS     -> results/t1/E/raw/egpu_<LIST>_<task>.jsonl
        LIST 0 = regime (a) work list (e1|a|..., e4|a|..., e4|ap|...), LIST 1 = regime (b) list (e1|b|..., e4|b|...),
        GPU units only, in the list's execution order; unit i of the filtered order -> task i % NTASKS.
Per replicate: group means and sizes from numpy default_rng(SeedSequence([20261010, 5, stable_hash(unit), rep])); the
samples from a torch CUDA generator seeded with the first 63 bits of the same SeedSequence's generate_state.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta as beta_dist, norm, spearmanr

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
CAP_S = 6 * 3600.0
E4_CAP_N = 1000


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


def lognormal_sizes(z, n):
    s2 = np.log(2.0)  # CV = 1
    return np.maximum(1, np.round(n * np.exp(np.sqrt(s2) * z - s2 / 2))).astype(int)


class E1Sampler:
    def __init__(self, p):
        self.p, self.outcome, self.n = p, p["outcome"], p["n"]
        self.ab = m_params(p["outcome"], p["rho"])

    def groups(self, rng, G):
        th, ab, n, s = theta_of(self.outcome), self.ab, self.n, self.p["size"]
        if s in ("b1", "b2"):
            k = 0.6 if s == "b1" else -0.6
            z1 = rng.standard_normal(G)
            z2 = k * z1 + np.sqrt(1 - k * k) * rng.standard_normal(G)
            return beta_dist.ppf(norm.cdf(z1), *ab), lognormal_sizes(z2, n)
        m = np.full(G, th) if ab is None else rng.beta(*ab, G)
        if s == "const":
            ng = np.full(G, n)
        elif s == "poisson":
            ng = 1 + rng.poisson(n - 1, G)
        elif s == "lognormal":
            ng = lognormal_sizes(rng.standard_normal(G), n)
        elif s == "b3":
            ng = np.clip(np.round(n * m / th), 1, 10 * n).astype(int)
        return m, ng

    def estimands(self):
        th = theta_of(self.outcome)
        if self.p["regime"] == "a":
            return th, th, 0.0
        rng = np.random.default_rng(501)
        num = den = 0.0
        for _ in range(10):
            m, ng = self.groups(rng, 1_000_000)
            num += float(np.sum(ng * m))
            den += float(np.sum(ng))
        m, ng = self.groups(rng, 200_000)
        r = ng * m - (num / den) * ng
        return th, num / den, float(np.sqrt(np.var(r, ddof=1) / 1e7) / np.mean(ng))


class E4Sampler:
    outcome = "flag"

    def __init__(self, p, slides):
        self.p = p
        self.m = np.array(slides["m_g"], float)
        self.N = np.minimum(np.array(slides["N_g"], int), E4_CAP_N)
        self.n = p.get("n") or float(self.N.mean())

    def groups(self, rng, G):
        j = rng.integers(0, len(self.m), G)
        if self.p["regime"] == "a":
            ng = np.full(G, self.p["n"])
        elif self.p["regime"] == "ap":
            ng = self.N[rng.integers(0, len(self.N), G)]
        else:
            ng = self.N[j]
        return self.m[j], ng

    def estimands(self):
        tg = float(self.m.mean())
        if self.p["regime"] == "b":
            return tg, float((self.N * self.m).sum() / self.N.sum()), 0.0
        return tg, tg, 0.0


def certify(radius, center, theta, start=0):
    out = {}
    for eps in EPS:
        okm = radius <= eps
        okm[:, :start] = False
        has = okm.any(1)
        j = okm.to(torch.uint8).argmax(1)
        c = center.gather(1, j[:, None])[:, 0]
        out[eps] = (has, (j + 1).double(), {k: (has & ((c - v).abs() > eps)) for k, v in theta.items()})
    return out


def draw_batch(uid, sampler, reps):
    groups = []
    for r in reps:
        ss = CM.seed_seq("E", uid, r)
        m, ng = sampler.groups(np.random.default_rng(ss), HORIZON)
        groups.append((m, ng, int(ss.generate_state(2, np.uint64)[1] & np.uint64(2 ** 63 - 1))))
    R, T = len(reps), max(int(g[1].sum()) for g in groups)
    x = torch.zeros(R, T, dtype=torch.float64, device=DEV)
    dm = torch.zeros(R, T, dtype=torch.bool, device=DEV)
    gsum = torch.zeros(R, HORIZON, dtype=torch.float64, device=DEV)
    gsq = torch.zeros(R, HORIZON, dtype=torch.float64, device=DEV)
    gen = torch.Generator(device=DEV)
    for i, (m, ng, s) in enumerate(groups):
        gen.manual_seed(s)
        ngt = torch.as_tensor(ng, device=DEV)
        mm = torch.repeat_interleave(torch.as_tensor(m, dtype=torch.float64, device=DEV), ngt)
        if sampler.outcome.startswith("flag"):
            y = (torch.rand(len(mm), generator=gen, dtype=torch.float64, device=DEV) < mm).double()
        else:
            g1 = torch._standard_gamma((mm * KAPPA_CONT).clamp_min(1e-12), generator=gen)
            g2 = torch._standard_gamma(((1 - mm) * KAPPA_CONT).clamp_min(1e-12), generator=gen)
            y = g1 / (g1 + g2)
        L = len(y)
        x[i, :L] = y
        ends = torch.cumsum(ngt, 0) - 1
        dm[i, ends] = True
        gid = torch.repeat_interleave(torch.arange(HORIZON, device=DEV), ngt)
        gsum[i].index_add_(0, gid, y)
        gsq[i].index_add_(0, gid, y * y)
    ngs = torch.as_tensor(np.stack([g[1] for g in groups]), dtype=torch.float64, device=DEV)
    return x, dm, gsum, gsq, ngs


def run_batch(uid, sampler, reps, theta, methods):
    x, dm, gsum, gsq, ng = draw_batch(uid, sampler, reps)
    gm = gsum / ng
    th = {k: torch.tensor(v, dtype=torch.float64, device=DEV) for k, v in theta.items()}
    thr = np.log(1 / ALPHA)
    res, rad = {}, {}
    lam = e_cs.prpl_lambda(x, ALPHA)
    for k, v in theta.items():
        res[f"N1|miscover|{k}"] = ((e_cs.logK_at(x, lam, v) >= thr) & dm).any(1)
    lo, hi = e_cs.grid_cs_mask(x, lam, dm, HORIZON, ALPHA)
    del x, dm, lam
    c1 = (lo + hi).double() / 2000.0
    r1 = (hi - lo).double() / 2000.0
    rad["N1"] = (r1, c1, 0)
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
    r2 = r1 * torch.sqrt(1 + (sampler.n - 1) * rho_hat)
    rad["N2"] = (r2, c1, 0)
    for k, v in th.items():
        res[f"N2|miscover|{k}"] = ((c1 - v).abs() > r2).any(1)
    if "N3" in methods:
        mu3, r3 = e_cs.asym_cs(gm, ALPHA)
        rad["N3"] = (r3, mu3, N3_SKIP)
        for k, v in th.items():
            res[f"N3|miscover|{k}"] = ((mu3 - v).abs() > r3)[:, N3_SKIP:].any(1)
    lamg = e_cs.prpl_lambda(gm, ALPHA)
    for k, v in theta.items():
        res[f"G1|miscover|{k}"] = (e_cs.logK_at(gm, lamg, v) >= thr).any(1)
    log_, hig = e_cs.grid_cs(gm, lamg, np.arange(HORIZON), ALPHA)
    rad["G1"] = ((hig - log_).double() / 2000.0, (log_ + hig).double() / 2000.0, 0)
    for name, (rr, cen, start) in rad.items():
        for eps, (has, gcert, fc) in certify(rr, cen, th, start).items():
            res[f"{name}|cert|eps{eps:g}"] = has
            res[f"{name}|groups_to_cert|eps{eps:g}"] = torch.where(has, gcert, torch.full_like(gcert, float("nan")))
            for k, f in fc.items():
                res[f"{name}|false_cert|eps{eps:g}|{k}"] = f
        res[f"{name}|width_at_100"] = 2 * rr[:, 99]
    out = {k: v.double().cpu().numpy() for k, v in res.items()}
    torch.cuda.empty_cache()
    return out


def run_cell(uid, sampler, methods, extra=None):
    t0 = time.time()
    th_g, th_s, th_s_se = sampler.estimands()
    base = dict(theta_group=th_g, theta_size=th_s, theta_size_se=th_s_se, **(extra or {}))
    status = "ok"
    if th_s_se >= 1e-4:
        status = "not run: theta_size MC SE >= 1e-4 (metrics below computed as exploratory only; never counted)"
    theta = {"theta_group": th_g, "theta_size": th_s}
    mean_n = float(sampler.n) * (3 if getattr(sampler, "p", {}).get("size") in ("lognormal", "b1", "b2", "b3") else 1)
    batch = max(10, min(1000, int(1e8 // (mean_n * HORIZON))))
    acc = {}
    for b0 in range(0, REPS, batch):
        for k, v in run_batch(uid, sampler, list(range(b0, min(REPS, b0 + batch))), theta, methods).items():
            acc.setdefault(k, []).append(v)
    acc = {k: np.concatenate(v) for k, v in acc.items()}
    res = dict(status=status, gap_over_eps={f"{e:g}": abs(th_s - th_g) / e for e in EPS}, reps=REPS,
               targeted={"N1": "theta_size", "N2": "theta_size", "N3": "theta_group", "G1": "theta_group"}, **base)
    for k, v in acc.items():
        ok = ~np.isnan(v)
        res[k] = float(np.nanmean(v)) if ok.any() else float("nan")
        res[k + "|se"] = float(np.nanstd(v, ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan")
    for name in methods:
        for eps in EPS:
            c = acc[f"{name}|cert|eps{eps:g}"]
            for k in theta:
                f = acc[f"{name}|false_cert|eps{eps:g}|{k}"]
                res[f"{name}|false_cert_given_cert|eps{eps:g}|{k}"] = float(f[c > 0].mean()) if (c > 0).any() else float("nan")
    res["seconds"] = time.time() - t0
    return res


def e4_slides(bb):
    """Per-slide flag rates m_g and unseen-eval counts N_g for camelyon_<bb>_s42 (cached in RAW)."""
    f = RAW / f"e4_slides_{bb}.json"
    if f.exists():
        return json.load(open(f))
    import scorer64 as S64
    from crossfit_ood import core as C
    cell = f"camelyon_{bb}_s42"
    d = CM.load_cell(cell)
    xtr, gtr, xid, gid, xood, fm = CM.paper2fold_inputs(d, 42)
    del xood, d
    tf, idf = C._apply_map(fm[0], gtr), C._apply_map(fm[0], gid)
    Xtr, Xid = S64.to_t(S64.l2n(xtr)), S64.to_t(S64.l2n(xid))
    del xtr, xid
    rows = {}
    for fo in (0, 1):
        fit = np.flatnonzero(tf == fo)
        gf = gtr[fit]
        F = Xtr[torch.as_tensor(fit, device=DEV)]
        qlogo = []
        for g in np.unique(gf):
            keep = torch.as_tensor(gf != g, device=DEV)
            lw = S64.LW(F[keep])
            qlogo.append(lw.q(F[~keep]))
            del lw
        tau = float(torch.quantile(torch.cat(qlogo).cpu(), 0.95))
        lw = S64.LW(F)
        uns = np.flatnonzero((idf >= 0) & (idf != fo))
        q = lw.q(Xid[torch.as_tensor(uns, device=DEV)]).cpu().numpy()
        for g in np.unique(gid[uns]):
            sel = gid[uns] == g
            rows[str(g)] = dict(fold_unseen_in=fo, m_g=float((q[sel] > tau).mean()), N_g=int(sel.sum()), tau=tau)
        del F, lw
    slides = sorted(rows)
    m = np.array([rows[s]["m_g"] for s in slides])
    N = np.array([rows[s]["N_g"] for s in slides])
    rng = np.random.default_rng(501)
    boot = []
    for _ in range(2000):
        j = rng.integers(0, len(slides), len(slides))
        boot.append(spearmanr(m[j], N[j]).statistic)
    boot = np.array(boot, float)
    res = dict(cell=cell, scorer="mahalanobis_l2 (Track A), flag = Q > 95th percentile of the fold's fit-LOGO Q (score below its 5th percentile)",
               slides=slides, m_g=m.tolist(), N_g=N.tolist(), per_slide=rows,
               spearman_m_N=float(spearmanr(m, N).statistic),
               spearman_ci95=[float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
               precommit=CM.precommit_hash())
    CM.atomic_write_text(f, json.dumps(res, indent=1))
    torch.cuda.empty_cache()
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


def spent_seconds():
    tot = 0.0
    for f in RAW.glob("egpu_*.jsonl"):
        for line in open(f):
            try:
                tot += float(json.loads(line).get("seconds", 0.0))
            except ValueError:
                pass
    return tot


def run_list(j, task, ntasks):
    CM.set_float64_torch()
    RAW.mkdir(parents=True, exist_ok=True)
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_E.json"))
    order = [u for u in add["work_lists"][j]["execution_order"] if u.startswith(("e1|", "e4|"))]
    params = add["unit_params"]
    mine = [u for i, u in enumerate(order) if i % ntasks == task]
    out = RAW / f"egpu_{j}_{task}.jsonl"
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
            p = params[uid]
            if uid.startswith("e1|"):
                res = run_cell(uid, E1Sampler(p), ("N1", "N2", "N3", "G1"), dict(deff=1 + (p["n"] - 1) * p["rho"]))
            else:
                t0 = time.time()
                sl = e4_slides(p["backbone"])
                prep = time.time() - t0
                res = run_cell(uid, E4Sampler(p, sl), ("N1", "N2", "G1"),
                               dict(spearman_m_N=sl["spearman_m_N"], spearman_ci95=sl["spearman_ci95"]))
                res["seconds"] += prep
            fh.write(json.dumps(dict(unit=uid, params=p, precommit=pre, **res)) + "\n")
            fh.flush()
            print(uid, f"{res.get('seconds', 0):.0f}s", res.get("N1|miscover|theta_size"), res.get("G1|miscover|theta_group"), flush=True)
    if "T1_MAXUNITS" not in os.environ:
        Path(str(out) + ".done").write_text("ok\n")


if __name__ == "__main__":
    if sys.argv[1] == "unittest":
        sys.exit(0 if unittest() else 1)
    run_list(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
