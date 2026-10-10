#!/usr/bin/env python3
"""T1 item B (per cell): kNN leakage on real features for both paper_2fold folds.

One exact float64 top-200 cosine search per (fit, eval set); knn_mean_cosine and k-th-distance scores for
k in {1, 5, 10, 20, 50, 100, 200}; Tier-L descriptors (rho_cos, PR, TwoNN, n_g), r_NN(k), T_unseen;
matched toy simulator (MTS, 16 replicates).

    item_B_cell.py CELL [CELL ...]  -> results/t1/B/raw/b_<cell>.json  (skips cells already done)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import scorer64 as S64  # noqa: E402

torch = S64.torch
RAW = Path(os.environ.get("T1_RAW", str(CM.RES / "B" / "raw")))
REG = {r["cell_id"]: r for r in csv.DictReader(open(CM.RES / "I" / "registry.csv"))}
R3 = {(r["cell"], r["scorer"]): r for r in csv.DictReader(open(CM.REPO / "results/r3/1/paper_ci_v2.csv"))
      if r["seen"] == "strict"}
KS = [1, 5, 10, 20, 50, 100, 200]
KMAX = 200
CHUNK = 2048
MTS_REPS = 16
MTS_M = 4000
MTS_ITERS = 30
MTS_LOGA = (np.log(1e-2), np.log(1e2))


def unit(x):
    """Track A preprocessing x / (||x|| + 1e-8), then exact unit vectors (sklearn metric='cosine' renormalises)."""
    t = S64.to_t(S64.l2n(x))
    return t / torch.linalg.norm(t, dim=1, keepdim=True)


def topk_sim(Q, F, k, exclude_self=False):
    vals, idx = [], []
    for i in range(0, Q.shape[0], CHUNK):
        s = Q[i:i + CHUNK] @ F.T
        if exclude_self:
            r = torch.arange(i, min(i + CHUNK, Q.shape[0]), device=s.device)
            s[r - i, r] = -float("inf")
        v, j = torch.topk(s, k, dim=1, largest=True)
        vals.append(v)
        idx.append(j)
    return torch.cat(vals), torch.cat(idx)


def knn_scores(sim, k):
    """(mean-over-k score, k-th-distance score); higher = more ID."""
    dist = 1.0 - sim[:, :k]
    return (-dist.mean(1)).cpu().numpy(), (-dist[:, k - 1]).cpu().numpy()


def twonn(F, seed):
    """Facco et al. (2017) TwoNN on a 20,000-point uniform subsample: mu = r2/r1, linear fit of -log(1-F_emp)
    on log(mu) through the origin after discarding the largest 10 % of mu."""
    rng = np.random.default_rng(seed)
    n = F.shape[0]
    sub = F[torch.as_tensor(np.sort(rng.choice(n, min(20000, n), replace=False)), device=F.device)]
    d2 = []
    for i in range(0, sub.shape[0], CHUNK):
        s = sub[i:i + CHUNK] @ sub.T
        r = torch.arange(i, min(i + CHUNK, sub.shape[0]), device=s.device)
        s[r - i, r] = -float("inf")
        v, _ = torch.topk(s, 2, dim=1)
        d2.append((2.0 - 2.0 * v).clamp_min(0.0))
    d2 = torch.cat(d2).cpu().numpy()
    r1, r2 = np.sqrt(d2[:, 0]), np.sqrt(d2[:, 1])
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    m = len(mu)
    Femp = np.arange(1, m + 1) / m
    keep = slice(0, int(np.floor(0.9 * m)))
    x, y = np.log(mu[keep]), -np.log(1.0 - Femp[keep])
    return float((x @ y) / (x @ x)), int(m)


def t_stat(E, F, seed_e, seed_f):
    """K5 / item D statistic on unit vectors (squared Euclidean NN distances): eval subsample m = min(5000, |E|),
    fit->fit leave-one-out distances for a uniform 20,000-point subsample of F against the full fit set."""
    re, rf = np.random.default_rng(seed_e), np.random.default_rng(seed_f)
    ie = np.sort(re.choice(E.shape[0], min(5000, E.shape[0]), replace=False))
    if_ = np.sort(rf.choice(F.shape[0], min(20000, F.shape[0]), replace=False))
    ve, _ = topk_sim(E[torch.as_tensor(ie, device=E.device)], F, 1)
    de = (2.0 - 2.0 * ve[:, 0]).clamp_min(0.0).cpu().numpy()
    Fs = F[torch.as_tensor(if_, device=F.device)]
    vf = []
    for i in range(0, Fs.shape[0], CHUNK):
        s = Fs[i:i + CHUNK] @ F.T
        rows = torch.arange(s.shape[0], device=s.device)
        s[rows, torch.as_tensor(if_[i:i + CHUNK], device=s.device)] = -float("inf")
        vf.append(torch.topk(s, 1, dim=1).values[:, 0])
    df = (2.0 - 2.0 * torch.cat(vf)).clamp_min(0.0).cpu().numpy()
    T = (de.mean() - df.mean()) / np.sqrt(de.var(ddof=1) / len(de) + df.var(ddof=1) / len(df))
    return float(T), int(len(de)), int(len(df))


# ---------------------------------------------------------------- matched toy simulator
def _gen(seed_int):
    g = torch.Generator(device=S64.DEV)
    g.manual_seed(int(seed_int))
    return g


def _toy_knn_mean(Qs, F, ks):
    """Toy kNN-k score (Euclidean mean distance to the k nearest fit points; higher = more ID) for every k."""
    fn = (F * F).sum(1)
    kmax = max(ks)
    out = []
    for i in range(0, Qs.shape[0], CHUNK):
        q = Qs[i:i + CHUNK]
        d2 = ((q * q).sum(1, keepdim=True) - 2.0 * q @ F.T + fn[None, :]).clamp_min(0.0)
        v = torch.sqrt(torch.topk(d2, kmax, dim=1, largest=False).values)
        out.append(torch.stack([-v[:, :k].mean(1) for k in ks], 1))
    return torch.cat(out)


def _auroc_t(s_id, s_ood):
    """AUROC per column (higher = more ID) for score matrices (m x K) on GPU; ties counted 1/2."""
    res = []
    for j in range(s_id.shape[1]):
        res.append(S64.auroc_id_pos(s_id[:, j].cpu().numpy(), s_ood[:, j].cpu().numpy()))
    return np.array(res)


class ToyRep:
    """One MTS replicate: fit set with the real group-size vector, seen / unseen eval, fixed OOD directions z."""

    def __init__(self, n_g, d, rho, ks, seed_int):
        self.g = _gen(seed_int)
        self.n_g, self.d, self.rho, self.ks = np.asarray(n_g), d, rho, [k for k in ks if k <= int(np.sum(n_g))]
        self.seed_int = seed_int
        F, u = self._fit()
        G = len(self.n_g)
        gi = torch.randint(0, G, (MTS_M,), device=S64.DEV, generator=self.g)
        seen = u[gi] + np.sqrt(1 - rho) * torch.randn(MTS_M, d, device=S64.DEV, dtype=torch.float64, generator=self.g)
        unseen = torch.randn(MTS_M, d, device=S64.DEV, dtype=torch.float64, generator=self.g)
        self.z = torch.randn(MTS_M, d, device=S64.DEV, dtype=torch.float64, generator=self.g)
        self.s_seen = _toy_knn_mean(seen, F, self.ks)
        self.s_unseen = _toy_knn_mean(unseen, F, self.ks)
        del F, u, seen, unseen

    def _fit(self):
        g = _gen(self.seed_int + 1)
        G, d, rho = len(self.n_g), self.d, self.rho
        u = np.sqrt(rho) * torch.randn(G, d, device=S64.DEV, dtype=torch.float64, generator=g)
        idx = torch.as_tensor(np.repeat(np.arange(G), self.n_g), device=S64.DEV)
        F = u[idx] + np.sqrt(1 - rho) * torch.randn(len(idx), d, device=S64.DEV, dtype=torch.float64, generator=g)
        return F, u

    def auroc_at(self, a_per_k):
        """AUROC_seen and AUROC_unseen per k with OOD = a_k * z (isotropic scale), fit set regenerated identically."""
        F, _ = self._fit()
        a_s, a_u = [], []
        for j, k in enumerate(self.ks):
            so = _toy_knn_mean(a_per_k[j] * self.z, F, [k])
            a_s.append(S64.auroc_id_pos(self.s_seen[:, j].cpu().numpy(), so[:, 0].cpu().numpy()))
            a_u.append(S64.auroc_id_pos(self.s_unseen[:, j].cpu().numpy(), so[:, 0].cpu().numpy()))
        del F
        return np.array(a_s), np.array(a_u)


def mts(cell, fold, n_g, pr, rho, target_unseen):
    d = int(min(round(pr), 1024))
    ks = [k for k in KS if k <= int(np.sum(n_g))]
    reps = []
    for r in range(MTS_REPS):
        seed_int = int(CM.seed_seq("B", dict(cell=cell, fold=fold, what="mts"), r).generate_state(1)[0])
        reps.append(ToyRep(n_g, d, rho, ks, seed_int))
    tgt = np.array([target_unseen[k] for k in ks])
    lo = np.full(len(ks), MTS_LOGA[0])
    hi = np.full(len(ks), MTS_LOGA[1])

    def mean_auc(loga):
        res = [rp.auroc_at(np.exp(loga)) for rp in reps]
        return np.mean([x[0] for x in res], 0), np.mean([x[1] for x in res], 0)

    _, u_lo = mean_auc(lo)
    _, u_hi = mean_auc(hi)
    clipped_low, clipped_high = tgt <= u_lo, tgt >= u_hi
    for _ in range(MTS_ITERS):
        mid = 0.5 * (lo + hi)
        _, u_mid = mean_auc(mid)
        up = u_mid < tgt
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    loga = np.where(clipped_low, MTS_LOGA[0], np.where(clipped_high, MTS_LOGA[1], 0.5 * (lo + hi)))
    a_s, a_u = mean_auc(loga)
    del reps
    torch.cuda.empty_cache()
    return dict(d_toy=d, rho_toy=rho, ks=ks, ood_scale_a=np.exp(loga).tolist(), toy_auroc_seen=a_s.tolist(),
                toy_auroc_unseen=a_u.tolist(), target_auroc_unseen=tgt.tolist(), delta_mts=(a_s - a_u).tolist(),
                bracket_clipped_low=clipped_low.tolist(), bracket_clipped_high=clipped_high.tolist(),
                toy_knn="Euclidean mean distance to the k nearest fit points (see DEVIATIONS)", reps=MTS_REPS, m_each=MTS_M)


def run_cell(cell):
    out = RAW / f"b_{cell}.json"
    if out.exists() and out.stat().st_size > 0:
        print("done", cell)
        return
    t0 = time.time()
    CM.set_float64_torch()
    reg = REG[cell]
    seed = int(reg["seed"])
    d = CM.load_cell(cell)
    xtr, gtr, xid, gid, xood, fm = CM.paper2fold_inputs(d, seed)
    from crossfit_ood import core as C
    tf, idf = C._apply_map(fm[0], gtr), C._apply_map(fm[0], gid)
    Utr, Uid, Uood = unit(xtr), unit(xid), unit(xood)
    del xtr, xid, xood, d
    folds = []
    for f in (0, 1):
        fit = tf == f
        F = Utr[torch.as_tensor(fit, device=S64.DEV)]
        gf = gtr[fit]
        N, dim = F.shape
        seen = idf == f
        unseen = (idf >= 0) & (idf != f)
        km = min(KMAX, N)
        sim_s, idx_s = topk_sim(Uid[torch.as_tensor(seen, device=S64.DEV)], F, km)
        sim_u, _ = topk_sim(Uid[torch.as_tensor(unseen, device=S64.DEV)], F, km)
        sim_o, _ = topk_sim(Uood, F, km)
        own = torch.as_tensor(np.searchsorted(np.unique(gf), gid[seen]), device=S64.DEV)
        gf_code = torch.as_tensor(np.searchsorted(np.unique(gf), gf), device=S64.DEV)
        same = (gf_code[idx_s] == own[:, None]).double()
        perk = {}
        for k in [k for k in KS if k <= N]:
            ms, ks_ = knn_scores(sim_s, k)
            mu_, ku = knn_scores(sim_u, k)
            mo, ko = knn_scores(sim_o, k)
            a_s, a_u = S64.auroc_id_pos(ms, mo), S64.auroc_id_pos(mu_, mo)
            b_s, b_u = S64.auroc_id_pos(ks_, ko), S64.auroc_id_pos(ku, ko)
            perk[k] = dict(auroc_seen=a_s, auroc_unseen=a_u, delta=a_s - a_u,
                           kth_auroc_seen=b_s, kth_auroc_unseen=b_u, kth_delta=b_s - b_u,
                           r_nn=float(same[:, :k].sum(1).div(k).mean()))
        del sim_s, sim_u, sim_o, idx_s, same
        st, u, cnt = S64.group_stats(F, gf, torch.eye(dim, dtype=torch.float64, device=S64.DEV))
        Fc = F - F.mean(0)
        T = Fc.T @ Fc / (N - 1)
        pr = float(torch.trace(T) ** 2 / (T * T).sum())
        del Fc, T
        tw_seed = int(CM.seed_seq("B", dict(cell=cell, fold=f, what="twonn")).generate_state(1)[0])
        twonn_d, twonn_m = twonn(F, tw_seed)
        sd = dict(cell=cell, fold=f, what="T")
        T_unseen, m_e, m_f = t_stat(Uid[torch.as_tensor(unseen, device=S64.DEV)], F,
                                    int(CM.seed_seq("B", dict(sd, set="unseen")).generate_state(1)[0]),
                                    int(CM.seed_seq("B", dict(sd, set="fit")).generate_state(1)[0]))
        T_seen, _, _ = t_stat(Uid[torch.as_tensor(seen, device=S64.DEV)], F,
                              int(CM.seed_seq("B", dict(sd, set="seen")).generate_state(1)[0]),
                              int(CM.seed_seq("B", dict(sd, set="fit")).generate_state(1)[0]))
        rho_cos = st["rho_anova"]
        m = mts(cell, f, cnt, pr, rho_cos, {k: v["auroc_unseen"] for k, v in perk.items()})
        folds.append(dict(cell_id=cell, fold=f, N=N, G=st["G"], d=dim, gsize_min=int(cnt.min()),
                          gsize_median=float(np.median(cnt)), gsize_max=int(cnt.max()), id_acc=reg["id_acc"],
                          rho_cos=rho_cos, rho_cos_raw=st["rho_raw"], pr=pr, twonn=twonn_d, twonn_m=twonn_m,
                          x1=rho_cos * np.sqrt(pr), x2=rho_cos * np.sqrt(dim), x3=rho_cos * np.sqrt(max(twonn_d, 0.0)),
                          d_over_N=dim / N, T_unseen=T_unseen, T_seen=T_seen, T_m_eval=m_e, T_m_fit=m_f,
                          n_seen=int(seen.sum()), n_unseen=int(unseen.sum()), n_ood=int(Uood.shape[0]),
                          perk={str(k): v for k, v in perk.items()}, mts=m))
        del F
        torch.cuda.empty_cache()
        print(cell, "fold", f, f"{time.time() - t0:.0f}s", flush=True)
    d50 = float(np.mean([x["perk"]["50"]["auroc_seen"] for x in folds]) - np.mean([x["perk"]["50"]["auroc_unseen"] for x in folds]))
    r3 = R3.get((cell, "knn_mean_cosine"))
    res = dict(cell_id=cell, dataset=reg["dataset"], backbone=reg["backbone"], family=reg["family"], seed=seed,
               scorer="knn_mean_cosine (Track A) k in " + str(KS) + "; k-th-distance variant (report only)",
               delta_k50_own=d50, delta_k50_r3=float(r3["delta"]) if r3 else None,
               abs_diff_k50_vs_r3=abs(d50 - float(r3["delta"])) if r3 else None,
               folds=folds, seconds=time.time() - t0, precommit=CM.precommit_hash())
    CM.atomic_write_text(out, json.dumps(res, indent=1))
    print(cell, f"{time.time() - t0:.0f}s", "delta_k50", d50, "r3", res["delta_k50_r3"], flush=True)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for c in sys.argv[1:]:
        run_cell(c)
