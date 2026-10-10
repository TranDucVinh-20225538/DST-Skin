#!/usr/bin/env python3
"""T1 item D, synthetic part: D1 sensitivity / specificity of the label-free fingerprint T and D2 adversarial search.

T (K5): (mean NN^2(eval -> fit) - mean NN^2(fit -> fit, leave-one-out)) / sqrt(Var_e / m + Var_f / n_f), squared Euclidean
nearest-neighbour distances; eval m = 5,000 points (round(omega * m) seen + the rest unseen); fit -> fit for a uniform subsample of
min(20,000, N) fit points against the full fit set. Primary: L2-normalised features; secondary: raw.
Delta_full: true leak at omega = 1 (AUROC_seen - AUROC_unseen vs OOD O1 c = 2) for Mahalanobis (LW) and kNN-k50.

    item_D_sim.py d1 TASK NTASKS     -> results/t1/D/raw/d1_<task>.jsonl
    item_D_sim.py d2r TASK NTASKS    -> results/t1/D/raw/d2r_<task>.jsonl
    item_D_sim.py cma                -> results/t1/D/raw/d2_cma.jsonl (needs every d2r unit)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / ".local" / "t1_pylib"))
import common as CM  # noqa: E402
import d_design  # noqa: E402
import scorer64 as S64  # noqa: E402
import sim_c  # noqa: E402

RAW = Path(os.environ.get("T1_RAW", str(CM.RES / "D" / "raw")))
DEV = S64.DEV
ADD = CM.RES / "PRECOMMIT_T1_addendum_D.json"
M_EVAL, M_OOD, M_FIT_LOO, REPS = 5000, 4000, 20000, 16
OMEGAS = (0.0, 0.25, 0.5, 1.0)
CHUNK = 2048


def nn2(Q, F, self_idx=None):
    fn = (F * F).sum(1)
    out = []
    for i in range(0, Q.shape[0], CHUNK):
        q = Q[i:i + CHUNK]
        d2 = ((q * q).sum(1, keepdim=True) - 2.0 * q @ F.T + fn[None, :]).clamp_min(0.0)
        if self_idx is not None:
            r = torch.arange(q.shape[0], device=DEV)
            d2[r, torch.as_tensor(self_idx[i:i + CHUNK], device=DEV)] = float("inf")
        out.append(d2.min(1).values)
    return torch.cat(out).cpu().numpy()


def unitize(X):
    return X / torch.linalg.norm(X, dim=1, keepdim=True).clamp_min(1e-300)


def t_from(de, df):
    return float((de.mean() - df.mean()) / np.sqrt(de.var(ddof=1) / len(de) + df.var(ddof=1) / len(df)))


def model_cfg(p, rng_sizes):
    """D1 / D2 parameters -> sim_c.Model configuration."""
    n_g = np.full(p["G"], p["n"], dtype=int)
    cv = p.get("cv_n", 0.0) or 0.0
    if cv > 0:
        s2 = np.log(1 + cv ** 2)
        n_g = np.maximum(1, np.round(p["n"] * np.exp(np.sqrt(s2) * rng_sizes.standard_normal(p["G"]) - s2 / 2))).astype(int)
    cfg = dict(d=p["d"], rho=p["rho"], n_g=n_g.tolist())
    if p.get("law") == "t5":
        cfg.update(nu_u=5.0, nu_e=5.0)
    if p.get("nu") is not None:
        cfg.update(nu_u=p["nu"], nu_e=p["nu"])
    if "alpha" in p:
        cfg["spectrum"] = d_design.spectrum(p).tolist()
        cfg["u_rank"] = p["u_rank"]
        cfg["K"], cfg["sep"] = p["K"], p["sep"]
    return cfg


def one_rep(cfg, ss, omegas, need_delta=True):
    dr = sim_c.Draw(ss)
    mod = sim_c.Model(cfg, dr)
    X = mod.fit()
    N, d = X.shape
    Zs, Zu = mod.seen(M_EVAL), mod.fresh(M_EVAL)
    idx = np.sort(dr.np.choice(N, min(M_FIT_LOO, N), replace=False))
    res = dict(N=N)
    for tag, f in (("norm", unitize), ("raw", lambda a: a)):
        Fx, S, U = f(X), f(Zs), f(Zu)
        ds, du = nn2(S, Fx), nn2(U, Fx)
        df = nn2(Fx[torch.as_tensor(idx, device=DEV)], Fx, self_idx=idx)
        for om in omegas:
            k = int(round(om * M_EVAL))
            res[f"T_{tag}_w{om:g}"] = t_from(np.r_[ds[:k], du[k:]], df)
        del Fx, S, U
    if need_delta:
        a2 = 1 + 2.0 * np.sqrt(2.0 / d)
        Zo = np.sqrt(a2) * mod.fresh(M_OOD)
        lw = S64.LW(X)
        qs, qu, qo = (lw.q(Z).cpu().numpy() for Z in (Zs, Zu, Zo))
        res["delta_lw"] = sim_c.auroc(qs, qo) - sim_c.auroc(qu, qo)
        k = min(50, N)
        ks, ku, ko = (sim_c.knn_dist(X, Z, [k])[:, 0].cpu().numpy() for Z in (Zs, Zu, Zo))
        res["delta_knn50"] = sim_c.auroc(ks, ko) - sim_c.auroc(ku, ko)
        del lw, Zo
    del X, Zs, Zu
    return res


def auc_t(t0, t1):
    """AUC of T for omega = 0 (positive: larger T = disjoint) vs omega = 1: P(T_0 > T_1) + 1/2 ties."""
    return S64.auroc_id_pos(np.asarray(t0), np.asarray(t1))


def eval_config(uid, p, omegas, null_rho0=False):
    rs = np.random.default_rng(CM.seed_seq("D", dict(unit=uid, what="sizes")))
    cfg = model_cfg(p, rs)
    reps = [one_rep(cfg, CM.seed_seq("D", uid, r), omegas) for r in range(REPS)]
    out = dict(N=reps[0]["N"], reps=REPS)
    for k in reps[0]:
        if k == "N":
            continue
        v = np.array([r[k] for r in reps])
        out[k] = v.tolist()
        out[k + "_mean"] = float(v.mean())
    out["delta_full"] = max(out["delta_lw_mean"], out["delta_knn50_mean"])
    out["material"] = bool(out["delta_full"] >= 0.02)
    out["auc_T_norm"] = auc_t(out["T_norm_w0"], out["T_norm_w1"])
    out["auc_T_raw"] = auc_t(out["T_raw_w0"], out["T_raw_w1"])
    if null_rho0:
        c0 = dict(cfg, rho=0.0)
        z = [one_rep(c0, CM.seed_seq("D", dict(unit=uid, what="rho0"), r), (0.0,), need_delta=False) for r in range(REPS)]
        for tag in ("norm", "raw"):
            v = np.array([r[f"T_{tag}_w0"] for r in z])
            out[f"rho0_T_{tag}"] = v.tolist()
            out[f"rho0_size_{tag}"] = float(np.mean(np.abs(v) > 1.959963984540054))
    return out


def run_list(prefix, task, ntasks):
    CM.set_float64_torch()
    RAW.mkdir(parents=True, exist_ok=True)
    add = json.load(open(ADD))
    order = [u for u in add["work_lists"][0]["execution_order"] if u.startswith(prefix + "|")]
    params = add["unit_params"]
    mine = [u for j, u in enumerate(order) if j % ntasks == task]
    out = RAW / f"{prefix}_{task}.jsonl"
    have = set()
    if out.exists():
        with open(out, "rb+") as fh:
            data = fh.read()
            if data and not data.endswith(b"\n"):
                fh.truncate(data.rfind(b"\n") + 1)
        have = {json.loads(line)["unit"] for line in open(out)}
    pre = CM.precommit_hash()
    todo = [u for u in mine if u not in have][:int(os.environ.get("T1_MAXUNITS", "100000000"))]
    with open(out, "a") as fh:
        for uid in todo:
            t0 = time.time()
            p = params[uid]
            res = eval_config(uid, p, OMEGAS if prefix == "d1" else (0.0, 1.0), null_rho0=(prefix == "d1"))
            fh.write(json.dumps(dict(unit=uid, params=p, seconds=time.time() - t0, precommit=pre, **res)) + "\n")
            fh.flush()
            torch.cuda.empty_cache()
            print(uid, f"{time.time() - t0:.1f}s", res["auc_T_norm"], res["delta_full"], flush=True)
    if "T1_MAXUNITS" not in os.environ:
        Path(str(out) + ".done").write_text("ok\n")


def run_cma():
    """CMA-ES (population 32, 40 generations, seed 401) minimising AUC(T) subject to Delta_full >= 0.02, started from the
    20 worst feasible random configs: initial mean = their centroid in the encoded space, initial sigma = mean coordinate SD."""
    import cma
    CM.set_float64_torch()
    rnd = [json.loads(line) for p in sorted(glob.glob(str(RAW / "d2r_*.jsonl"))) for line in open(p)]
    if len(rnd) < 3000:
        sys.exit(f"only {len(rnd)} of 3000 D2 random configs present")
    feas = sorted([r for r in rnd if r["material"]], key=lambda r: r["auc_T_norm"])[:20]
    X0 = np.array([r["params"]["x"] for r in feas])
    x0, sigma0 = X0.mean(0), float(X0.std(0, ddof=1).mean())
    out = RAW / "d2_cma.jsonl"
    done = {}
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done[(r["gen"], r["i"])] = r
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {"popsize": 32, "seed": 401, "bounds": [0, 1], "maxiter": 40, "verbose": -9})
    pre = CM.precommit_hash()
    with open(out, "a") as fh:
        for gen in range(40):
            xs = es.ask()
            fit = []
            for i, x in enumerate(xs):
                if (gen, i) in done:
                    r = done[(gen, i)]
                else:
                    t0 = time.time()
                    p = dict(x=list(map(float, x)), **d_design.decode(x))
                    uid = f"d2cma|g{gen:02d}|i{i:02d}"
                    res = eval_config(uid, p, (0.0, 1.0))
                    r = dict(unit=uid, gen=gen, i=i, params=p, seconds=time.time() - t0, precommit=pre, **res)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    torch.cuda.empty_cache()
                    print(uid, f"{r['seconds']:.1f}s", r["auc_T_norm"], r["delta_full"], flush=True)
                pen = 0.0 if r["delta_full"] >= 0.02 else 1.0 + 10.0 * (0.02 - r["delta_full"])
                fit.append(r["auc_T_norm"] + pen)
            es.tell(xs, fit)
    CM.atomic_write_text(RAW / "d2_cma.done", json.dumps(dict(x0=x0.tolist(), sigma0=sigma0, start_units=[r["unit"] for r in feas])))


if __name__ == "__main__":
    if sys.argv[1] == "cma":
        run_cma()
    else:
        run_list(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
