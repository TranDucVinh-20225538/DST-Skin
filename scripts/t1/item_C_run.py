#!/usr/bin/env python3
"""T1 item C runner (C1, C3, C4; C5 in stage 3): processes the units of one phase in the item-C work-list execution order.

    item_C_run.py PHASE TASK NTASKS    PHASE in {c1, c3, c4, c5}; units of the phase in execution order, unit j handled by
                                       task j % NTASKS; output results/t1/C/raw/shard_<phase>_<task>.jsonl (+ .done), resumable.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import scorer64 as S64  # noqa: E402
import sim_c  # noqa: E402

RAW = Path(os.environ.get("T1_RAW", str(CM.RES / "C" / "raw")))
DEV = S64.DEV
ADD = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_C.json"))
PARAMS = ADD["unit_params"]
ORDER = ADD["work_lists"][0]["execution_order"]
PREFIX = {"core": ("c1|", "c3|", "c4tv|", "c4sc|"), "c5": ("c5|",)}
CAP_S = 160 * 3600
STOP_S8 = CM.RES / "C" / "STOP_S8.json"


def spent_seconds():
    return sum(float(p.read_text() or 0) for p in RAW.glob("shard_*.secs"))


def s8_violation(uid, res):
    """S8: a non-vacuous Gaussian configuration whose 99.9 % upper bound of E[Delta] exceeds the cap K3."""
    if not res.get("cap_nonvacuous"):
        return None
    if uid.startswith("c4tv|") and res["ub999_oracle"] > res["cap"]:
        return dict(unit=uid, scorer="oracle", ub999=res["ub999_oracle"], cap=res["cap"])
    if uid.startswith("c4sc|gauss|"):
        for k, v in res.items():
            if k.endswith("|exceeds_cap") and v:
                name = k.split("|")[0]
                return dict(unit=uid, scorer=name, ub999=res[f"{name}|ub999"], cap=res["cap"])
    return None


def t(a):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=DEV)


def seed_fn(uid):
    return lambda r: CM.seed_seq("C", uid, r)


# ---------------------------------------------------------------- C1 / C3
def run_c1(uid, p):
    cfg = dict(d=p["d"], rho=p["rho"], n_g=[p["n"]] * p["G"])
    agg, _ = sim_c.run_config(cfg, seed_fn(uid))
    return dict(agg)


def run_c3(uid, p):
    cfg = dict(d=p["d"], rho=p["rho"], n_g=[p["n"]] * p["G"])
    agg, _ = sim_c.run_config(cfg, seed_fn(uid), scorers=("maha", "knn", "vim"), ks=(1, 5, 20))
    return dict(agg)


# ---------------------------------------------------------------- C4
def cap(d, rho, G, omega=1.0):
    return float(omega / 2 * np.sqrt(((1 - rho) ** (-d) - 1) / G))


def grid_axes(d):
    if d == 1:
        return [np.linspace(-9, 9, 6001)]
    if d == 2:
        return [np.linspace(-7, 7, 421)] * 2
    return [np.linspace(-7, 7, 256)] * 3


def tv_and_oracle(rng, d, G, rho):
    """Exact numerical TV(P_in, P_out) on the grid and the oracle (likelihood-ratio) AUROC(P_in vs P_out) - 1/2."""
    u = rng.standard_normal((G, d)) * np.sqrt(rho)
    s2 = 1 - rho
    axes = [t(a) for a in grid_axes(d)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    pts = torch.stack([m.reshape(-1) for m in mesh], 1)
    dA = float(np.prod([float(a[1] - a[0]) for a in axes]))
    del mesh
    pin = torch.zeros(pts.shape[0], dtype=torch.float64, device=DEV)
    for a in u:
        pin += torch.exp(-((pts - t(a)) ** 2).sum(1) / (2 * s2))
    pin *= 1.0 / (G * (2 * np.pi * s2) ** (d / 2))
    pout = torch.exp(-(pts ** 2).sum(1) / 2) / (2 * np.pi) ** (d / 2)
    del pts
    tv = float(0.5 * (pin - pout).abs().sum() * dA)
    wi, wo = pin * dA, pout * dA
    llr = torch.log(pin.clamp_min(1e-300)) - torch.log(pout.clamp_min(1e-300))
    o = torch.argsort(llr)
    wi, wo, llr = wi[o], wo[o], llr[o]
    # P(llr(X_in) > llr(X_out)) + 1/2 ties, X_in ~ wi, X_out ~ wo (grid masses renormalised)
    wi, wo = wi / wi.sum(), wo / wo.sum()
    cum_o = torch.cumsum(wo, 0) - wo
    au = float((wi * (cum_o + 0.5 * wo)).sum())
    return tv, au - 0.5, float(pin.sum() * dA), float(pout.sum() * dA)


def run_c4tv(uid, p):
    rng = np.random.default_rng(CM.seed_seq("C", uid, 0))
    vals = np.array([tv_and_oracle(rng, p["d"], p["G"], p["rho"]) for _ in range(64)])
    c = cap(p["d"], p["rho"], p["G"])
    tv, orc = vals[:, 0], vals[:, 1]
    return dict(draws=64, grid=[len(a) for a in grid_axes(p["d"])], E_TV=float(tv.mean()), E_TV_se=float(tv.std(ddof=1) / 8),
                E_delta_oracle=float(orc.mean()), E_delta_oracle_se=float(orc.std(ddof=1) / 8),
                mass_in_min=float(vals[:, 2].min()), mass_out_min=float(vals[:, 3].min()), cap=c, cap_nonvacuous=bool(c < 0.45),
                ub999_oracle=float(orc.mean() + 3.090232306167813 * orc.std(ddof=1) / 8))


def oracle_scores(mod, Z):
    """Gaussian likelihood ratio that knows the fit group effects u_g (in whitened coordinates if Sigma != I)."""
    Zw = Z if mod.sq is None else Z / mod.sq[None, :]
    s2 = 1 - mod.rho
    d = Zw.shape[1]
    out = []
    for i in range(0, Zw.shape[0], 2048):
        z = Zw[i:i + 2048]
        d2 = ((z * z).sum(1, keepdim=True) - 2 * z @ mod.u.T + (mod.u * mod.u).sum(1)[None, :]).clamp_min(0.0)
        lin = torch.logsumexp(-d2 / (2 * s2), 1) - np.log(d2.shape[1]) - d / 2 * np.log(s2)
        lout = -(z * z).sum(1) / 2
        out.append(lin - lout)
    return torch.cat(out)


def run_c4sc(uid, p):
    d, rho, G, n = p["d"], p["rho"], p["G"], p["n"]
    cfg = dict(d=d, rho=rho, n_g=[n] * G)
    if p["law"] == "t5":
        cfg.update(nu_u=5.0, nu_e=5.0)
    if p["law"] == "aniso":
        sp = np.arange(1, d + 1, dtype=float) ** -1.0
        cfg["spectrum"] = (sp / sp.mean()).tolist()
    reps = []
    for r in range(64):
        dr = sim_c.Draw(CM.seed_seq("C", uid, r))
        mod = sim_c.Model(cfg, dr)
        X = mod.fit()
        Zs, Zu, Zo = mod.seen(sim_c.M), mod.fresh(sim_c.M), mod.fresh(sim_c.M)
        rr = {}
        k1 = lambda Z: sim_c.knn_dist(X, Z, [1])[:, 0].cpu().numpy()  # noqa: E731
        sc = {"knn1": (k1(Zs), k1(Zu), k1(Zo))}
        fam, _, _ = sim_c.maha_family(X, lams=(0.0,) if X.shape[0] >= 2 * d else (0.01,), lw=True)
        for name, ent in fam.items():
            if name.startswith("maha_u") or name == "maha_lw":
                sc[name] = tuple(sim_c.maha_q(ent, Z).cpu().numpy() for Z in (Zs, Zu, Zo))
        sc["oracle"] = tuple((-oracle_scores(mod, Z)).cpu().numpy() for Z in (Zs, Zu, Zo))
        for name, (a, b, o) in sc.items():
            rr[name] = sim_c.auroc(a, o) - sim_c.auroc(b, o)
        reps.append(rr)
        del X, Zs, Zu, Zo, fam
    c = cap(d, rho, G)
    out = dict(law=p["law"], reps=64, cap=c, cap_nonvacuous=bool(c < 0.45))
    for name in reps[0]:
        v = np.array([r[name] for r in reps])
        m, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
        out[f"{name}|E_delta"] = m
        out[f"{name}|se"] = se
        out[f"{name}|ub999"] = m + 3.090232306167813 * se
        out[f"{name}|tightness"] = m / c if c > 0 else float("nan")
        out[f"{name}|exceeds_cap"] = bool(m + 3.090232306167813 * se > c)
    return out


HANDLERS = {"c1|": run_c1, "c3|": run_c3, "c4tv|": run_c4tv, "c4sc|": run_c4sc}


def main(phase, task, ntasks):
    CM.set_float64_torch()
    RAW.mkdir(parents=True, exist_ok=True)
    units = [u for u in ORDER if u.startswith(PREFIX[phase])]
    mine = [u for j, u in enumerate(units) if j % ntasks == task]
    out = RAW / f"shard_{phase}_{task}.jsonl"
    done = Path(str(out) + ".done")
    if done.exists():
        print("done", out)
        return
    have = set()
    if out.exists():
        for line in open(out):
            try:
                have.add(json.loads(line)["unit"])
            except Exception:  # noqa: BLE001  (a truncated last line from a killed job is recomputed)
                pass
        with open(out, "rb+") as fh:
            data = fh.read()
            if data and not data.endswith(b"\n"):
                fh.truncate(data.rfind(b"\n") + 1)
    PRE = CM.precommit_hash()
    todo = [u for u in mine if u not in have][:int(os.environ.get("T1_MAXUNITS", "100000000"))]
    secs_file = Path(str(out).replace(".jsonl", ".secs"))
    secs = float(secs_file.read_text() or 0) if secs_file.exists() else 0.0
    with open(out, "a") as fh:
        for uid in todo:
            if STOP_S8.exists():
                print("S8 stop present; stopping", flush=True)
                return
            if spent_seconds() >= CAP_S:
                print("C cap reached; stopping (rule 16)", flush=True)
                return
            t0 = time.time()
            p = PARAMS[uid]
            h = [f for k, f in HANDLERS.items() if uid.startswith(k)][0]
            res = h(uid, p)
            rec = dict(unit=uid, params=p, seconds=time.time() - t0, precommit=PRE, **res)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            secs += rec["seconds"]
            CM.atomic_write_text(secs_file, f"{secs}")
            v = s8_violation(uid, res)
            if v is not None:
                CM.atomic_write_text(STOP_S8, json.dumps(dict(v, seed=f"SeedSequence([20261010, 3, stable_hash('{uid}'), rep])"), indent=1))
                print("S8 violation", v, flush=True)
                return
            torch.cuda.empty_cache()
            print(uid, f"{time.time() - t0:.1f}s", rec.get("R", ""), flush=True)
    if "T1_MAXUNITS" not in os.environ:
        done.write_text("ok\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
