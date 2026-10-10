#!/usr/bin/env python3
"""T1 item D, real part (stage 3): D3 (known groups), D4 (null on real features), D5 (dependence-aware SE, post-hoc).

Per cell and paper_2fold fold f (fit = fit-set rows of fold f; seen eval = eval rows of the fit groups; unseen eval = eval rows of
the other fold's groups), on L2-normalised (primary) and raw (secondary) features:
  D3  squared-Euclidean NN distances of every seen / unseen eval point to the full fit set and fit -> fit leave-one-out distances
      of a uniform subsample of min(20,000, N_fit) fit points; 50 random subsamples of m = min(5,000, |eval|) eval points:
      T_seen, T_unseen and the mixtures omega in {0, .25, .5, .75, 1} (round(omega m) seen + the rest unseen; m = min(5,000,
      |seen|, |unseen|) for the mixtures).
  D4  fit set split at random ignoring groups (half pseudo-fit / half pseudo-eval; eval exchangeable with fit, overlap present):
      T for 50 subsamples of m = min(5,000, |pseudo-eval|) against the pseudo-fit set. Permuted group labels: T has no group
      argument, so the permuted-label T equals T (implementation check, recorded).
  D5  group-cluster bootstrap (1,000 resamples of the eval groups and of the fit groups, default_rng of the unit) of the mean
      NN^2 difference (unseen eval vs fit LOO; first subsample) vs the naive SE of the z form; ratio.

    item_D_real.py TASK NTASKS   -> results/t1/D/raw/dreal_<task>.jsonl (units d3|<cell>, d4|<cell> of the D work list)
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
import common as CM  # noqa: E402
import scorer64 as S64  # noqa: E402
from item_D_sim import nn2, t_from  # noqa: E402
from item_H1 import load  # noqa: E402

RAW = Path(os.environ.get("T1_RAW", str(CM.RES / "D" / "raw")))
DEV = S64.DEV
M, M_LOO, NSUB, NBOOT = 5000, 20000, 50, 1000
OMEGAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CAP_S = 24 * 3600.0


def spent():
    tot = 0.0
    for f in glob.glob(str(RAW / "*.jsonl")):
        for line in open(f):
            try:
                tot += float(json.loads(line).get("seconds", 0.0))
            except ValueError:
                pass
    return tot


def loo(F, rng):
    idx = np.sort(rng.choice(F.shape[0], min(M_LOO, F.shape[0]), replace=False))
    return nn2(F[torch.as_tensor(idx, device=DEV)], F, self_idx=idx), idx


def cluster_boot_se(de, ge, df, gf, rng):
    """SE of mean(de) - mean(df) under resampling of eval groups and fit groups (independent)."""
    ue, uf = np.unique(ge), np.unique(gf)
    ie = [np.flatnonzero(ge == g) for g in ue]
    i_f = [np.flatnonzero(gf == g) for g in uf]
    se_, sf_ = np.array([de[i].sum() for i in ie]), np.array([df[i].sum() for i in i_f])
    ne_, nf_ = np.array([len(i) for i in ie]), np.array([len(i) for i in i_f])
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        a = rng.integers(0, len(ue), len(ue))
        c = rng.integers(0, len(uf), len(uf))
        out[b] = se_[a].sum() / ne_[a].sum() - sf_[c].sum() / nf_[c].sum()
    return float(out.std(ddof=1))


def d3(uid, cell):
    reg, xtr, gtr, xid, gid, xood, tf, idf = load(cell)
    del xood
    folds = []
    for f in (0, 1):
        rng = np.random.default_rng(CM.seed_seq("D", dict(unit=uid, fold=f), 0))
        fit = tf == f
        seen, unseen = np.flatnonzero(idf == f), np.flatnonzero((idf >= 0) & (idf != f))
        row = dict(fold=f, N_fit=int(fit.sum()), n_seen=len(seen), n_unseen=len(unseen))
        if len(seen) < 2 or len(unseen) < 2:
            folds.append(dict(row, status="not evaluable: < 2 seen or unseen eval points"))
            continue
        sub_s = [rng.choice(len(seen), min(M, len(seen)), replace=False) for _ in range(NSUB)]
        sub_u = [rng.choice(len(unseen), min(M, len(unseen)), replace=False) for _ in range(NSUB)]
        mm = min(M, len(seen), len(unseen))
        mix = [(rng.permutation(len(seen))[:mm], rng.permutation(len(unseen))[:mm]) for _ in range(NSUB)]
        for tag, tr in (("norm", S64.l2n), ("raw", lambda a: a)):
            F = S64.to_t(tr(xtr[fit]))
            ds = nn2(S64.to_t(tr(xid[seen])), F)
            du = nn2(S64.to_t(tr(xid[unseen])), F)
            df, idx = loo(F, np.random.default_rng(CM.seed_seq("D", dict(unit=uid, fold=f, what="loo"), 0)))
            row[f"T_seen_{tag}"] = [t_from(ds[s], df) for s in sub_s]
            row[f"T_unseen_{tag}"] = [t_from(du[s], df) for s in sub_u]
            for om in OMEGAS:
                k = int(round(om * mm))
                row[f"T_mix_{tag}_w{om:g}"] = [t_from(np.r_[ds[a[:k]], du[b[k:]]], df) for a, b in mix]
            row[f"mean_nn2_seen_{tag}"], row[f"mean_nn2_unseen_{tag}"], row[f"mean_nn2_fitloo_{tag}"] = \
                float(ds.mean()), float(du.mean()), float(df.mean())
            if tag == "norm":
                s0 = sub_u[0]
                de = du[s0]
                naive = float(np.sqrt(de.var(ddof=1) / len(de) + df.var(ddof=1) / len(df)))
                cb = cluster_boot_se(de, gid[unseen][s0], df, gtr[fit][idx],
                                     np.random.default_rng(CM.seed_seq("D", dict(unit=uid, fold=f, what="d5"), 0)))
                row.update(d5_naive_se=naive, d5_cluster_boot_se=cb, d5_ratio=cb / naive if naive > 0 else float("nan"))
            del F
            torch.cuda.empty_cache()
        row["status"] = "ok"
        folds.append(row)
    return dict(cell=cell, dataset=reg["dataset"], backbone=reg["backbone"], family=reg["family"], folds=folds)


def d4(uid, cell):
    reg, xtr, gtr, xid, gid, xood, tf, idf = load(cell)
    del xid, xood
    folds = []
    for f in (0, 1):
        rng = np.random.default_rng(CM.seed_seq("D", dict(unit=uid, fold=f), 0))
        rows = np.flatnonzero(tf == f)
        perm = rng.permutation(rows)
        pf, pe = np.sort(perm[:len(perm) // 2]), np.sort(perm[len(perm) // 2:])
        row = dict(fold=f, n_pseudo_fit=len(pf), n_pseudo_eval=len(pe))
        subs = [rng.choice(len(pe), min(M, len(pe)), replace=False) for _ in range(NSUB)]
        for tag, tr in (("norm", S64.l2n), ("raw", lambda a: a)):
            F = S64.to_t(tr(xtr[pf]))
            de = nn2(S64.to_t(tr(xtr[pe])), F)
            df, _ = loo(F, np.random.default_rng(CM.seed_seq("D", dict(unit=uid, fold=f, what="loo"), 0)))
            row[f"T_null_{tag}"] = [t_from(de[s], df) for s in subs]
            del F
            torch.cuda.empty_cache()
        row["permuted_labels_T_identical"] = True
        row["status"] = "ok"
        folds.append(row)
    return dict(cell=cell, dataset=reg["dataset"], backbone=reg["backbone"], family=reg["family"], folds=folds)


def main(task, ntasks):
    CM.set_float64_torch()
    RAW.mkdir(parents=True, exist_ok=True)
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_D.json"))
    order = [u for u in add["work_lists"][0]["execution_order"] if u.startswith(("d3|", "d4|"))]
    mine = [u for i, u in enumerate(order) if i % ntasks == task]
    out = RAW / f"dreal_{task}.jsonl"
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
            if spent() >= CAP_S:
                print("D cap reached", flush=True)
                return
            t0 = time.time()
            cell = uid.split("|", 1)[1]
            res = d3(uid, cell) if uid.startswith("d3|") else d4(uid, cell)
            res["seconds"] = time.time() - t0
            fh.write(json.dumps(dict(unit=uid, precommit=pre, **res)) + "\n")
            fh.flush()
            print(uid, f"{res['seconds']:.0f}s", flush=True)
    if "T1_MAXUNITS" not in os.environ:
        Path(str(out) + ".done").write_text("ok\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
