#!/usr/bin/env python3
"""T1 item A2: within-backbone subsampling designs (Camelyon, seed 42, mahalanobis_l2).

    item_A2.py BACKBONE FOLD  -> results/t1/A/raw/a2_<backbone>_f<fold>.jsonl (+ .done)

Units a2|<backbone>|f<fold>|G<G'>|n<n'>|r<rep> are run in the item-A work-list execution order (restricted to this
backbone x fold). Per unit: G'/3 slides per hospital {0,3,4} drawn without replacement from the hospital's 5 slides of
the fold (G' = 15 keeps all), n' patches per retained slide drawn without replacement (all if the slide has fewer, or
n' = all); RNG = default_rng(SeedSequence([20261010, 1, stable_hash(config), rep])), config = {backbone, fold, G, n}.
Seen-eval = ID-eval patches of the retained slides; unseen-eval unchanged (ID-eval of the other fold's slides).
Descriptors, s_logo, P_dQ, P_Delta and Delta_meas recomputed on the subsampled fit set exactly as in A1.
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
from item_A1_cell import logo_s  # noqa: E402

RAW = Path(__import__("os").environ.get("T1_RAW", str(CM.RES / "A" / "raw")))
HOSP = {r["slide"]: int(r["hospital"]) for r in csv.DictReader(open(CM.RES / "I" / "camelyon_manifest.csv"))}


def main(backbone, fold):
    fold = int(fold)
    out = RAW / f"a2_{backbone}_f{fold}.jsonl"
    done_marker = Path(str(out) + ".done")
    if done_marker.exists():
        print("done", backbone, fold)
        return
    order = [u for u in json.load(open(CM.RES / "PRECOMMIT_T1_addendum_A.json"))["work_lists"][0]["execution_order"]
             if u.startswith(f"a2|{backbone}|f{fold}|")]
    have = set()
    if out.exists():
        for line in open(out):
            try:
                have.add(json.loads(line)["unit"])
            except Exception:  # noqa: BLE001  (a truncated last line from a killed job is ignored and recomputed)
                pass
    todo = [u for u in order if u not in have][:int(os.environ.get("T1_MAXUNITS", "1000000"))]
    if not todo:
        done_marker.write_text("ok\n")
        return
    torch = CM.set_float64_torch()
    cell = f"camelyon_{backbone}_s42"
    d = CM.load_cell(cell)
    xtr, gtr, xid, gid, xood, fm = CM.paper2fold_inputs(d, 42)
    from crossfit_ood import core as C
    tf, idf = C._apply_map(fm[0], gtr), C._apply_map(fm[0], gid)
    gtr, gid = gtr.astype(str), gid.astype(str)
    Xtr, Xid, Xood = S64.to_t(S64.l2n(xtr)), S64.to_t(S64.l2n(xid)), S64.to_t(S64.l2n(xood))
    del xtr, xid, xood, d
    fit_slides = sorted(set(gtr[tf == fold]))
    by_h = {h: sorted(s for s in fit_slides if HOSP[s] == h) for h in (0, 3, 4)}
    if any(len(v) != 5 for v in by_h.values()):
        raise SystemExit(f"S11: fold {fold} of {cell} does not hold 5 slides per hospital: { {h: len(v) for h, v in by_h.items()} }")
    idx_of = {s: np.flatnonzero((gtr == s) & (tf == fold)) for s in fit_slides}
    unseen = (idf >= 0) & (idf != fold)
    q_ood_cache = {}
    with open(out, "a") as fh:
        for unit in todo:
            t0 = time.time()
            _, bb, f_s, G_s, n_s, r_s = unit.split("|")
            Gp, n_p, rep = int(G_s[1:]), n_s[1:], int(r_s[1:])
            cfg = dict(backbone=bb, fold=fold, G=Gp, n=n_p)
            rng = np.random.default_rng(CM.seed_seq("A", cfg, rep))
            keep = []
            for h in (0, 3, 4):
                sl = by_h[h]
                keep += sorted(sl) if Gp == 15 else sorted(rng.choice(sl, Gp // 3, replace=False).tolist())
            keep = sorted(keep)
            rows = []
            for s in keep:
                ix = idx_of[s]
                if n_p != "all" and len(ix) > int(n_p):
                    ix = np.sort(rng.choice(ix, int(n_p), replace=False))
                rows.append(ix)
            sel = np.concatenate(rows)
            Xf = Xtr[torch.as_tensor(sel, device=S64.DEV)]
            gf = gtr[sel]
            N, dim = Xf.shape
            lw = S64.LW(Xf)
            delta = lw.shrinkage
            st, u, cnt = S64.group_stats(Xf, gf, lw.P)
            s_logo, scheme = logo_s(Xf, gf, lw, cell, fold)
            q_id = lw.q(Xid).cpu().numpy()
            q_ood = lw.q(Xood).cpu().numpy()
            seen = np.isin(gid, keep)
            q_s, q_u = q_id[seen], q_id[unseen]
            a_seen = S64.auroc_id_pos(-q_s, -q_ood)
            a_unseen = S64.auroc_id_pos(-q_u, -q_ood)
            ug, cg = np.unique(gid[seen], return_counts=True)
            n_of = dict(zip(u, cnt))
            n_g = np.array([n_of[x] for x in ug], dtype=float)
            rho = st["rho_w"]
            dq = S64.dq_pred_groups(n_g, cg / cg.sum(), rho, s_logo, N, delta)
            p_delta = S64.auroc_id_pos(-(q_u - abs(dq)), -q_ood) - a_unseen
            rec = dict(unit=unit, backbone=bb, fold=fold, G_prime=Gp, n_prime=n_p, rep=rep, slides=keep, N=N, G=st["G"], d=dim,
                       n_bar=N / st["G"], lw_delta=delta, rho_w=rho, rho_raw=st["rho_raw"], rho_anova=st["rho_anova"],
                       s_logo=s_logo, s_logo_scheme=scheme, s_tr=lw.s_tr(),
                       r_sat=abs(dq * (1 - delta)) / (rho * s_logo) if rho * s_logo != 0 else float("nan"),
                       dq_pred=dq, dq_meas=float(q_s.mean() - q_u.mean()), auroc_seen=a_seen, auroc_unseen=a_unseen,
                       delta_meas=a_seen - a_unseen, p_delta=p_delta, n_seen=int(seen.sum()), n_unseen=int(unseen.sum()),
                       d_over_N=dim / N, seconds=time.time() - t0)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            del Xf, lw
    if "T1_MAXUNITS" not in os.environ:
        done_marker.write_text("ok\n")
    print(backbone, fold, "units", len(todo), flush=True)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    main(sys.argv[1], sys.argv[2])
