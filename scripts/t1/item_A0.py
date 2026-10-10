#!/usr/bin/env python3
"""T1 item A0: unit tests before any real-data claim.

    item_A0.py a0a   toy reproduction of the s2a cells (closed form + float64 GPU simulator, 24 replicates)
    item_A0.py a0b   Track A mahalanobis_l2 Delta recomputed with own fit/score code on 3 cells vs R3 item 1
    item_A0.py a0c   unit conversion of the scorer precision (LW) to 1e-9 relative on one cell
Outputs results/t1/A/a0a.json, a0b.json, a0c.json.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import sim_re  # noqa: E402

OUT = CM.RES / "A"
A0A_CELLS = [  # (d, G, n, rho, lam, bundle sim dQ, bundle SE, bundle theory dQ) from s2a_out.txt
    (128, 40, 10, 0.6, 0.0, -170.27, 0.66, -168.93),
    (128, 100, 5, 0.3, 0.1, -12.73, 0.23, -12.99),
    (32, 40, 10, 0.6, 0.0, -10.89, 0.20, -10.43),
]
A0B_CELLS = ["camelyon_resnet50_s42", "camelyon_uni_s42", "camelyon_dinov2_vitb14_s42"]


def a0a():
    CM.set_float64_torch()
    rows = []
    for d, G, n, rho, lam, b_sim, b_se, b_th in A0A_CELLS:
        cfg = dict(test="A0a", d=d, G=G, n=n, rho=rho, lam=lam)
        reps = np.array([sim_re.sim_maha_dq(d, G, n, rho, lam, CM.seed_seq("A", cfg, r)) for r in range(24)])
        qin, qout = reps.mean(0)
        se_in, se_out = reps.std(0) / np.sqrt(len(reps))  # same SE convention as s2a (np.std, ddof 0; hypot of sides)
        se_new = float(np.hypot(se_in, se_out))
        dq_new = float(qin - qout)
        se_paired = float((reps[:, 0] - reps[:, 1]).std(ddof=1) / np.sqrt(len(reps)))
        s_th, dq_th = sim_re.k2_theory(d, G, n, rho, lam)
        comb = float(np.hypot(se_new, b_se))
        rows.append(dict(**cfg, replicates=24, sim_dq_new=dq_new, se_new=se_new, se_new_paired_report_only=se_paired,
                         sim_dq_bundle=b_sim, se_bundle=b_se, z_combined=(dq_new - b_sim) / comb,
                         pass_sim=bool(abs(dq_new - b_sim) <= 3 * comb),
                         theory_dq_new=dq_th, theory_s_new=s_th, theory_dq_bundle=b_th,
                         theory_rel_diff=abs(dq_th / b_th - 1), pass_theory=bool(abs(dq_th / b_th - 1) <= 0.01)))
        print(rows[-1], flush=True)
    res = dict(rows=rows, pass_A0a=all(r["pass_sim"] and r["pass_theory"] for r in rows),
               rule="new simulation within 3 combined SE of the bundle simulation; theory within 1 % relative of the bundle theory")
    CM.atomic_write_text(OUT / "a0a.json", json.dumps(res, indent=1))
    return res


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def lw_fit(X, torch):
    """Ledoit-Wolf (sklearn formula, assume_centered=False, ddof 0) in float64 on GPU. Returns mu, S_emp, shrinkage, mu_tr."""
    Xt = torch.from_numpy(X).to("cuda", torch.float64)
    n, p = Xt.shape
    loc = Xt.mean(0)
    Xc = Xt - loc
    S = Xc.T @ Xc / n
    X2 = Xc * Xc
    emp_tr = X2.sum(0) / n
    mu = emp_tr.sum() / p
    beta_ = (X2.T @ X2).sum()
    delta_ = (S * S).sum()  # = sum((Xc'Xc)^2)/n^2
    beta = 1.0 / (p * n) * (beta_ / n - delta_)
    delta = (delta_ - 2.0 * mu * emp_tr.sum() + p * mu ** 2) / p
    beta = torch.minimum(beta, delta)
    shr = 0.0 if float(beta) == 0 else float(beta / delta)
    return loc, S, shr, float(mu)


def auroc_id_pos(s_id, s_ood):
    """P(s_ood < s_id) + 0.5 P(tie)."""
    so = np.sort(s_ood)
    lo = np.searchsorted(so, s_id, side="left")
    hi = np.searchsorted(so, s_id, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(s_id) * len(so)))


def a0b():
    torch = CM.set_float64_torch()
    from crossfit_ood import core as C
    ref = {(r["cell"], r["scorer"]): r for r in csv.DictReader(open(CM.REPO / "results/r3/1/paper_ci_v2.csv"))}
    rows = []
    for cell in A0B_CELLS:
        t0 = time.time()
        d = CM.load_cell(cell)
        xtr, gtr, xid, gid, xood, fm = CM.paper2fold_inputs(d, 42)
        tf, idf = C._apply_map(fm[0], gtr), C._apply_map(fm[0], gid)
        xtr, xid, xood = l2n(xtr), l2n(xid), l2n(xood)
        a_seen, a_unseen = [], []
        for f in (0, 1):
            loc, S, shr, mu = lw_fit(xtr[tf == f], torch)
            p = S.shape[0]
            cov = (1 - shr) * S + shr * mu * torch.eye(p, dtype=torch.float64, device="cuda")
            P = torch.linalg.inv(cov)

            def score(Z):
                Zc = torch.from_numpy(Z).to("cuda", torch.float64) - loc
                return (-torch.sqrt(torch.clamp(((Zc @ P) * Zc).sum(1), min=0.0))).cpu().numpy()

            s_id, s_ood = score(xid), score(xood)
            a_seen.append(auroc_id_pos(s_id[idf == f], s_ood))
            a_unseen.append(auroc_id_pos(s_id[(idf >= 0) & (idf != f)], s_ood))
        delta = float(np.mean(a_seen) - np.mean(a_unseen))
        r3 = ref[(cell, "mahalanobis_l2")]
        rows.append(dict(cell=cell, scorer="mahalanobis_l2", auroc_seen=float(np.mean(a_seen)), auroc_unseen=float(np.mean(a_unseen)),
                         delta_own=delta, delta_r3_paper_ci=float(r3["delta"]), delta_r3_trackA=float(r3["trackA_delta"]),
                         abs_diff=abs(delta - float(r3["delta"])), pass_=bool(abs(delta - float(r3["delta"])) <= 1e-6),
                         seconds=time.time() - t0))
        print(rows[-1], flush=True)
    res = dict(rows=rows, pass_A0b=all(r["pass_"] for r in rows), rule="|own Delta - R3 item-1 paper_ci Delta| <= 1e-6 (S2 if not)",
               reference="results/r3/1/paper_ci_v2.csv column delta (R3 item-1 paper_ci.csv values, seen = strict)")
    CM.atomic_write_text(OUT / "a0b.json", json.dumps(res, indent=1))
    return res


def a0c(cell="camelyon_uni_s42"):
    torch = CM.set_float64_torch()
    from crossfit_ood import core as C
    from crossfit_ood.scorers import MahalanobisL2Scorer
    d = CM.load_cell(cell)
    xtr, gtr, *_ , fm = CM.paper2fold_inputs(d, 42)
    tf = C._apply_map(fm[0], gtr)
    X = xtr[tf == 0]
    sc = MahalanobisL2Scorer().fit(X)
    P_scorer = sc.precision_
    from sklearn.covariance import LedoitWolf
    Xn = l2n(X)
    lw = LedoitWolf().fit(Xn)
    delta = float(lw.shrinkage_)
    S_hat = np.cov(Xn, rowvar=False, bias=True)
    mu = np.trace(S_hat) / S_hat.shape[0]
    S_n = S_hat / mu
    lam_eff = delta / (1 - delta)
    P_formula = (1.0 / (1 - delta)) * np.linalg.inv(S_n + lam_eff * np.eye(S_n.shape[0])) / mu
    rel_fro = float(np.linalg.norm(P_formula - P_scorer) / np.linalg.norm(P_scorer))
    rel_max = float(np.max(np.abs(P_formula - P_scorer)) / np.max(np.abs(P_scorer)))
    res = dict(cell=cell, fold=0, N=int(X.shape[0]), d=int(X.shape[1]), delta_hat=delta, mu_hat=float(mu), lam_eff=lam_eff,
               rel_frobenius=rel_fro, rel_max_abs=rel_max, tolerance=1e-9, pass_A0c=bool(rel_fro <= 1e-9 and rel_max <= 1e-9),
               centring="class-agnostic (Track A mahalanobis_l2 uses one mean; Camelyon caches have no labels)",
               note="S_hat = covariance the scorer centred and fitted (L2-normalised features, divisor N); P_scorer = package MahalanobisL2Scorer.precision_")
    print(res, flush=True)
    CM.atomic_write_text(OUT / "a0c.json", json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"a0a": a0a, "a0b": a0b, "a0c": a0c}[sys.argv[1]]()
