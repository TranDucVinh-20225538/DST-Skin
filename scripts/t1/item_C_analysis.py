#!/usr/bin/env python3
"""T1 item C analysis: c1_blocks.csv, c1_correction.json, c2_auroc.csv, c3_collapse.csv, c4_cap.csv, c5_variants.csv,
c5_failure_boundary.json, tests.csv, verdict.json, figures (c1_heatmap_<block>.png, c3_collapse.png, c4_tightness.png).

Readings (see DEVIATIONS.md):
- C1 rows are config x lambda for the uncentred S_lambda (the scorers with the K2 closed form); err = |dQ_sim / dQ_th - 1|.
  C-i uses TEST-interp and TEST-extrap, each restricted to R0 = {d >= 64, G >= 12, N >= 2d}. Units truncated by the cap
  (rule 16, fixed random order) are not run and the blocks use the completed units.
- Correction family fit on TRAIN rows with dQ_sim / dQ_th > 0 (ridge on standardised features; penalty by grouped CV over the
  four TRAIN d values, i.e. leave-one-d-out, because TRAIN has four d levels).
- C3 collapse score per kNN-k (k in {1, 5}), null OOD O3, interior 0.02 < Delta < 0.45: in-sample R2 of the isotonic fit on
  x = rho sqrt(d) divided by the in-sample R2 of a Nadaraya-Watson Gaussian-kernel regression on (rho, log2 d) (standardised),
  bandwidth by leave-one-out CV.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
from item_A_analysis import write_csv  # noqa: E402

OUT = CM.RES / "C"
RAW = OUT / "raw"
LAMS = ("0", "0.01", "0.1", "1")
FEATS = ("inv_d", "inv_G", "inv_N", "rho", "d_over_N", "lam")


def load():
    rows = {}
    for p in sorted(glob.glob(str(RAW / "shard_*.jsonl"))):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows[r["unit"]] = r
    return rows


def boot_ci(x, seed, nb=10000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bm = x[rng.integers(0, len(x), (nb, len(x)))].mean(1)
    return float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))


# ------------------------------------------------------------------ C1 / C2
def c1_rows(R, P):
    rows = []
    for u, r in R.items():
        if not u.startswith("c1|"):
            continue
        p = P[u]
        for lam in LAMS:
            k = f"maha_u_l{lam}"
            if f"{k}|dq_th" not in r:
                continue
            dq_s, dq_t, s_s, s_t = r[f"{k}|dq_sim"], r[f"{k}|dq_th"], r[f"{k}|s_sim"], r[f"{k}|s_th"]
            ratio = dq_s / dq_t if dq_t != 0 else float("nan")
            row = dict(unit=u, d=p["d"], rho=p["rho"], G=p["G"], n=p["n"], N=p["G"] * p["n"], lam=float(lam), block=p["block"],
                       in_R0=p["in_R0"], R=r["R"], dq_sim=dq_s, dq_sim_se=r[f"{k}|dq_sim|se"], dq_th=dq_t, ratio=ratio,
                       err=abs(ratio - 1), s_sim=s_s, s_th=s_t, s_relerr=abs(s_s / s_t - 1),
                       bound_rho_s_violation_sim=bool(abs(dq_s) >= p["rho"] * s_s), k1_max_abs_dev=r["k1_max_abs_dev"])
            for o in ("O1c1", "O1c2", "O1c4", "O2c0.5", "O2c1", "O2c2"):
                if f"{k}|{o}|pred_add" in r:
                    dl = r[f"{k}|{o}|delta"]
                    row[f"delta_{o}"] = dl
                    row[f"err_add_{o}"] = abs(dl - r[f"{k}|{o}|pred_add"])
                    row[f"err_gauss_{o}"] = abs(dl - r[f"{k}|{o}|pred_gauss"])
            rows.append(row)
    return rows


def feats(r):
    return [1 / r["d"], 1 / r["G"], 1 / r["N"], r["rho"], r["d"] / r["N"], r["lam"]]


def fit_correction(train):
    from sklearn.linear_model import Ridge
    ok = [r for r in train if r["ratio"] > 0 and np.isfinite(r["ratio"])]
    X = np.array([feats(r) for r in ok])
    y = np.log([r["ratio"] for r in ok])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    groups = np.array([r["d"] for r in ok])
    grid = np.logspace(-4, 4, 41)
    cv = []
    for a in grid:
        se = 0.0
        for gv in np.unique(groups):
            m = groups != gv
            mdl = Ridge(alpha=a).fit(Z[m], y[m])
            se += float(((mdl.predict(Z[~m]) - y[~m]) ** 2).sum())
        cv.append(se / len(y))
    a = float(grid[int(np.argmin(cv))])
    mdl = Ridge(alpha=a).fit(Z, y)
    beta_std = mdl.coef_
    coef = beta_std / sd
    b0 = float(mdl.intercept_ - (beta_std * mu / sd).sum())
    return dict(alpha=a, cv_mse=dict(zip(map(float, grid), cv)), n_train_rows=len(ok), n_train_dropped_nonpositive=len(train) - len(ok),
                beta0=b0, beta=dict(zip(FEATS, map(float, coef))), cv_scheme="grouped by d (leave-one-d-out over the 4 TRAIN d values)")


def predict_corr(c, r):
    return float(np.exp(c["beta0"] + sum(c["beta"][f] * v for f, v in zip(FEATS, feats(r)))))


def block_stats(rows, label, corr, seed):
    if not rows:
        return dict(block=label, n_rows=0, n_configs=0)
    e = np.array([r["err"] for r in rows])
    ec = np.array([abs(r["ratio"] / predict_corr(corr, r) - 1) for r in rows])
    out = dict(block=label, n_rows=len(rows), n_configs=len({r["unit"] for r in rows}),
               median_err=float(np.median(e)), p90_err=float(np.quantile(e, 0.9)),
               median_err_corrected=float(np.median(ec)), p90_err_corrected=float(np.quantile(ec, 0.9)),
               median_s_relerr=float(np.median([r["s_relerr"] for r in rows])))
    lo, hi = boot_ci(e - ec, seed)
    out.update(mean_err_reduction=float((e - ec).mean()), err_reduction_ci_lo=lo, err_reduction_ci_hi=hi)
    for o in ("O1c2", "O2c1"):
        ea = [r[f"err_add_{o}"] for r in rows if f"err_add_{o}" in r]
        eg = [r[f"err_gauss_{o}"] for r in rows if f"err_gauss_{o}" in r]
        if ea:
            out[f"median_delta_err_add_{o}"] = float(np.median(ea))
            out[f"median_delta_err_gauss_{o}"] = float(np.median(eg))
    return out


# ------------------------------------------------------------------ C3
def isotonic_r2(x, y):
    from sklearn.isotonic import IsotonicRegression
    f = IsotonicRegression(increasing="auto").fit(x, y).predict(x)
    return 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)


def nw_r2(Z, y):
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    best = None
    for h in np.logspace(-2, 1, 31):
        W = np.exp(-D2 / (2 * h * h))
        Wl = W.copy()
        np.fill_diagonal(Wl, 0)
        den = Wl.sum(1)
        if np.any(den <= 0):
            continue
        loo = (Wl @ y) / den
        cv = float(np.mean((y - loo) ** 2))
        if best is None or cv < best[0]:
            best = (cv, h)
    h = best[1]
    W = np.exp(-D2 / (2 * h * h))
    f = (W @ y) / W.sum(1)
    return 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2), h


def c3(R, P):
    rows = []
    for u, r in R.items():
        if not u.startswith("c3|"):
            continue
        p = P[u]
        row = dict(unit=u, **p, x=p["rho"] * np.sqrt(p["d"]), R=r["R"])
        for k in ("knn1", "knn5", "knn20", "maha_lw", "vim"):
            if f"{k}|O3|delta" in r:
                row[f"delta_{k}_O3"] = r[f"{k}|O3|delta"]
                row[f"delta_{k}_O1c2"] = r.get(f"{k}|O1c2|delta")
        rows.append(row)
    res = {}
    for k in ("knn1", "knn5"):
        pts = [r for r in rows if f"delta_{k}_O3" in r and 0.02 < r[f"delta_{k}_O3"] < 0.45]
        if len(pts) < 5:
            res[k] = dict(n_interior=len(pts), score=float("nan"))
            continue
        y = np.array([r[f"delta_{k}_O3"] for r in pts])
        x = np.array([r["x"] for r in pts])
        Z = np.c_[[r["rho"] for r in pts], np.log2([r["d"] for r in pts])]
        Z = (Z - Z.mean(0)) / Z.std(0)
        r1 = isotonic_r2(x, y)
        r2, h = nw_r2(Z, y)
        res[k] = dict(n_interior=len(pts), r2_isotonic_x=float(r1), r2_kernel_2d=float(r2), bandwidth=float(h), score=float(r1 / r2))
    # n-independence at equal x (k <= n): spread of Delta over n within (d, rho, G)
    spread = {}
    for k in ("knn1", "knn5"):
        cells = {}
        for r in rows:
            if f"delta_{k}_O3" in r:
                cells.setdefault((r["d"], r["rho"], r["G"]), []).append(r[f"delta_{k}_O3"])
        sp = [max(v) - min(v) for v in cells.values() if len(v) == 3]
        spread[k] = dict(n_cells=len(sp), median_range_over_n=float(np.median(sp)) if sp else None,
                         p90_range_over_n=float(np.quantile(sp, 0.9)) if sp else None)
    return rows, res, spread


# ------------------------------------------------------------------ C4
def c4(R, P):
    rows = []
    for u, r in R.items():
        if u.startswith("c4tv|"):
            p = P[u]
            rows.append(dict(unit=u, kind="exact_tv", law="gauss", scorer="oracle", d=p["d"], rho=p["rho"], G=p["G"], cap=r["cap"],
                             cap_nonvacuous=r["cap_nonvacuous"], E_TV=r["E_TV"], E_delta=r["E_delta_oracle"], se=r["E_delta_oracle_se"],
                             ub999=r["ub999_oracle"], tightness=r["E_delta_oracle"] / r["cap"] if r["cap"] > 0 else float("nan"),
                             exceeds_cap=bool(r["ub999_oracle"] > r["cap"]), mass_in_min=r["mass_in_min"], mass_out_min=r["mass_out_min"]))
        elif u.startswith("c4sc|"):
            p = P[u]
            for k in [k.split("|")[0] for k in r if k.endswith("|E_delta")]:
                rows.append(dict(unit=u, kind="scorer", law=p["law"], scorer=k, d=p["d"], rho=p["rho"], G=p["G"], cap=r["cap"],
                                 cap_nonvacuous=r["cap_nonvacuous"], E_delta=r[f"{k}|E_delta"], se=r[f"{k}|se"], ub999=r[f"{k}|ub999"],
                                 tightness=r[f"{k}|tightness"], exceeds_cap=r[f"{k}|exceeds_cap"]))
    viol = [r for r in rows if r["law"] == "gauss" and r["cap_nonvacuous"] and r["exceeds_cap"]]
    viol_other = [r for r in rows if r["law"] != "gauss" and r["cap_nonvacuous"] and r["exceeds_cap"]]
    small_g = {}
    for r in rows:
        if r["cap"] < 0.1:
            key = f"d{r['d']}_rho{r['rho']}"
            small_g[key] = min(small_g.get(key, 10 ** 9), r["G"])
    nv = [r for r in rows if r["cap_nonvacuous"] and r["law"] == "gauss"]
    tight = dict(n_nonvacuous_gauss_rows=len(nv),
                 median_tightness_nonvacuous=float(np.median([r["tightness"] for r in nv])) if nv else None,
                 max_tightness_nonvacuous=float(np.max([r["tightness"] for r in nv])) if nv else None,
                 smallest_G_cap_below_0_1=small_g)
    return rows, viol, viol_other, tight


# ------------------------------------------------------------------ C5
def c5(R, P):
    rows = []
    for u, r in R.items():
        if not u.startswith("c5|"):
            continue
        p = P[u]
        base = dict(unit=u, variant=p["variant"], block=p["block"], in_R0=p["in_R0"], d=p["d"], rho=p["rho"], G=p["G"], n=p["n"],
                    **{k: v for k, v in p.items() if k.startswith("v_")}, **{k: v for k, v in r.items() if k.startswith("diag_")})
        for pre, tag in (("", "raw_ridge"), ("wh|", "whitened_ridge")):
            if pre and f"{pre}R" not in r:
                continue
            for lam in LAMS:
                k = f"{pre}maha_u_l{lam}"
                if f"{k}|dq_th" not in r:
                    continue
                ratio = r[f"{k}|dq_sim"] / r[f"{k}|dq_th"]
                row = dict(base, ridge=tag, lam=float(lam), dq_sim=r[f"{k}|dq_sim"], dq_th=r[f"{k}|dq_th"], err=abs(ratio - 1))
                if f"{k}|O1c2|pred_add" in r:
                    row["delta_abs_err_O1c2"] = abs(r[f"{k}|O1c2|delta"] - r[f"{k}|O1c2|pred_add"])
                rows.append(row)
            if p["variant"] == "d" and not pre:
                for lam in LAMS:
                    k = f"cc_l{lam}"
                    if f"{k}|dq_sim" in r and f"maha_u_l{lam}|dq_th" in r:
                        rows.append(dict(base, ridge="class_conditional", lam=float(lam), dq_sim=r[f"{k}|dq_sim"],
                                         dq_th=r[f"maha_u_l{lam}|dq_th"], err=abs(r[f"{k}|dq_sim"] / r[f"maha_u_l{lam}|dq_th"] - 1)))
    summ = []
    keys = sorted({(r["variant"], r["ridge"]) for r in rows})
    for v, rd in keys:
        rs = [r for r in rows if r["variant"] == v and r["ridge"] == rd]
        de = [r["delta_abs_err_O1c2"] for r in rs if "delta_abs_err_O1c2" in r]
        summ.append(dict(variant=v, ridge=rd, n_rows=len(rs), n_configs=len({r["unit"] for r in rs}),
                         frac_within_0_25=float(np.mean([r["err"] <= 0.25 for r in rs])),
                         frac_within_0_10=float(np.mean([r["err"] <= 0.10 for r in rs])),
                         frac_delta_err_le_0_02=float(np.mean([x <= 0.02 for x in de])) if de else None))
    # failure boundary: one row per config (lambda = 0.1, raw ridge / class-agnostic), fail := err > 0.25
    fb = {}
    pts = [r for r in rows if r["lam"] == 0.1 and r["ridge"] == "raw_ridge"]
    if pts:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        def X_of(rs):
            return np.array([[r["diag_logcond"], r["diag_pr_over_d"], r["diag_kurtosis"], r["diag_delta_lw"],
                              np.log(r["diag_N_over_d"]), np.log(r["diag_G"]), r["diag_class_sep"]] for r in rs])
        tr = [r for r in pts if r["variant"] in ("a", "b", "d")]
        te = [r for r in pts if r["variant"] in ("c", "e")]
        if tr and te and len({r["err"] > 0.25 for r in tr}) == 2:
            Xt, yt = X_of(tr), np.array([r["err"] > 0.25 for r in tr])
            mu, sd = Xt.mean(0), Xt.std(0)
            sd[sd == 0] = 1
            lr = LogisticRegression(max_iter=5000).fit((Xt - mu) / sd, yt)
            ye = np.array([r["err"] > 0.25 for r in te])
            pe = lr.predict_proba((X_of(te) - mu) / sd)[:, 1]
            auc = float(roc_auc_score(ye, pe)) if len(set(ye)) == 2 else float("nan")
            fb = dict(n_train=len(tr), n_test=len(te), fail_rate_train=float(yt.mean()), fail_rate_test=float(ye.mean()),
                      auroc_heldout=auc, auroc_floor_0_5=max(auc, 0.5) if np.isfinite(auc) else None,
                      features=["log_cond_T", "PR_over_d", "excess_kurtosis_whitened", "delta_LW", "log_N_over_d", "log_G", "class_sep"],
                      coef_standardised=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]),
                      a1_cells_map="not run: A = NO-GO (order: skip C5's real-cell validity map)")
    return rows, summ, fb


def figures(c1, c3rows, c4rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for blk in ("TRAIN", "TEST-interp", "TEST-extrap"):
        rs = [r for r in c1 if r["block"] == blk]
        if not rs:
            continue
        ds, gs = sorted({r["d"] for r in rs}), sorted({r["G"] for r in rs})
        M = np.full((len(ds), len(gs)), np.nan)
        for i, d in enumerate(ds):
            for j, g in enumerate(gs):
                v = [r["err"] for r in rs if r["d"] == d and r["G"] == g]
                if v:
                    M[i, j] = np.median(v)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(np.clip(M, 0, 1), origin="lower", cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(gs)), gs)
        ax.set_yticks(range(len(ds)), ds)
        ax.set_xlabel("G")
        ax.set_ylabel("d")
        ax.set_title(f"C1 {blk}: median |dQ_sim/dQ_th - 1| (clipped at 1)")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(OUT / f"c1_heatmap_{blk}.png", dpi=120)
        plt.close(fig)
    if c3rows:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, k in zip(ax, ("knn1", "knn5")):
            rs = [r for r in c3rows if f"delta_{k}_O3" in r]
            sc = a.scatter([r["x"] for r in rs], [r[f"delta_{k}_O3"] for r in rs], c=np.log2([r["d"] for r in rs]), s=10)
            a.set_xscale("log")
            a.set_xlabel("rho sqrt(d)")
            a.set_ylabel("Delta (null OOD)")
            a.set_title(k)
            fig.colorbar(sc, ax=a, label="log2 d")
        fig.tight_layout()
        fig.savefig(OUT / "c3_collapse.png", dpi=120)
        plt.close(fig)
    nv = [r for r in c4rows if r["law"] == "gauss"]
    if nv:
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in sorted({r["scorer"] for r in nv}):
            rs = [r for r in nv if r["scorer"] == s]
            ax.scatter([r["cap"] for r in rs], [r["E_delta"] for r in rs], s=10, label=s)
        lim = [1e-3, 1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.axvline(0.45, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("cap K3")
        ax.set_ylabel("E[Delta]")
        ax.legend(fontsize=7)
        ax.set_title("C4: E[Delta] vs cap (Gaussian)")
        fig.tight_layout()
        fig.savefig(OUT / "c4_tightness.png", dpi=120)
        plt.close(fig)


def main():
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_C.json"))
    P = add["unit_params"]
    order = add["work_lists"][0]["execution_order"]
    R = load()
    tests = []
    req = {k: [u for u in order if u.startswith(k + "|")] for k in ("c1", "c3", "c4tv", "c4sc", "c5")}
    done = {k: sum(u in R for u in v) for k, v in req.items()}
    s8 = (OUT / "STOP_S8.json").exists()

    c1 = c1_rows(R, P)
    train = [r for r in c1 if r["block"] == "TRAIN"]
    corr = fit_correction(train) if train else None
    blocks = []
    if corr:
        for lab, rs in (("TRAIN", train),
                        ("TEST-interp", [r for r in c1 if r["block"] == "TEST-interp"]),
                        ("TEST-interp in R0", [r for r in c1 if r["block"] == "TEST-interp" and r["in_R0"]]),
                        ("TEST-extrap", [r for r in c1 if r["block"] == "TEST-extrap"]),
                        ("TEST-extrap in R0", [r for r in c1 if r["block"] == "TEST-extrap" and r["in_R0"]])):
            blocks.append(block_stats(rs, lab, corr, 301))
        CM.atomic_write_text(OUT / "c1_correction.json", json.dumps(corr, indent=1))
    write_csv(OUT / "c1_blocks.csv", blocks)
    write_csv(OUT / "c1_rows.csv", c1)
    bi = {b["block"]: b for b in blocks}
    ok_blocks = [bi.get(k, {}) for k in ("TEST-interp in R0", "TEST-extrap in R0")]
    if not corr or any(b.get("n_rows", 0) == 0 for b in ok_blocks):
        C_i = "not evaluable"
    else:
        C_i = "pass" if all(b["median_err"] <= 0.10 and b["p90_err"] <= 0.25 for b in ok_blocks) else "fail"
    tests.append(dict(test="C-i: uncorrected K2 median |dQ_sim/dQ_th - 1| <= 0.10 and p90 <= 0.25 on TEST-interp and TEST-extrap (in R0)",
                      status=C_i, value="; ".join(f"{b.get('block')}: median {b.get('median_err', float('nan')):.3f}, p90 {b.get('p90_err', float('nan')):.3f}, "
                                                    f"{b.get('n_configs', 0)} configs" for b in ok_blocks)
                      + f"; C1 units completed {done['c1']}/{len(req['c1'])}", reference="median <= 0.10, p90 <= 0.25", **{"pass": C_i == "pass"}))
    for b in blocks:
        if b["block"].startswith("TEST") and b.get("n_rows"):
            claim = b["err_reduction_ci_lo"] > 0
            tests.append(dict(test=f"C1 corrected formula (post-hoc claim): error reduction on {b['block']}", status="post-hoc",
                              value=f"mean reduction {b['mean_err_reduction']:.4f} [{b['err_reduction_ci_lo']:.4f}, {b['err_reduction_ci_hi']:.4f}]; "
                                    f"corrected median {b['median_err_corrected']:.3f}", reference="LB > 0 to claim", **{"pass": claim}))
    # C2
    write_csv(OUT / "c2_auroc.csv", [{k: v for k, v in r.items() if k.startswith(("unit", "d", "rho", "G", "n", "N", "lam", "block", "delta_", "err_add", "err_gauss", "k1_", "bound_"))}
                                     for r in c1])
    k1 = max([r["k1_max_abs_dev"] for r in c1], default=float("nan"))
    nviol = sum(r["bound_rho_s_violation_sim"] for r in c1)
    tests.append(dict(test="C2 implementation check: omega-linearity |Delta(omega) - omega Delta(1)| <= 1e-12", status="pass" if k1 <= 1e-12 else "fail",
                      value=f"max {k1:.2e}", reference="<= 1e-12", **{"pass": k1 <= 1e-12}))
    tests.append(dict(test="C2 report: bound |dQ| < rho s violated (simulated dQ, s)", status="report-only", value=f"{nviol}/{len(c1)} rows",
                      reference="count", **{"pass": "na"}))
    # C3
    c3rows, coll, spread = c3(R, P)
    write_csv(OUT / "c3_collapse.csv", c3rows)
    if done["c3"] < len(req["c3"]) or any(not np.isfinite(coll[k]["score"]) for k in coll):
        C_iii = "not evaluable"
    else:
        C_iii = "supported" if all(coll[k]["score"] >= 0.90 for k in coll) else "not supported"
    tests.append(dict(test="C-iii: collapse score >= 0.90 in the interior region (kNN-1 and kNN-5, null OOD)", status=C_iii,
                      value="; ".join(f"{k}: {v['score']:.3f} (n={v['n_interior']})" for k, v in coll.items()), reference=">= 0.90",
                      **{"pass": C_iii == "supported"}))
    # C4
    c4rows, viol, viol_other, tight = c4(R, P)
    write_csv(OUT / "c4_cap.csv", c4rows)
    if s8 or viol:
        C_ii = "fail"
    elif done["c4tv"] < len(req["c4tv"]) or done["c4sc"] < len(req["c4sc"]):
        C_ii = "not evaluable"
    else:
        C_ii = "pass"
    tests.append(dict(test="C-ii: no S8 violation (non-vacuous Gaussian configuration with 99.9 % UB of E[Delta] > cap)", status=C_ii,
                      value=f"{len(viol)} violations; C4 units {done['c4tv']}/{len(req['c4tv'])} exact-TV, {done['c4sc']}/{len(req['c4sc'])} scorer; "
                            f"median tightness (non-vacuous) {tight['median_tightness_nonvacuous']}", reference="none", **{"pass": C_ii == "pass"}))
    tests.append(dict(test="C4 report: heavy-tailed / anisotropic repeats exceeding the cap (not claimed)", status="report-only",
                      value=f"{len(viol_other)} rows", reference="reported, not claimed", **{"pass": "na"}))
    # C5
    c5rows, c5sum, fb = c5(R, P)
    write_csv(OUT / "c5_variants.csv", c5sum)
    write_csv(OUT / "c5_rows.csv", c5rows)
    CM.atomic_write_text(OUT / "c5_failure_boundary.json", json.dumps(fb, indent=1))
    bw = [s for s in c5sum if s["variant"] == "b" and s["ridge"] == "whitened_ridge"]
    if done["c5"] < len(req["c5"]) and not bw:
        C_iv = "not run" if done["c5"] == 0 else "not evaluable"
    elif done["c5"] < len(req["c5"]):
        C_iv = "not evaluable"
    else:
        C_iv = "supported" if bw and bw[0]["frac_within_0_25"] >= 0.80 else "not supported"
    tests.append(dict(test="C-iv: variant (b) proportional with whitened ridge, >= 80 % of configs within 0.25", status=C_iv,
                      value=(f"{bw[0]['frac_within_0_25']:.3f}" if bw else "") + f"; C5 units {done['c5']}/{len(req['c5'])}",
                      reference=">= 0.80", **{"pass": C_iv == "supported"}))
    for s in c5sum:
        tests.append(dict(test=f"C5 report: variant ({s['variant']}) {s['ridge']}", status="report-only",
                          value=f"within 0.25: {s['frac_within_0_25']:.3f}; within 0.10: {s['frac_within_0_10']:.3f}; Delta err <= 0.02: {s['frac_delta_err_le_0_02']}",
                          reference="stated separately", **{"pass": "na"}))
    if C_i == "fail" and C_ii == "fail":
        verdict = "NO-GO"
    elif "not evaluable" in (C_i, C_ii):
        verdict = "INCONCLUSIVE"
    elif C_i == "pass" and C_ii == "pass":
        verdict = "GO"
    else:
        verdict = "PARTIAL"
    with open(OUT / "tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["test", "status", "value", "reference", "pass"])
        w.writeheader()
        w.writerows(tests)
    secs = sum(float(p.read_text() or 0) for p in RAW.glob("shard_*.secs"))
    v = dict(item="C", verdict=verdict, C_i=C_i, C_ii=C_ii, C_iii=C_iii, C_iv=C_iv, blocks=blocks, collapse=coll, n_independence=spread,
             cap=tight, s8=s8, units_done=done, units_required={k: len(v) for k, v in req.items()}, gpu_hours=secs / 3600,
             rule="GO: C-i and C-ii; PARTIAL: exactly one; NO-GO: neither; a core criterion not evaluable (and none failed) -> INCONCLUSIVE")
    CM.atomic_write_text(OUT / "verdict.json", json.dumps(v, indent=1, default=str))
    figures(c1, c3rows, c4rows)
    print(json.dumps({k: v[k] for k in ("verdict", "C_i", "C_ii", "C_iii", "C_iv", "units_done")}))


if __name__ == "__main__":
    main()
