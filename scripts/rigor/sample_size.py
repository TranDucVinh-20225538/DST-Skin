#!/usr/bin/env python
"""Sample-size / ranking-stability analysis (decisions/precommit_sample_size_2026-10-05.md).

Stages:
  umat     per (arch, seed, score): patient-pair AUROC numerators U[i, j] (ID patient i x OOD
           patient j) from the per-patch score caches of a tree, so that the AUROC of any patient
           subset / bootstrap is sum_ij w_i v_j U_ij / (sum w_i n_i)(sum v_j m_j).
  axes     step 1: seed, architecture, ID-patient, OOD-patient axes and the seed x arch grid.
  varcomp  step 2: variance components (statsmodels MixedLM, REML).
  features step 3: covariance spectrum of the train features (Camelyon 8 archs x 5 seeds, CIFAR-10).

Usage: python scripts/rigor/sample_size.py STAGE [--tree stable_maha_vim8|stable]
Large intermediates: outputs/rigor_pack/sample_size/ (gitignored); tables and figures:
outputs/reports/rigor_pack/sample_size/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from persample import clusters_for  # noqa: E402

DOMAIN = "camelyon17"
KEY = {v: k for k, v in C.DISPLAY.items()}
PRIMARY = "stable_maha_vim8"
TREES = {PRIMARY: "primary", "stable": "appendix"}
N_DRAWS = 1000
BASE_SEED = 20261005
TAU_MIN, TOP1_MIN = 0.8, 0.8
B_VARCOMP = 100
CIFAR_FEATS = Path("/data2/hpcshared/ood-numstab/features/cifar10")


def tree_root(tree: str) -> Path:
    return C.REPO / "outputs/rigor_pack/stability/variants" / tree


def work_dir() -> Path:
    return C.ensure_dir(C.default_out(C.REPO) / "sample_size")


def rep_dir() -> Path:
    return C.ensure_dir(C.default_reports(C.REPO) / "sample_size")


def gen(i: int) -> np.random.Generator:
    return np.random.default_rng([BASE_SEED, i])


# ---------------------------------------------------------------------------------------------
# U matrices
# ---------------------------------------------------------------------------------------------
def pair_numerators(p, q, idc, oc, n_i, n_o):
    """U[i, j] = sum over ID patches of patient i and OOD patches of patient j of [p > q] + 0.5 [p == q]."""
    U = np.zeros((n_i, n_o))
    for j in range(n_o):
        qs = np.sort(q[oc == j])
        lo = np.searchsorted(qs, p, "left")
        hi = np.searchsorted(qs, p, "right")
        U[:, j] = np.bincount(idc, weights=lo + 0.5 * (hi - lo), minlength=n_i)
    return U


def umat(tree: str) -> None:
    root = tree_root(tree)
    archs, seeds, methods = list(C.ARCHS), list(C.SEEDS), list(C.METHODS_ORDER)
    codes = None
    U = None
    for ai, a in enumerate(archs):
        for si, s in enumerate(seeds):
            z = np.load(C.score_cache_path(C.default_out(root), DOMAIN, a, s))
            if codes is None:
                idc, oc, names, note = clusters_for(DOMAIN, C.REPO, len(z["id_labels"]), len(z["ood_labels"]),
                                                    z["id_labels"], z["ood_labels"])
                if idc is None:
                    raise SystemExit("STOP: " + note)
                print(note, flush=True)
                codes = (idc, oc)
                n_i, n_o = int(idc.max()) + 1, int(oc.max()) + 1
                U = np.zeros((len(archs), len(seeds), len(methods), n_i, n_o))
                n_id = np.bincount(idc, minlength=n_i).astype(np.float64)
                n_ood = np.bincount(oc, minlength=n_o).astype(np.float64)
            for mi, m in enumerate(methods):
                p = np.asarray(z["id_" + KEY[m]], np.float32).astype(np.float64)
                q = np.asarray(z["ood_" + KEY[m]], np.float32).astype(np.float64)
                U[ai, si, mi] = pair_numerators(p, q, codes[0], codes[1], n_i, n_o)
        print("umat %s: %s done" % (tree, a), flush=True)
    auc = U.sum(axis=(3, 4)) / (n_id.sum() * n_ood.sum())
    cube = C.load_cube(root, DOMAIN, archs, seeds)
    diff = np.abs(auc - cube)
    chk = pd.DataFrame([{"tree": tree, "arch": a, "seed": s, "score": m, "auroc_from_cache": auc[i, j, k],
                         "auroc_tree_csv": cube[i, j, k], "abs_diff": diff[i, j, k]}
                        for i, a in enumerate(archs) for j, s in enumerate(seeds) for k, m in enumerate(methods)])
    chk.to_csv(rep_dir() / ("auroc_check_%s.csv" % tree), index=False)
    np.savez(work_dir() / ("U_%s.npz" % tree), U=U, n_id=n_id, n_ood=n_ood,
             archs=np.array(archs), seeds=np.array(seeds), methods=np.array(methods))
    print("max |AUROC(cache) - AUROC(tree csv)| = %.3g" % np.nanmax(diff), flush=True)
    if not np.nanmax(diff) <= 1e-6:
        raise SystemExit("STOP: AUROC from score caches differs from the tree CSV by > 1e-6 (see auroc_check)")


def load_u(tree: str):
    z = np.load(work_dir() / ("U_%s.npz" % tree))
    return z["U"], z["n_id"], z["n_ood"]


def auc_w(U, n_id, n_ood, wi, wo):
    """wi (D, n_i), wo (D, n_o) -> AUROC (D, A, S, M)."""
    num = np.einsum("asmij,di,dj->dasm", U, wi, wo, optimize=True)
    den = (wi @ n_id) * (wo @ n_ood)
    return num / den[:, None, None, None]


# ---------------------------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------------------------
def ranks_desc(x):
    """Average ranks along the last axis, 1 = highest."""
    from scipy.stats import rankdata
    return rankdata(-x, axis=-1, method="average")


def tie_sum(x):
    """sum over tie groups of t^3 - t along the last axis (= sum_i (c_i^2 - 1))."""
    eq = (x[..., :, None] == x[..., None, :]).sum(-1).astype(np.float64)
    return (eq ** 2 - 1).sum(-1)


def w_batch(x):
    """x (..., k, n) values -> tie-corrected Kendall W across the k raters."""
    return C.kendall_w_batch(ranks_desc(x), tie_sum(x))


def tau_b(x, ref):
    """x (D, n), ref (n,) -> Kendall tau-b per row."""
    n = len(ref)
    i, j = np.triu_indices(n, 1)
    sx = np.sign(x[:, i] - x[:, j])
    sr = np.sign(ref[i] - ref[j])
    n0 = len(i)
    n1 = (sx == 0).sum(1)
    n2 = (sr == 0).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sx * sr).sum(1) / np.sqrt((n0 - n1) * (n0 - n2))


def top1_match(x, ref):
    top = int(np.argmax(ref))
    ismax = x == x.max(1, keepdims=True)
    return ismax[:, top] / ismax.sum(1)


def summarise(axis, size, full, tau, top1, w):
    q = lambda v, p: float(np.nanpercentile(v, p)) if np.isfinite(v).any() else np.nan
    return {"axis": axis, "size": size, "full_size": full, "n_draws": len(tau),
            "tau_median": q(tau, 50), "tau_q25": q(tau, 25), "tau_q75": q(tau, 75), "tau_mean": float(np.nanmean(tau)),
            "p_top1": float(np.mean(top1)),
            "w_median": q(w, 50), "w_q25": q(w, 25), "w_q75": q(w, 75),
            "reliable": bool(q(tau, 50) >= TAU_MIN and np.mean(top1) >= TOP1_MIN)}


def subsets(rng, n, k, D):
    """D draws of k of n items without replacement -> boolean mask (D, n)."""
    idx = np.argsort(rng.random((D, n)), axis=1)[:, :k]
    mask = np.zeros((D, n), bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def min_size(df):
    """Smallest sub-full size such that it and every larger sub-full size are reliable."""
    sub = df[df["size"] < df.full_size].sort_values("size")
    ok = sub.reliable.to_numpy()
    best = None
    for i in range(len(sub) - 1, -1, -1):
        if not ok[i]:
            break
        best = int(sub["size"].iloc[i])
    return best


# ---------------------------------------------------------------------------------------------
# step 1
# ---------------------------------------------------------------------------------------------
def axes(tree: str) -> None:
    U, n_id, n_ood = load_u(tree)
    A, S, M, NI, NO = U.shape
    A0 = U.sum(axis=(3, 4)) / (n_id.sum() * n_ood.sum())  # (A, S, M)
    ref = A0.mean(axis=(0, 1))
    rows, grid = [], []
    D = N_DRAWS

    # seeds
    rng = gen(0)
    for k in range(1, S + 1):
        ms = subsets(rng, S, k, D)  # (D, S)
        x = np.einsum("asm,ds->dm", A0, ms) / (A * k)
        if k >= 2:
            w = np.full(D, np.nan)
            for d in range(D):
                w[d] = np.mean(w_batch(A0[:, ms[d], :]))
        else:
            w = np.full(D, np.nan)
        rows.append(summarise("seed", k, S, tau_b(x, ref), top1_match(x, ref), w))

    # architectures
    rng = gen(1)
    w_seed_arch = None
    for m in range(2, A + 1):
        ma = subsets(rng, A, m, D)
        x = np.einsum("asm,da->dm", A0, ma) / (m * S)
        w = np.array([np.mean(w_batch(np.transpose(A0[ma[d]], (1, 0, 2)))) for d in range(D)])
        rows.append(summarise("arch", m, A, tau_b(x, ref), top1_match(x, ref), w))

    # patients
    for name, gi, n_tot in (("id_patients", 2, NI), ("ood_patients", 3, NO)):
        rng = gen(gi)
        for n in range(2, n_tot + 1):
            mk = subsets(rng, n_tot, n, D).astype(np.float64)
            if name == "id_patients":
                auc = auc_w(U, n_id, n_ood, mk, np.ones((D, NO)))
            else:
                auc = auc_w(U, n_id, n_ood, np.ones((D, NI)), mk)
            x = auc.mean(axis=(1, 2))
            w = w_batch(np.transpose(auc, (0, 2, 1, 3))).mean(axis=1)  # (D, S, A, M) -> W per seed
            rows.append(summarise(name, n, n_tot, tau_b(x, ref), top1_match(x, ref), w))

    # grid
    rng = gen(4)
    for k in range(1, S + 1):
        for m in range(2, A + 1):
            ms = subsets(rng, S, k, D)
            ma = subsets(rng, A, m, D)
            x = np.einsum("asm,da,ds->dm", A0, ma, ms) / (m * k)
            w = np.empty(D)
            for d in range(D):
                w[d] = np.mean(w_batch(np.transpose(A0[np.ix_(ma[d], ms[d])], (1, 0, 2))))
            r = summarise("grid", "%dx%d" % (k, m), "%dx%d" % (S, A), tau_b(x, ref), top1_match(x, ref), w)
            r.update(k_seeds=k, m_archs=m)
            grid.append(r)

    df = pd.DataFrame(rows)
    df.insert(0, "tree", tree)
    gd = pd.DataFrame(grid)
    gd.insert(0, "tree", tree)
    rep = rep_dir()
    df.to_csv(rep / ("stability_axes_%s.csv" % tree), index=False)
    gd.to_csv(rep / ("stability_grid_%s.csv" % tree), index=False)
    mins = [{"tree": tree, "axis": ax, "full_size": int(g.full_size.iloc[0]),
             "min_reliable_size": (lambda v: v if v is not None else "không đạt trong dữ liệu hiện có")(min_size(g))}
            for ax, g in df.groupby("axis", sort=False)]
    pd.DataFrame(mins).to_csv(rep / ("min_size_%s.csv" % tree), index=False)
    pd.DataFrame({"score": C.METHODS_ORDER, "mean_auroc_full": ref,
                  "rank_full": ranks_desc(ref)}).to_csv(rep / ("reference_ranking_%s.csv" % tree), index=False)
    plot_axes(df, gd, tree)
    print(df.round(3).to_string(index=False))
    print(pd.DataFrame(mins).to_string(index=False))


def plot_axes(df, gd, tree):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"seed": "seeds (8 archs, all patients)", "arch": "architectures (5 seeds, all patients)",
              "id_patients": "ID patients (OOD full)", "ood_patients": "OOD Hospital-2 patients (ID full)"}
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, (name, t) in zip(axs, titles.items()):
        g = df[df.axis == name].sort_values("size")
        s = g["size"].astype(int)
        ax.fill_between(s, g.tau_q25, g.tau_q75, alpha=0.2, color="C0")
        ax.plot(s, g.tau_median, "o-", color="C0", label="median tau (IQR)")
        ax.plot(s, g.p_top1, "s-", color="C1", label="P(top-1 match)")
        ax.plot(s, g.w_median, "^--", color="C2", label="median W")
        ax.axhline(0.8, color="k", lw=0.8, ls=":")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("subset size")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Ranking stability vs subset size (%s)" % tree, fontsize=10)
    fig.tight_layout()
    fig.savefig(rep_dir() / ("stability_axes_%s.png" % tree), dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, t in zip(axs, ("tau_median", "p_top1"), ("median Kendall tau", "P(top-1 match)")):
        P = gd.pivot(index="k_seeds", columns="m_archs", values=col)
        im = ax.imshow(P.to_numpy(), vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                v = P.to_numpy()[i, j]
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=7,
                        color="w" if v < 0.8 else "k", fontweight="bold" if v >= 0.8 else "normal")
        ax.set_xticks(range(P.shape[1]), P.columns)
        ax.set_yticks(range(P.shape[0]), P.index)
        ax.set_xlabel("architectures m")
        ax.set_ylabel("seeds k")
        ax.set_title(t, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Seed x architecture grid (%s); bold = >= 0.8" % tree, fontsize=10)
    fig.tight_layout()
    fig.savefig(rep_dir() / ("stability_grid_%s.png" % tree), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# step 2
# ---------------------------------------------------------------------------------------------
def varcomp(tree: str) -> None:
    import statsmodels.formula.api as smf

    U, n_id, n_ood = load_u(tree)
    A, S, M, NI, NO = U.shape
    rng = gen(10)
    wi = np.empty((B_VARCOMP, NI))
    wo = np.empty((B_VARCOMP, NO))
    for b in range(B_VARCOMP):
        wi[b] = np.bincount(rng.integers(0, NI, NI), minlength=NI)
        wo[b] = np.bincount(rng.integers(0, NO, NO), minlength=NO)
    auc = auc_w(U, n_id, n_ood, wi, wo)  # (B, A, S, M)
    bb, aa, ss, mm = np.meshgrid(range(B_VARCOMP), range(A), range(S), range(M), indexing="ij")
    df = pd.DataFrame({"auroc_pct": 100.0 * auc.ravel(), "rep": bb.ravel(), "arch": aa.ravel(), "seed": ss.ravel(),
                       "score": mm.ravel()})
    df["archseed"] = df.arch * S + df.seed
    df["score_arch"] = df.score * A + df.arch
    df["score_archseed"] = df.score * A * S + df.archseed
    df["score_rep"] = df.score * B_VARCOMP + df.rep
    df["g"] = 1
    models = {
        "primary": {"arch": "0 + C(arch)", "seed": "0 + C(seed)", "replicate": "0 + C(rep)"},
        "secondary": {"arch": "0 + C(arch)", "arch:seed": "0 + C(archseed)", "replicate": "0 + C(rep)",
                      "score:arch": "0 + C(score_arch)", "score:arch:seed": "0 + C(score_archseed)",
                      "score:replicate": "0 + C(score_rep)"},
    }
    out = []
    for name, vc in models.items():
        t0 = time.time()
        md = smf.mixedlm("auroc_pct ~ C(score)", df, groups="g", re_formula="0", vc_formula=vc)
        fit = md.fit(reml=True, method=["lbfgs", "powell"])
        comps = dict(zip(md.exog_vc.names, np.asarray(fit.vcomp, dtype=float)))
        comps["residual"] = float(fit.scale)
        tot = sum(comps.values())
        fe = fit.fe_params.to_numpy()
        fe_eff = np.r_[0.0, fe[1:]]
        for c, v in comps.items():
            out.append({"tree": tree, "model": name, "component": c, "variance_auroc2": v / 1e4,
                        "sd_auroc": np.sqrt(max(v, 0.0)) / 100.0, "share_of_random": v / tot,
                        "converged": bool(fit.converged), "seconds": round(time.time() - t0, 1)})
        out.append({"tree": tree, "model": name, "component": "(fixed: score, variance of score means)",
                    "variance_auroc2": float(np.var(fe_eff)) / 1e4, "sd_auroc": float(np.std(fe_eff)) / 100.0,
                    "share_of_random": np.nan, "converged": bool(fit.converged),
                    "seconds": round(time.time() - t0, 1)})
        print(name, "converged", fit.converged, "%.0fs" % (time.time() - t0), flush=True)
    res = pd.DataFrame(out)
    res.to_csv(rep_dir() / ("variance_components_%s.csv" % tree), index=False)
    print(res.round(5).to_string(index=False))


# ---------------------------------------------------------------------------------------------
# step 3
# ---------------------------------------------------------------------------------------------
def spectrum(X, l2: bool, chunk: int = 20000):
    n, d = X.shape

    def rows(i):
        x = np.asarray(X[i:i + chunk], dtype=np.float64)
        if l2:
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return x

    mu = np.zeros(d)
    for i in range(0, n, chunk):
        mu += rows(i).sum(0)
    mu /= n
    G = np.zeros((d, d))
    for i in range(0, n, chunk):
        xc = rows(i) - mu
        G += xc.T @ xc
    ev = np.linalg.eigvalsh(G / n)
    lmax, lmin = float(ev.max()), float(ev.min())
    p = np.clip(ev, 0, None)
    p = p / p.sum()
    p = p[p > 0]
    return {"dim": d, "n": n, "lambda_max": lmax, "lambda_min": lmin,
            "cond": lmax / lmin if lmin > 0 else float("inf"),
            "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
            "n_eig_below_1e-6_max": int((ev < 1e-6 * lmax).sum())}


def features(_tree: str) -> None:
    import torch

    rows = []
    for a in C.ARCHS:
        for s in C.SEEDS:
            p = C.feature_path(C.REPO, DOMAIN, a, s)
            t0 = time.time()
            if not p.exists():
                rows.append({"dataset": "camelyon17", "arch": a, "seed": s, "note": "missing"})
                continue
            d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            X = d["train_feats"]
            X = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
            for l2 in (False, True):
                rows.append({"dataset": "camelyon17", "arch": a, "seed": s, "l2": l2, **spectrum(X, l2)})
            del d, X
            print("features %s s%d %.0fs" % (a, s, time.time() - t0), flush=True)
    for sd in sorted(CIFAR_FEATS.glob("s*")):
        X = np.load(sd / "id_train_feats.npy", mmap_mode="r")
        for l2 in (False, True):
            rows.append({"dataset": "cifar10", "arch": "resnet18_32x32", "seed": sd.name, "l2": l2, **spectrum(X, l2)})
    df = pd.DataFrame(rows)
    df.to_csv(rep_dir() / "feature_spectrum.csv", index=False)
    print(df.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["umat", "axes", "varcomp", "features"])
    ap.add_argument("--tree", default=PRIMARY, choices=list(TREES))
    args = ap.parse_args()
    t0 = time.time()
    {"umat": umat, "axes": axes, "varcomp": varcomp, "features": features}[args.stage](args.tree)
    print("stage %s (%s) %.0fs" % (args.stage, args.tree, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
