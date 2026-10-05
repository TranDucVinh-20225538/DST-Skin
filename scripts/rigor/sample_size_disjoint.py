#!/usr/bin/env python
"""Disjoint-partition sample size (decisions/precommit_isbi_patch_2026-10-05.md, section D).

As sample_size.py, but the ranking of a subset A is compared with the ranking of the DISJOINT
complement B instead of the full data. Seeds, archs, the seed x arch grid and OOD patients are
enumerated exhaustively; ID patients use 1000 draws. Reads the U matrices written by
`sample_size.py umat`.

Usage: python scripts/rigor/sample_size_disjoint.py --tree stable_maha_vim8|stable
Writes outputs/reports/rigor_pack/sample_size_disjoint/{stability_axes,stability_grid,min_size,
side_by_side}_<tree>.csv and figures.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import sample_size as SS  # noqa: E402

LABEL = "disjoint_partition"


def rep_dir() -> Path:
    return C.ensure_dir(C.default_reports(C.REPO) / "sample_size_disjoint")


def tau_rows(x, y):
    """Kendall tau-b between rows of x and y, both (D, n)."""
    i, j = np.triu_indices(x.shape[1], 1)
    sx, sy = np.sign(x[:, i] - x[:, j]), np.sign(y[:, i] - y[:, j])
    n0 = len(i)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sx * sy).sum(1) / np.sqrt((n0 - (sx == 0).sum(1)) * (n0 - (sy == 0).sum(1)))


def top1_rows(x, y):
    mx = x == x.max(1, keepdims=True)
    my = y == y.max(1, keepdims=True)
    return (mx & my).sum(1) / mx.sum(1)


def masks(n, k):
    combos = list(itertools.combinations(range(n), k))
    m = np.zeros((len(combos), n), bool)
    for r, c in enumerate(combos):
        m[r, list(c)] = True
    return m


def summ(axis, size, full, xa, xb, w):
    r = SS.summarise(axis, size, full, tau_rows(xa, xb), top1_rows(xa, xb), w)
    r["complement_size"] = (full - size) if isinstance(size, int) else None
    return r


def run(tree: str) -> None:
    U, n_id, n_ood = SS.load_u(tree)
    A, S, M, NI, NO = U.shape
    A0 = U.sum(axis=(3, 4)) / (n_id.sum() * n_ood.sum())
    rows, grid = [], []

    for k in range(1, S):
        ms = masks(S, k)
        xa = np.einsum("asm,ds->dm", A0, ms) / (A * k)
        xb = np.einsum("asm,ds->dm", A0, ~ms) / (A * (S - k))
        w = np.array([np.mean(SS.w_batch(A0[:, m, :])) if k >= 2 else np.nan for m in ms])
        rows.append(summ("seed", k, S, xa, xb, w))

    for m in range(2, A - 1):
        ma = masks(A, m)
        xa = np.einsum("asm,da->dm", A0, ma) / (m * S)
        xb = np.einsum("asm,da->dm", A0, ~ma) / ((A - m) * S)
        w = np.array([np.mean(SS.w_batch(np.transpose(A0[x], (1, 0, 2)))) for x in ma])
        rows.append(summ("arch", m, A, xa, xb, w))

    for name, n_tot, rng in (("id_patients", NI, SS.gen(20)), ("ood_patients", NO, None)):
        for n in range(2, n_tot - 1):
            mk = SS.subsets(rng, n_tot, n, SS.N_DRAWS) if rng is not None else masks(n_tot, n)
            D = len(mk)
            ones_i, ones_o = np.ones((D, NI)), np.ones((D, NO))
            if name == "id_patients":
                aa = SS.auc_w(U, n_id, n_ood, mk.astype(float), ones_o)
                ab = SS.auc_w(U, n_id, n_ood, (~mk).astype(float), ones_o)
            else:
                aa = SS.auc_w(U, n_id, n_ood, ones_i, mk.astype(float))
                ab = SS.auc_w(U, n_id, n_ood, ones_i, (~mk).astype(float))
            w = SS.w_batch(np.transpose(aa, (0, 2, 1, 3))).mean(axis=1)
            rows.append(summ(name, n, n_tot, aa.mean(axis=(1, 2)), ab.mean(axis=(1, 2)), w))

    for k in range(1, S):
        ms_all = masks(S, k)
        for m in range(2, A - 1):
            ma_all = masks(A, m)
            xa, xb, w = [], [], []
            for ms in ms_all:
                for ma in ma_all:
                    xa.append(A0[np.ix_(ma, ms)].mean(axis=(0, 1)))
                    xb.append(A0[np.ix_(~ma, ~ms)].mean(axis=(0, 1)))
                    w.append(np.mean(SS.w_batch(np.transpose(A0[np.ix_(ma, ms)], (1, 0, 2)))))
            r = summ("grid", "%dx%d" % (k, m), "%dx%d" % (S, A), np.array(xa), np.array(xb), np.array(w))
            r.update(k_seeds=k, m_archs=m)
            grid.append(r)

    df = pd.DataFrame(rows)
    df.insert(0, "analysis", LABEL)
    df.insert(0, "tree", tree)
    gd = pd.DataFrame(grid)
    gd.insert(0, "analysis", LABEL)
    gd.insert(0, "tree", tree)
    rep = rep_dir()
    df.to_csv(rep / ("stability_axes_%s.csv" % tree), index=False)
    gd.to_csv(rep / ("stability_grid_%s.csv" % tree), index=False)

    old = pd.read_csv(C.default_reports(C.REPO) / "sample_size" / ("stability_axes_%s.csv" % tree))
    oldg = pd.read_csv(C.default_reports(C.REPO) / "sample_size" / ("stability_grid_%s.csv" % tree))
    mins = []
    for ax, g in df.groupby("axis", sort=False):
        g = g.assign(full_size=np.inf)  # every size here is below full
        v = SS.min_size(g)
        o = SS.min_size(old[old.axis == ax])
        mins.append({"tree": tree, "axis": ax, "min_reliable_disjoint_partition":
                     v if v is not None else "không đạt trong dữ liệu hiện có",
                     "min_reliable_overlapping_subset": o if o is not None else "không đạt trong dữ liệu hiện có",
                     "sizes_tested_disjoint": "%d..%d" % (g["size"].min(), g["size"].max())})
    pd.DataFrame(mins).to_csv(rep / ("min_size_%s.csv" % tree), index=False)

    cols = ["tau_median", "p_top1", "w_median", "reliable"]
    sb = df[["axis", "size"] + cols].merge(old[["axis", "size"] + cols], on=["axis", "size"], how="left",
                                           suffixes=("_disjoint", "_overlapping"))
    sb["overlap_passed_disjoint_failed"] = sb.reliable_overlapping.fillna(False).astype(bool) & ~sb.reliable_disjoint
    sb.insert(0, "tree", tree)
    sb.to_csv(rep / ("side_by_side_%s.csv" % tree), index=False)
    gsb = gd[["k_seeds", "m_archs"] + cols].merge(oldg[["k_seeds", "m_archs"] + cols], on=["k_seeds", "m_archs"],
                                                    how="left", suffixes=("_disjoint", "_overlapping"))
    gsb["overlap_passed_disjoint_failed"] = gsb.reliable_overlapping.astype(bool) & ~gsb.reliable_disjoint
    gsb.insert(0, "tree", tree)
    gsb.to_csv(rep / ("side_by_side_grid_%s.csv" % tree), index=False)
    plot(df, old, gd, tree)
    print(sb.round(3).to_string(index=False))
    print(gsb.round(2).to_string(index=False))
    print(pd.DataFrame(mins).to_string(index=False))


def plot(df, old, gd, tree):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"seed": "seeds (A vs other seeds)", "arch": "architectures (A vs other archs)",
              "id_patients": "ID patients (A vs other ID patients)", "ood_patients": "OOD patients (A vs other OOD)"}
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, (name, t) in zip(axs, titles.items()):
        g, o = df[df.axis == name].sort_values("size"), old[old.axis == name].sort_values("size")
        ax.fill_between(g["size"], g.tau_q25, g.tau_q75, alpha=0.2, color="C0")
        ax.plot(g["size"], g.tau_median, "o-", color="C0", label="median tau, disjoint")
        ax.plot(g["size"], g.p_top1, "s-", color="C1", label="P(top-1), disjoint")
        ax.plot(o["size"], o.tau_median, "o:", color="C0", alpha=0.5, label="median tau, overlapping")
        ax.plot(o["size"], o.p_top1, "s:", color="C1", alpha=0.5, label="P(top-1), overlapping")
        ax.axhline(0.8, color="k", lw=0.8, ls=":")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("subset size")
    axs[0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Ranking stability, disjoint partitions vs overlapping subsets (%s)" % tree, fontsize=10)
    fig.tight_layout()
    fig.savefig(rep_dir() / ("stability_axes_%s.png" % tree), dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
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
        ax.set_xlabel("architectures m (vs the other 8-m)")
        ax.set_ylabel("seeds k (vs the other 5-k)")
        ax.set_title(t, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Disjoint seed x architecture blocks (%s); bold = >= 0.8" % tree, fontsize=10)
    fig.tight_layout()
    fig.savefig(rep_dir() / ("stability_grid_%s.png" % tree), dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=SS.PRIMARY, choices=list(SS.TREES))
    run(ap.parse_args().tree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
