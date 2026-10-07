#!/usr/bin/env python3
"""R3 item 3: mechanism follow-up on Camelyon (post-hoc; 13 backbones, 30 training slides, seed 42).

Stage `whiten --model M`: on each H11c fold-f fit set (leakfree_fit_scores.folds, seed 42; the set the Track A
Mahalanobis scorer is fitted on), the exact Ledoit-Wolf fit of that scorer (OODScorer: LedoitWolf() on
L2-normalised float64 features) -> shrinkage and precision P; with T = total covariance and B = between-slide
covariance of the same fit set:
    icc_raw_fold = tr(B) / tr(T),   icc_whitened_fold = tr(P B) / tr(P T).
Writes outputs/rigor_pack/r3/item3/{M}.json (per fold + fold mean).

Stage `analyze`: joins slide_identity.csv (probe balanced accuracy, subsample ICC, d, Track A seed-42 Delta_fit
per scorer) with the whitened ICC; tests (all listed in the report):
  (a) Spearman rho(Delta_s, probe) and rho(Delta_s, ICC), s in {Mahalanobis, kNN, ViM}
  (b) rho(Delta_s, whitened ICC)
  (c) for every rho in (a), (b), (d): permutation p (10,000, default_rng(1)), bootstrap 95% CI over backbones
      (10,000, default_rng(2), percentile; resamples with < 3 distinct backbones or constant ranks dropped),
      partial Spearman controlling for CNN/FM (rank-residualise on the indicator), within-group rho (CNN n = 8,
      FM n = 5); scatter plots coloured CNN / FM
  (d) rho(Delta_s, d)
Writes results/r3/3/{mechanism.csv, tests.csv, scatter.png, REPORT.md}.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
sys.path.insert(0, str(REPO))
CACHE = REPO / "outputs/rigor_pack/r3/item3"
OUT = REPO / "results/r3/3"
SI = REPO / "outputs/reports/rigor_pack/miccai_campaign/trackA_foundation/slide_identity.csv"
FMS = ("dinov2_vitb14", "uni", "conch_v1_5", "virchow2", "dinov2_vitl14")
SCORERS = ("Mahalanobis", "kNN", "ViM")


def load(model):
    import common as C
    meta = C.load_camelyon_metadata(REPO)
    if model in FMS:
        z = np.load(REPO / "outputs/rigor_pack/foundation_gate/feats" / ("camelyon_%s.npz" % model))
        g = lambda k: np.asarray(z[k])  # noqa: E731
    else:
        import torch
        d = torch.load(C.feature_path(REPO, "camelyon17", model, 42, indexed=True), map_location="cpu",
                       weights_only=False)
        g = lambda k: np.asarray(d[k].numpy() if hasattr(d[k], "numpy") else d[k])  # noqa: E731
    vk = "id_slide" if model in FMS else "val_slide"
    return g("train_feats"), g("train_slide"), meta.center.to_numpy()[g("train_idx")], g(vk)


def whiten(model):
    from sklearn.covariance import LedoitWolf
    from leakfree_fit_scores import REPRO_FOLD_SIZES, folds
    from src.utils.scoring import OODScorer
    X, tsl, thosp, vsl = load(model)
    tf, _ = folds(tsl, thosp, vsl, 42)
    if ((tf == 0).sum(), (tf == 1).sum()) != REPRO_FOLD_SIZES:
        raise SystemExit("STOP: H11c fold sizes differ")
    res = {"model": model, "d": int(X.shape[1]), "folds": []}
    for f in (0, 1):
        sel = tf == f
        Z = OODScorer.l2_normalize(np.asarray(X[sel], dtype=np.float64))
        lw = LedoitWolf().fit(Z)
        P = lw.precision_
        s = tsl[sel]
        mu = Z.mean(0)
        Zc = Z - mu
        T = Zc.T @ Zc / len(Z)
        M = np.stack([Z[s == u].mean(0) - mu for u in np.unique(s)])
        w = np.array([(s == u).sum() for u in np.unique(s)], dtype=np.float64) / len(Z)
        B = (M * w[:, None]).T @ M
        res["folds"].append({"fold": f, "n_fit": int(sel.sum()), "n_slides": int(len(np.unique(s))),
                             "shrinkage": float(lw.shrinkage_), "icc_raw": float(np.trace(B) / np.trace(T)),
                             "icc_whitened": float(np.sum(P * B) / np.sum(P * T))})
    for k in ("shrinkage", "icc_raw", "icc_whitened"):
        res[k + "_mean"] = float(np.mean([r[k] for r in res["folds"]]))
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / ("%s.json" % model)).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def rank(a):
    from scipy.stats import rankdata
    return rankdata(a)


def partial_spearman(x, y, g):
    """Spearman of x, y controlling for the binary group g: residualise ranks on the indicator."""
    rx, ry = rank(x), rank(y)
    res = lambda r: r - np.array([r[g == v].mean() for v in g])  # noqa: E731
    ex, ey = res(rx), res(ry)
    return float(np.corrcoef(ex, ey)[0, 1])


def tests_for(D, x, y, label):
    xv, yv, gv = D[x].to_numpy(float), D[y].to_numpy(float), (D.kind == "FM").to_numpy()
    r = spearman(xv, yv)
    rng = np.random.default_rng(1)
    null = np.array([spearman(xv, rng.permutation(yv)) for _ in range(10000)])
    p = float((np.sum(np.abs(null) >= abs(r)) + 1) / 10001)
    brng = np.random.default_rng(2)
    boots = []
    for _ in range(10000):
        i = brng.integers(0, len(D), len(D))
        if len(np.unique(i)) < 3 or np.ptp(xv[i]) == 0 or np.ptp(yv[i]) == 0:
            continue
        boots.append(spearman(xv[i], yv[i]))
    boots = np.array(boots)
    out = {"test": label, "x": x, "y": y, "n": len(D), "rho": r, "perm_p": p,
           "boot_lo": float(np.percentile(boots, 2.5)), "boot_hi": float(np.percentile(boots, 97.5)),
           "n_boot_used": int(len(boots)), "rho_partial_kind": partial_spearman(xv, yv, gv)}
    for k, m in (("CNN", ~gv), ("FM", gv)):
        out["rho_within_%s" % k] = spearman(xv[m], yv[m])
        out["n_%s" % k] = int(m.sum())
    return out


def analyze():
    S = pd.read_csv(SI)
    W = pd.DataFrame([json.loads((CACHE / ("%s.json" % m)).read_text()) for m in S.model])
    D = S.merge(W[["model", "shrinkage_mean", "icc_raw_mean", "icc_whitened_mean"]], on="model")
    D["shrinkage_fold0"] = [json.loads((CACHE / ("%s.json" % m)).read_text())["folds"][0]["shrinkage"] for m in D.model]
    D["shrinkage_fold1"] = [json.loads((CACHE / ("%s.json" % m)).read_text())["folds"][1]["shrinkage"] for m in D.model]
    D = D.sort_values(["kind", "model"]).reset_index(drop=True)
    T = []
    for s in SCORERS:
        y = "dfit_%s" % s
        T.append(tests_for(D, "slide_probe_bacc", y, "a"))
        T.append(tests_for(D, "icc", y, "a"))
        T.append(tests_for(D, "icc_whitened_mean", y, "b"))
        T.append(tests_for(D, "feat_dim", y, "d"))
    T = pd.DataFrame(T)
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["model", "kind", "feat_dim", "slide_probe_bacc", "icc", "icc_raw_mean", "icc_whitened_mean",
            "shrinkage_fold0", "shrinkage_fold1", "dfit_Mahalanobis", "dfit_kNN", "dfit_ViM", "dfit_feature_median"]
    D[cols].to_csv(OUT / "mechanism.csv", index=False)
    T.to_csv(OUT / "tests.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = (("slide_probe_bacc", "slide-ID probe balanced acc."), ("icc", "ICC (raw)"),
          ("icc_whitened_mean", "ICC (LW-whitened)"), ("feat_dim", "feature dim d"))
    fig, ax = plt.subplots(len(SCORERS), len(xs), figsize=(4 * len(xs), 3.4 * len(SCORERS)))
    for i, s in enumerate(SCORERS):
        for j, (x, xl) in enumerate(xs):
            a = ax[i, j]
            for k, c in (("CNN", "tab:blue"), ("FM", "tab:orange")):
                g = D[D.kind == k]
                a.scatter(g[x], g["dfit_%s" % s], c=c, label=k, s=28)
                for _, r in g.iterrows():
                    a.annotate(r.model, (r[x], r["dfit_%s" % s]), fontsize=6, alpha=0.7)
            t = T[(T.x == x) & (T.y == "dfit_%s" % s)].iloc[0]
            a.set_title("rho=%+.2f [%+.2f, %+.2f]" % (t.rho, t.boot_lo, t.boot_hi), fontsize=9)
            a.set_xlabel(xl)
            a.set_ylabel("Delta_fit %s" % s)
            if i == 0 and j == 0:
                a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "scatter.png", dpi=130)

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    fmt = lambda v: "%+.3f" % v  # noqa: E731
    lines = ["# R3 item 3: mechanism follow-up (Camelyon, 13 backbones, 30 slides, seed 42)", "",
             "commit: %s" % commit, "",
             "Delta_fit = Track A seed-42 same - disjoint AUROC per scorer (n_groups_fit = 15 slides per fold, K = 2)."
             " ID accuracy is not defined per backbone for the probe-free feature scorers; the slide-ID probe"
             " accuracy is the model-side column.", "",
             "## Per backbone", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in D[cols].iterrows():
        lines.append("| " + " | ".join(str(r[c]) if isinstance(r[c], str) else
                                      ("%d" % r[c] if c == "feat_dim" else "%.3f" % r[c]) for c in cols) + " |")
    lines += ["", "## Tests (every test run)", "",
              "| # | part | y | x | n | rho | perm p | boot 95% CI | partial rho (CNN/FM) | rho CNN (n=8) | rho FM (n=5) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, t in T.iterrows():
        lines.append("| %d | %s | %s | %s | %d | %s | %.4f | [%s, %s] | %s | %s | %s |" % (
            i + 1, t.test, t.y, t.x, t.n, fmt(t.rho), t.perm_p, fmt(t.boot_lo), fmt(t.boot_hi),
            fmt(t.rho_partial_kind), fmt(t.rho_within_CNN), fmt(t.rho_within_FM)))
    strongest = T.loc[T.perm_p.idxmin()]
    lines += ["", "![scatter](scatter.png)", "",
              "Verdict: smallest permutation p over %d tests = %.4f (%s vs %s, rho %s); no multiplicity correction"
              " applied." % (len(T), strongest.perm_p, strongest.y, strongest.x, fmt(strongest.rho)), "",
              "Caveats: n = 13 backbones (FM n = 5); post-hoc, not preregistered; all %d tests listed above." % len(T)]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("whiten", "analyze"))
    ap.add_argument("--model")
    a = ap.parse_args()
    whiten(a.model) if a.stage == "whiten" else analyze()


if __name__ == "__main__":
    main()
