#!/usr/bin/env python3
"""Mechanism check (post-hoc, descriptive; requested 2026-10-07 after the preliminary L4 / Track A read).

Camelyon only (same 30 training slides and same folds for every backbone, so the number of groups is constant):
does the scorer-fit inflation Δ_fit (seed 42, median of Mahalanobis / kNN / ViM, Track A) scale with how strongly
a backbone's features encode slide identity?

Per backbone, on the training patches: subsample N_TR + N_TE patches per slide (default_rng(0), identical patch
indices for every backbone), L2-normalise, then
  slide_probe_bacc : balanced accuracy of a linear probe (StandardScaler + LogisticRegression C=1, lbfgs)
                     predicting the slide, trained on N_TR and tested on the other N_TE patches of each slide;
  icc              : sum over dims of between-slide variance / sum of total variance (subsample, L2-normalised).
Then Spearman / Pearson of Δ_fit vs each, with a permutation p (10,000, default_rng(1)).

Stage `probe --model M` (one CPU job per backbone) -> outputs/rigor_pack/miccai_campaign/slide_identity/{M}.json
Stage `analyze` -> outputs/reports/rigor_pack/miccai_campaign/trackA_foundation/slide_identity.{csv,md}
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

CNN8 = ("resnet18", "resnet50", "densenet121", "convnext_tiny", "mobilenet_v3_large", "regnet_y_3_2gf", "effb3",
        "efficientnet_v2_s")
FMS = ("dinov2_vitb14", "uni", "conch_v1_5", "virchow2", "dinov2_vitl14")
N_TR, N_TE = 300, 100
CACHE = C.REPO / "outputs/rigor_pack/miccai_campaign/slide_identity"
OUT = C.REPO / "outputs/reports/rigor_pack/miccai_campaign/trackA_foundation"


def load(model: str):
    meta = C.load_camelyon_metadata(C.REPO)
    if model in FMS:
        z = np.load(C.REPO / "outputs/rigor_pack/foundation_gate/feats" / ("camelyon_%s.npz" % model))
        return np.asarray(z["train_feats"]), meta.slide.to_numpy()[np.asarray(z["train_idx"])]
    import torch
    d = torch.load(C.feature_path(C.REPO, "camelyon17", model, 42, indexed=True), map_location="cpu",
                   weights_only=False)
    g = lambda k: np.asarray(d[k].numpy() if hasattr(d[k], "numpy") else d[k])  # noqa: E731
    return g("train_feats"), meta.slide.to_numpy()[g("train_idx")]


def probe(model: str) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    X, sl = load(model)
    rng = np.random.default_rng(0)
    tr, te = [], []
    for s in np.unique(sl):
        idx = rng.permutation(np.flatnonzero(sl == s))
        n_te = min(N_TE, len(idx) // 4)
        te.append(idx[:n_te])
        tr.append(idx[n_te:n_te + N_TR])
    tr, te = np.concatenate(tr), np.concatenate(te)
    Xs = np.asarray(X[np.concatenate([tr, te])], dtype=np.float64)
    Xs /= np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8
    ys = sl[np.concatenate([tr, te])]
    Xtr, Xte, ytr, yte = Xs[:len(tr)], Xs[len(tr):], ys[:len(tr)], ys[len(tr):]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    bacc = float(balanced_accuracy_score(yte, clf.predict(sc.transform(Xte))))
    mu = Xs.mean(0)
    between = sum(((ys == s).sum() * (Xs[ys == s].mean(0) - mu) ** 2).sum() for s in np.unique(ys))
    total = ((Xs - mu) ** 2).sum()
    res = {"model": model, "n_slides": int(len(np.unique(sl))), "n_train": int(len(tr)), "n_test": int(len(te)),
           "feat_dim": int(X.shape[1]), "slide_probe_bacc": bacc, "icc": float(between / total),
           "chance": 1.0 / len(np.unique(sl)), "seconds": round(time.time() - t0, 1)}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / ("%s.json" % model)).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def perm_corr(x, y, kind, n=10000):
    from scipy.stats import pearsonr, spearmanr
    f = (lambda a, b: spearmanr(a, b).statistic) if kind == "spearman" else (lambda a, b: pearsonr(a, b).statistic)
    r = f(x, y)
    rng = np.random.default_rng(1)
    null = np.array([f(x, rng.permutation(y)) for _ in range(n)])
    return float(r), float((np.sum(np.abs(null) >= abs(r)) + 1) / (n + 1))


def analyze() -> None:
    A = pd.read_csv(C.REPO / "outputs/reports/rigor_pack/miccai_campaign/trackA_foundation/summary.csv")
    A = A[(A.dataset == "camelyon17") & (A.seed == 42)].set_index("backbone")
    rows = []
    for m in CNN8 + FMS:
        p = CACHE / ("%s.json" % m)
        if p.exists() and m in A.index:
            j = json.loads(p.read_text())
            rows.append({**j, "kind": "FM" if m in FMS else "CNN", "dfit_feature_median": A.loc[m, "median_feature_delta"],
                         **{"dfit_%s" % s: A.loc[m, "dfit_%s" % s] for s in ("Mahalanobis", "kNN", "ViM", "ReAct")}})
    D = pd.DataFrame(rows).sort_values("dfit_feature_median")
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "slide_identity.csv", index=False)
    lines = ["# Slide-identity mechanism check (Camelyon, seed 42; post-hoc, descriptive)", "",
             "Same 30 training slides and folds for every backbone. Probe: balanced accuracy of a linear slide-ID probe"
             " (%d / %d patches per slide, chance 1/30). ICC: between-slide / total variance of L2-normalised"
             " features." % (N_TR, N_TE), "",
             "```", D[["model", "kind", "feat_dim", "slide_probe_bacc", "icc", "dfit_feature_median", "dfit_Mahalanobis",
                       "dfit_kNN", "dfit_ViM", "dfit_ReAct"]].round(3).to_string(index=False), "```", ""]
    for x in ("slide_probe_bacc", "icc"):
        for sub, g in (("all", D), ("CNN", D[D.kind == "CNN"]), ("FM", D[D.kind == "FM"])):
            if len(g) >= 4:
                for kind in ("spearman", "pearson"):
                    r, p = perm_corr(g[x].to_numpy(), g.dfit_feature_median.to_numpy(), kind)
                    lines.append("- Δ_fit vs %s, %s (n = %d), %s r = %+.3f, permutation p = %.4f" % (x, sub, len(g), kind, r, p))
    (OUT / "slide_identity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("probe", "analyze"))
    ap.add_argument("--model")
    a = ap.parse_args()
    probe(a.model) if a.stage == "probe" else analyze()


if __name__ == "__main__":
    main()
