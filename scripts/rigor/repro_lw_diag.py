#!/usr/bin/env python3
"""REPRO mismatch diagnosis: is the Maha 2-fold drift caused by float32 Ledoit-Wolf under multithreaded BLAS?

Camelyon ResNet50 seed 42, slide-disjoint fold 0 exactly as leakfree_knn.py. LedoitWolf is fitted repeatedly
under (dtype, BLAS threads) settings; each fit is compared with the float64 single-thread reference
(max |precision diff|, max |d2 diff| on the scored patches, Maha AUROC disjoint / same-slide).
Only REPRO-baseline quantities are computed; no campaign cell is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_info, threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from src.utils.benchmark_metrics import calc_auroc  # noqa: E402

ARCH, SEED, FOLD = "resnet50", 42, 0
RUNS = [("float64", 1, "ref"), ("float32", None, "a"), ("float32", None, "b"), ("float32", 16, "a"),
        ("float32", 16, "b"), ("float32", 1, "a"), ("float64", None, "a"), ("float64", None, "b"),
        ("float64", 16, "a"), ("float64", 16, "b")]


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def d2(x, mu, prec):
    diff = np.asarray(x, np.float64) - np.asarray(mu, np.float64)
    return np.sum((diff @ np.asarray(prec, np.float64)) * diff, axis=1)


def main() -> None:
    out = Path("outputs/rigor_pack/foundation_gate/repro_diag")
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps([{k: i.get(k) for k in ("internal_api", "version", "num_threads", "threading_layer")}
                      for i in threadpool_info()]), flush=True)
    meta = C.load_camelyon_metadata(C.REPO)
    d = torch.load(C.feature_path(C.REPO, "camelyon17", ARCH, SEED, indexed=True), map_location="cpu",
                   weights_only=False)
    g = lambda k: np.asarray(d[k].numpy() if hasattr(d[k], "numpy") else d[k])  # noqa: E731
    if "train_slide" in d:
        tsl, vsl = g("train_slide"), g("val_slide")
    else:
        tsl, vsl = (meta.slide.to_numpy()[g(k)] for k in ("train_idx", "val_idx"))
    thosp = meta.center.to_numpy()[g("train_idx")]
    rng = np.random.default_rng(1000 + SEED)
    fold_of = {}
    for h in np.unique(thosp):
        sl = np.unique(tsl[thosp == h])
        rng.shuffle(sl)
        for j, x in enumerate(sl):
            fold_of[x] = j % 2
    tf = np.array([fold_of[x] for x in tsl])
    vf = np.array([fold_of.get(x, -1) for x in vsl])
    tr = l2n(g("train_feats")[tf == FOLD])
    print("train dtype", tr.dtype, "shape", tr.shape, flush=True)
    fi, fs, fo = (l2n(g("val_feats")[vf == 1 - FOLD]), l2n(g("val_feats")[vf == FOLD]), l2n(g("ood_feats")))
    q = np.concatenate([fi, fs, fo])

    ref, rows = None, []
    for dt, th, tag in RUNS:
        x = tr.astype(dt)
        with threadpool_limits(limits=th, user_api="blas"):
            lw = LedoitWolf().fit(x)
        s = d2(q, lw.location_, lw.precision_)
        r = {"dtype": dt, "threads": th if th else "default", "run": tag, "cov_dtype": str(lw.covariance_.dtype),
             "shrinkage": float(lw.shrinkage_), "cond": float(np.linalg.cond(np.asarray(lw.covariance_, np.float64))),
             "auroc_disjoint": calc_auroc(-s[:len(fi)], -s[len(fi) + len(fs):]),
             "auroc_same": calc_auroc(-s[len(fi):len(fi) + len(fs)], -s[len(fi) + len(fs):])}
        if ref is None:
            ref = (lw.precision_.astype(np.float64), s, r)
        r["max_abs_prec_diff_vs_ref"] = float(np.abs(lw.precision_.astype(np.float64) - ref[0]).max())
        r["max_rel_d2_diff_vs_ref"] = float(np.max(np.abs(s - ref[1]) / np.abs(ref[1])))
        r["auroc_disjoint_diff_vs_ref"] = r["auroc_disjoint"] - ref[2]["auroc_disjoint"]
        rows.append(r)
        print(json.dumps(r), flush=True)
    (out / "lw_diag.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
