#!/usr/bin/env python3
"""Diagnostic for the R3 item 1 stop-check: re-run the Track A A-fit block of medbench_scores.py (current code, same
npz) on given cells and report Delta_fit next to the stored JSON value, with the query logits taken either from the
stored network logits (Track A) or from features @ W.T + b (what the package ViM uses).
Usage: tracka_recompute_afit.py <ds>/<tag> [...]   e.g. kermany/convnext_tiny_s42_std
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/rigor")]
import medbench_scores as M  # noqa: E402

FIT = ("Mahalanobis", "kNN", "ViM")


def run(ds, tag):
    fm_cell = tag.startswith("fm_")
    root = REPO / "outputs/rigor_pack" / ("foundation_gate" if fm_cell else "medbench")
    rep = REPO / "outputs/reports/rigor_pack" / ("foundation_gate" if fm_cell else "medbench")
    z = np.load(root / ds / f"{tag}.npz")
    f64 = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
    w, b = f64("fc_weight"), f64("fc_bias")
    sets = {k[:-6] for k in z.files if k.endswith("_feats")}
    seen = ["test_seen"] if "test_seen" in sets else ["seen"]
    ood = sorted(k for k in sets if k.startswith("ood"))[0]
    gt = M.groups_of(ds, z["train_keys"])
    fm = M.fold_map(ds, gt, z["train_labels"])
    tf = np.array([fm[g] for g in gt])
    gs = M.groups_of(ds, np.concatenate([z[f"{n}_keys"] for n in seen]))
    fs = np.array([fm.get(g, -1) for g in gs])
    out = {"cell": f"{ds}_{tag}", "ood_key": ood,
           "stored": json.loads((rep / ds / f"scores_{tag}.json").read_text())["afit"]["delta_fit"]}
    for mode in ("stored_logits", "head_logits"):
        lg = (lambda n: f64(f"{n}_logits")) if mode == "stored_logits" else (lambda n: f64(f"{n}_feats") @ w.T + b)
        same, dis = {m: [] for m in FIT}, {m: [] for m in FIT}
        for f in (0, 1):
            sel = tf == f
            sc = M.scorer(f64("train_feats")[sel], lg("train")[sel], w, b)
            si = {m: np.concatenate([M.all_scores(sc, lg(n), f64(f"{n}_feats"))[m] for n in seen]) for m in FIT}
            so = M.all_scores(sc, lg(ood), f64(f"{ood}_feats"))
            for m in FIT:
                same[m].append(M.auroc(si[m][fs == f], so[m]))
                dis[m].append(M.auroc(si[m][fs == 1 - f], so[m]))
        out[mode] = {m: float(np.mean(same[m]) - np.mean(dis[m])) for m in FIT}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run(*c.split("/", 1))
