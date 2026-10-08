#!/usr/bin/env python3
"""R3 item 5b Step B (results/r3/5b/PRECOMMIT.json): fold-count stability of the matched-pooled cross-fit AUROC.

One FM x scorer per call, K in {2, 5, 10}. Folds over the 30 training slides, stratified within hospital: per
hospital (sorted) the slides are sorted, shuffled with default_rng([20261008, K]) (one generator per K, used across
hospitals in order) and dealt round-robin, the counter continuing across hospitals.
X_K = crossfit_auroc(protocol="default", fold_ids = fold of each id_val patch's slide), point estimate
(uncertainty="bootstrap", n_bootstrap=0: no CI). --jackknife: also the jackknife CI (blocks = slides).
Input: R3 item-1 inputs $R3_INPUTS/camelyon_{fm}_s42.npz (groups = slide ids).
Writes results/r3/5b/stepB/{fm}_{scorer}[_jk].json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
import common as C  # noqa: E402

KS = (2, 5, 10)
OUT = REPO / "results/r3/5b/stepB"


def slide_folds(slides, hosp, K):
    rng = np.random.default_rng([20261008, K])
    fold, c = {}, 0
    for h in sorted(set(hosp.values())):
        sl = sorted(s for s in slides if hosp[s] == h)
        for s in [sl[i] for i in rng.permutation(len(sl))]:
            fold[s] = c % K
            c += 1
    return fold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True)
    ap.add_argument("--scorer", required=True, choices=["mahalanobis_l2", "knn_mean_cosine"])
    ap.add_argument("--jackknife", action="store_true")
    ap.add_argument("--ks", default="2,5,10")
    ap.add_argument("--gpu-knn", action="store_true", help="knn_mean_cosine via the validated GPU scorer")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from crossfit_ood import crossfit_auroc
    if a.gpu_knn:
        sys.path.insert(0, str(REPO / "scripts/r3"))
        from crossfit_ood import scorers as S
        from recompute_gpu_knn import GPUKNNMeanCosineScorer
        S._REGISTRY["knn_mean_cosine"] = GPUKNNMeanCosineScorer

    z = np.load(Path(os.environ["R3_INPUTS"]) / ("camelyon_%s_s42.npz" % a.fm))
    xtr, gtr = z["features_train"], z["groups_train"].astype(str)
    xid, gid, xood = z["features_id_eval"], z["groups_id_eval"].astype(str), z["features_ood"]
    meta = C.load_camelyon_metadata(REPO)
    sm = meta[["slide", "center"]].drop_duplicates()
    hosp = {str(s): int(h) for s, h in zip(sm.slide, sm.center)}
    slides = sorted(set(gtr))
    assert len(slides) == 30, len(slides)
    no_id = sorted(set(slides) - set(gid))
    cj = REPO / "outputs/reports/rigor_pack/foundation_gate/cells" / ("camelyon_%s.json" % a.fm)
    if cj.exists():
        id_acc = json.loads(cj.read_text())["probe"]["id_acc"]
    else:
        import pandas as pd
        c = pd.read_csv(Path(os.environ["R3_INPUTS"]).parent / "cells_camelyon.csv").set_index("cell")
        id_acc = float(c.loc["camelyon_%s_s42" % a.fm, "id_acc"])

    OUT.mkdir(parents=True, exist_ok=True)
    tag = ("_jk" if a.jackknife else "") + a.tag
    outp = OUT / ("%s_%s%s.json" % (a.fm, a.scorer, tag))
    res = json.loads(outp.read_text()) if outp.exists() else {}
    res.update({"fm": a.fm, "scorer": a.scorer, "d": int(xtr.shape[1]), "n_train": int(len(xtr)),
                "n_id_eval": int(len(xid)), "n_ood": int(len(xood)), "n_slides": 30,
                "train_slides_without_id_val": no_id, "probe_id_acc": id_acc, "gpu_knn": a.gpu_knn,
                "hospitals": sorted(set(hosp[s] for s in slides))})
    for K in [int(k) for k in a.ks.split(",")]:
        fold = slide_folds(slides, hosp, K)
        fids = np.array([fold[g] for g in gid])
        per_fold = []
        for k in range(K):
            out_sl = {s for s, f in fold.items() if f == k}
            per_fold.append({"fold": k, "slides_out": sorted(out_sl),
                             "hospitals_out": sorted(hosp[s] for s in out_sl),
                             "n_fit_patches": int((~np.isin(gtr, sorted(out_sl))).sum()),
                             "n_groups_fit": 30 - len(out_sl), "n_id_eval": int((fids == k).sum())})
        t0 = time.time()
        kw = dict(uncertainty="jackknife", n_jackknife_blocks=50) if a.jackknife else \
            dict(uncertainty="bootstrap", n_bootstrap=0)
        rep = crossfit_auroc(xtr, gtr, xid, gid, xood, scorers=[a.scorer], protocol="default", n_splits=K,
                             fold_ids=fids, random_state=0, **kw)
        r = rep.results[a.scorer]
        nf = [p["n_fit_patches"] for p in per_fold]
        res["K%d" % K] = {"X_K": r.auroc_crossfit, "auroc_leaky": r.auroc_leaky, "delta": r.delta,
                          "X_K_ci": list(r.auroc_crossfit_ci) if a.jackknife else None,
                          "delta_ci": list(r.delta_ci) if a.jackknife else None,
                          "fit_patches_min_mean_max": [min(nf), float(np.mean(nf)), max(nf)],
                          "n_groups_fit_min_max": [min(p["n_groups_fit"] for p in per_fold),
                                                   max(p["n_groups_fit"] for p in per_fold)],
                          "folds": per_fold, "seconds": round(time.time() - t0, 1)}
        outp.write_text(json.dumps(res, indent=1) + "\n")
        print(a.fm, a.scorer, K, res["K%d" % K]["X_K"], res["K%d" % K]["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
