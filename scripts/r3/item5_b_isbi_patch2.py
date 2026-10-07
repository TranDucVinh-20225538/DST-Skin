#!/usr/bin/env python3
"""R3 item 5(b), post-hoc: gap closed by within-S cross-fit on the isbi_patch2 slide-disjoint retrains.

Model (arch, seed, fold f) was trained on the fold-f slides S (logit_retrain_slide_disjoint_v2.balanced_folds).
Cached: outputs/rigor_pack/logit_retrain_slide_disjoint_v2/{arch}_s{s}_f{f}.npz; patch order = tr_idx[tf == f] /
iv_idx (fold_assignment), so slide IDs come from metadata (no re-extraction).
- truth = scorer fitted on all S train patches, ID = id_val of slides not in S  (isbi_patch2_scores 'unseen')
- leaky = same scorer, ID = id_val of S                                          (isbi_patch2_scores 'seen')
- F2 = Track E cross-fit within S: S train slides split by leakfree_fit_scores.folds (H11c rule, model seed);
  id_val of S half g scored by the scorer fitted on half 1-g, OOD scores averaged over the two fits.
Scorer = mech_cpu.fit (OODScorer, Ledoit-Wolf float64); AUROC = calc_auroc.
Writes results/r3/5/cells/b_{arch}_s{seed}_f{fold}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
sys.path.insert(0, str(REPO))
import common as C  # noqa: E402
from leakfree_fit_scores import folds  # noqa: E402
from logit_retrain_slide_disjoint_v2 import fold_assignment  # noqa: E402
from mech_cpu import auroc, fit, scores  # noqa: E402

FIT = ("Mahalanobis", "kNN", "ViM", "ReAct")
OUT = REPO / "results/r3/5/cells"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()
    a, s, f = args.arch, args.seed, args.fold
    dst = OUT / ("b_%s_s%d_f%d.json" % (a, s, f))
    if dst.exists():
        return 0
    t0 = time.time()
    meta = C.load_camelyon_metadata(REPO)
    tr_idx, iv_idx, tf, vf = fold_assignment(meta)
    z = np.load(REPO / "outputs/rigor_pack/logit_retrain_slide_disjoint_v2" / ("%s_s%d_f%d.npz" % (a, s, f)))
    if not np.array_equal(np.asarray(z["id_val_fold"]), vf) or len(z["train_feats"]) != int((tf == f).sum()):
        raise SystemExit("STOP: cached patch order does not match fold_assignment")
    prev = json.loads((C.default_reports(REPO) / "leakage/logit_retrain_slide_disjoint_v2"
                       / ("scores_%s_s%d_f%d.json" % (a, s, f))).read_text())
    g = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
    Xtr, Ltr, Xv, Lv, Xo, Lo = g("train_feats"), g("train_logits"), g("val_feats"), g("val_logits"), g("ood_feats"), g("ood_logits")
    W, b = g("fc_weight"), g("fc_bias")
    slide, center = meta.slide.to_numpy(), meta.center.to_numpy()
    tsl_S, thosp_S = slide[tr_idx][tf == f], center[tr_idx][tf == f]
    ht, hv = folds(tsl_S, thosp_S, slide[iv_idx], s)
    inS = vf == f
    if np.any(hv[inS] < 0) or np.any(hv[~inS] >= 0):
        raise SystemExit("STOP: id_val slides of S without a half")
    id_cf = np.full(len(vf), np.nan)
    ood_cf = {m: np.zeros(len(Xo)) for m in FIT}
    idc = {m: id_cf.copy() for m in FIT}
    for h in (0, 1):
        sc = fit(Xtr[ht == h], Ltr[ht == h], W, b)
        sv, so = scores(sc, Lv, Xv), scores(sc, Lo, Xo)
        del sc
        for m in FIT:
            idc[m][inS & (hv == 1 - h)] = sv[m][inS & (hv == 1 - h)]
            ood_cf[m] += so[m] / 2.0
        print("half %d %.0fs" % (h, time.time() - t0), flush=True)
    res = {"arch": a, "seed": s, "fold": f, "n_slides_S": int(len(np.unique(tsl_S))),
           "n_groups_fit": [int(len(np.unique(tsl_S[ht == h]))) for h in (0, 1)],
           "n_fit_patches": [int((ht == h).sum()) for h in (0, 1)], "n_id_S": int(inS.sum()),
           "n_id_notS": int((~inS).sum()), "n_ood": int(len(Xo)), "d": int(Xtr.shape[1])}
    for m in FIT:
        t, l = float(prev["auroc_%s_unseen" % m]), float(prev["auroc_%s_seen" % m])
        F2 = auroc(idc[m][inS], ood_cf[m])
        res[m] = {"truth": t, "leaky": l, "F2": F2, "gap": l - t,
                  "frac_closed": (l - F2) / (l - t) if l != t else None}
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
