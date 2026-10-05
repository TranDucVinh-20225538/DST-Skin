#!/usr/bin/env python3
"""Slide-disjoint 2-fold for the other train-fitted scores, ViM and ReAct
(decisions/precommit_isbi_patch_2026-10-05.md, section A; extends H11c of leakfree_knn.py).

Same folds as leakfree_knn.py (train slides per hospital, default_rng(1000 + seed), alternating).
Per fold f the published scorer config (OODScorer use_vim, use_react p90, vim_dim=None) is fitted
on fold-f train patches; ID = id_val patches of fold 1-f slides (slide-disjoint) or of fold f
slides (same-slides contrast); OOD = all hospital 2. AUROCs averaged over the 2 folds.

Writes leakage/leakfree_vim_react.csv/.txt (one row per arch x seed):
  n_train_fold0/1, n_id_fold0/1          patches per fold (REPRO: 97099 / 205337 train)
  auroc_{vim,react}_published            as-run AUROC (full train fit, all id_val)
  auroc_{vim,react}_2fold_slide_disjoint, auroc_{vim,react}_2fold_same_slides, delta_{vim,react}_2fold
  vim_dim                                ViM principal-subspace dim of fold 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

REPRO_FOLD_SIZES = (97099, 205337)


def folds(tsl, thosp, vsl, seed):
    rng = np.random.default_rng(1000 + seed)
    fold_of = {}
    for h in np.unique(thosp):
        sl = np.unique(tsl[thosp == h])
        rng.shuffle(sl)
        for j, x in enumerate(sl):
            fold_of[x] = j % 2
    return np.array([fold_of[x] for x in tsl]), np.array([fold_of.get(x, -1) for x in vsl])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--archs", nargs="+", default=list(C.ARCHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--out-name", default="leakfree_vim_react")
    args = ap.parse_args()
    root = Path(args.root)
    rep = C.ensure_dir(C.default_reports(root) / "leakage")
    import torch
    from src.utils.benchmark_metrics import calc_auroc
    from src.utils.ood_vim_react import react_energy_score, vim_score
    from src.utils.scoring import OODScorer

    meta = C.load_camelyon_metadata(root)
    rows = []
    for s in args.seeds:
        for a in args.archs:
            t0 = time.time()
            p = C.feature_path(root, "camelyon17", a, s, indexed=True)
            d = torch.load(p, map_location="cpu", weights_only=False)
            g = lambda k: np.asarray(d[k].numpy() if hasattr(d[k], "numpy") else d[k])  # noqa: E731
            tsl, vsl = g("train_slide"), g("val_slide")
            thosp = meta.center.to_numpy()[g("train_idx")]
            tf, vf = folds(tsl, thosp, vsl, s)
            sizes = (int((tf == 0).sum()), int((tf == 1).sum()))
            if s == 42 and sizes != REPRO_FOLD_SIZES:
                raise SystemExit("STOP: fold sizes %s != H11c %s for %s" % (sizes, REPRO_FOLD_SIZES, a))
            pub = C.metric_vector(root, "camelyon17", a, s)
            rec = {"arch": a, "seed": s, "n_train_fold0": sizes[0], "n_train_fold1": sizes[1],
                   "n_id_fold0": int((vf == 0).sum()), "n_id_fold1": int((vf == 1).sum()),
                   "auroc_vim_published": float(pub[C.METHODS_ORDER.index("ViM")]),
                   "auroc_react_published": float(pub[C.METHODS_ORDER.index("ReAct")])}
            res = {k: [] for k in ("vim_dis", "vim_same", "react_dis", "react_same")}
            oo = g("ood_feats")
            for f in (0, 1):
                m = tf == f
                sc = OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None)
                sc.fit(g("train_feats")[m], train_logits=g("train_logits")[m],
                       fc_weight=g("fc_weight"), fc_bias=g("fc_bias"))
                if f == 0:
                    rec["vim_dim"] = int(sc.vim_params["d"])
                vim = lambda x: -vim_score(x, sc.vim_params)  # noqa: E731
                react = lambda x: react_energy_score(x, sc.react_params, T=1.0)  # noqa: E731
                vo, ro = vim(oo), react(oo)
                for tag, sel in (("dis", vf == 1 - f), ("same", vf == f)):
                    x = g("val_feats")[sel]
                    res["vim_" + tag].append(calc_auroc(vim(x), vo))
                    res["react_" + tag].append(calc_auroc(react(x), ro))
                del sc
            for sc_name in ("vim", "react"):
                rec["auroc_%s_2fold_slide_disjoint" % sc_name] = float(np.mean(res[sc_name + "_dis"]))
                rec["auroc_%s_2fold_same_slides" % sc_name] = float(np.mean(res[sc_name + "_same"]))
                rec["delta_%s_2fold" % sc_name] = (rec["auroc_%s_2fold_slide_disjoint" % sc_name]
                                                   - rec["auroc_%s_2fold_same_slides" % sc_name])
            rec["seconds"] = round(time.time() - t0, 1)
            rows.append(rec)
            print(rec, flush=True)
            del d
    df = pd.DataFrame(rows)
    df.to_csv(rep / ("%s.csv" % args.out_name), index=False)
    C.write_text(rep / ("%s.txt" % args.out_name),
                 ["ViM / ReAct slide-disjoint 2-fold (precommit_isbi_patch_2026-10-05.md, A)",
                  df.round(4).to_string(index=False)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
