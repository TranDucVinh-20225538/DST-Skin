#!/usr/bin/env python3
"""P5 of decisions/precommit_isbi_patch2_2026-10-06.md: all 7 scores on each v2 retrained model.

Published scorer config fitted on the fold's train patches of the retrained model (features cast to
float64, so Ledoit-Wolf Mahalanobis is fitted and evaluated in float64); ID = unseen-slide id_val
(other fold, primary) or seen-slide id_val (same fold); OOD = all hospital 2. AUROC via calc_auroc
(pipeline convention).

Writes outputs/reports/rigor_pack/leakage/logit_retrain_slide_disjoint_v2/scores_{arch}_s{s}_f{f}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()
    from src.utils.benchmark_metrics import calc_auroc
    from src.utils.scoring import OODScorer

    a, s, f = args.arch, args.seed, args.fold
    src = C.default_out(C.REPO) / "logit_retrain_slide_disjoint_v2" / ("%s_s%d_f%d.npz" % (a, s, f))
    out = C.default_reports(C.REPO) / "leakage/logit_retrain_slide_disjoint_v2" / ("scores_%s_s%d_f%d.json" % (a, s, f))
    t0 = time.time()
    z = np.load(src)
    g = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
    vf = z["id_val_fold"]
    sc = OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None)
    sc.fit(g("train_feats"), train_logits=g("train_logits"), fc_weight=g("fc_weight"), fc_bias=g("fc_bias"))
    si = sc.get_all_scores(g("val_logits"), g("val_feats"))
    so = sc.get_all_scores(g("ood_logits"), g("ood_feats"))
    res = {"arch": a, "seed": s, "fold": f, "n_fit": int(len(z["train_feats"])), "vim_dim": int(sc.vim_params["d"])}
    for k, name in C.DISPLAY.items():
        res["auroc_%s_unseen" % name] = calc_auroc(si[k][vf == 1 - f], so[k])
        res["auroc_%s_seen" % name] = calc_auroc(si[k][vf == f], so[k])
    res["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
