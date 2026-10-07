#!/usr/bin/env python3
"""R3 item 2: crossfit-ood v3 coverage_study_paper2fold.py on designs read from the REAL splits.

The design (per-group train sizes, ID-eval sizes per training group, OOD units) is read from the item-1 input of
one feature set per dataset (the split is shared by every backbone of a dataset): camelyon_resnet50_s42,
breakhis_resnet50_s42_std_r0, dermamnist_resnet50_s42_std, isic2019_resnet50_s42_std, kermany_resnet50_s42_std.
OOD images are the resampling units (as in item 1: no OOD groups), so ood_sizes = 1 per image. v3 defaults kept:
SCALE camelyon 1/30 (all sizes), kermany 1/2 train and 1/8 OOD (ID-eval unscaled), others 1; N_FEATURES = 128;
every other setting is the unchanged v3 script (pilot -> calibrate -> truth -> cover, 100 datasets per cell).

    python item2_coverage_real.py <phase> --out DIR [--n-features D] [--designs ...] [v3 options]
    python item2_coverage_real.py summary            # print the real designs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

CF = Path.home() / "handoff/crossfit-ood_v3"
W = Path.home() / "r3work/item1/inputs"
sys.path.insert(0, str(CF / "scripts"))
SOURCE = {"camelyon": "camelyon_resnet50_s42", "breakhis": "breakhis_resnet50_s42_std_r0",
          "dermamnist": "dermamnist_resnet50_s42_std", "isic2019": "isic2019_resnet50_s42_std",
          "kermany": "kermany_resnet50_s42_std"}
SCALE = {"camelyon": (1 / 30, 1 / 30, 1 / 30), "kermany": (1 / 2, 1.0, 1 / 8)}  # train, eval, ood


def _scale(a, s):
    return np.maximum(1, np.round(np.asarray(a, float) * s)).astype(int)


def real_design(name, scale=None):
    z = np.load(W / (SOURCE[name] + ".npz"), allow_pickle=True)
    gtr, gid = z["groups_train"].astype(str), z["groups_id_eval"].astype(str)
    u, tr = np.unique(gtr, return_counts=True)
    ev = np.array([(gid == g).sum() for g in u])
    n_ood = len(z["features_ood"])
    st, se, so = SCALE.get(name, (1.0, 1.0, 1.0))
    trs = _scale(tr, st) if st != 1 else tr
    evs = np.where(ev > 0, _scale(ev, se), 0) if se != 1 else ev
    oo = np.ones(max(1, int(round(n_ood * so))), int)
    chk = dict(real_train=int(tr.sum()), real_groups=int(len(u)), real_eval=int(ev.sum()),
               real_eval_groups=int((ev > 0).sum()), real_eval_not_in_train=int((~np.isin(gid, u)).sum()),
               real_ood=int(n_ood), source=SOURCE[name])
    info = dict(train_sizes=trs, eval_sizes=evs, ood_sizes=oo, scale=st, checks=chk,
                sources=["item-1 input %s (real split)" % SOURCE[name]])
    info["summary"] = dict(n_train_groups=int(len(trs)), n_train=int(trs.sum()),
                           n_groups_with_eval=int((evs > 0).sum()), n_eval=int(evs.sum()),
                           n_ood_groups=int(len(oo)), n_ood=int(oo.sum()),
                           train_size_min_med_max=[int(trs.min()), float(np.median(trs)), int(trs.max())],
                           scale=[st, se, so])
    return info


def main():
    if sys.argv[1:2] == ["summary"]:
        import json
        for n in SOURCE:
            d = real_design(n)
            print(n, json.dumps(d["summary"]), json.dumps(d["checks"]))
        return
    nf = None
    if "--n-features" in sys.argv:
        i = sys.argv.index("--n-features")
        nf = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    scorers = None
    if "--scorers" in sys.argv:
        i = sys.argv.index("--scorers")
        scorers = tuple(sys.argv[i + 1].split(","))
        del sys.argv[i:i + 2]
    import coverage_study_paper2fold as cs
    cs.design = real_design
    if nf:
        cs.N_FEATURES = nf
    if scorers:
        cs.SCORERS = scorers
    cs.main()


if __name__ == "__main__":
    main()
