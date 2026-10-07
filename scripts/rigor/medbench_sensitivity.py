#!/usr/bin/env python3
"""Medbench sensitivity readouts on M_std runs (reported only, no bar).

- isic2019 (precommit L1): drop images without lesion_id (singleton groups `IMG:*`) from ID_seen and
  ID_unseen; class-matched gap + cluster-bootstrap CI, Δ_fit point estimate on the remaining seen images.
- kermany (v2 / v3 check): ID_unseen restricted to v2 test images (test_unseen, 86 images, underpowered)
  and to the v3 supplement (unseen_extra) separately.
Writes outputs/reports/rigor_pack/medbench/<ds>/sens_<arch>_s<seed>_std.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medbench_common as MC  # noqa: E402
import medbench_scores as MS  # noqa: E402


def gap_block(Ss, Su, ys, yu, gs, gu, So):
    mt, n = MS.matched_auroc(Ss, Su, ys, yu, So, np.random.default_rng(0))
    bs = MS.bootstrap(Ss, Su, So, ys, yu, gs, gu, n, np.random.default_rng(1))
    return {"n_seen": int(len(ys)), "n_unseen": int(len(yu)), "groups_seen": int(len(np.unique(gs))),
            "groups_unseen": int(len(np.unique(gu))), "matched_counts": {int(c): k for c, k in n.items()},
            "matched": mt, "gap": {m: mt[m]["seen"] - mt[m]["unseen"] for m in MS.ORDER}, "gap_ci": bs["gap_ci"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["isic2019", "kermany"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    ds, tag = args.ds, f"{args.arch}_s{args.seed}_std"
    z = np.load(MC.REPO / f"outputs/rigor_pack/medbench/{ds}/{tag}.npz")
    f64 = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
    w, b = f64("fc_weight"), f64("fc_bias")
    sc = MS.scorer(f64("train_feats"), f64("train_logits"), w, b)
    names = ["test_seen", "test_unseen"] + (["unseen_extra"] if "unseen_extra_feats" in z.files else [])
    S = {k: MS.all_scores(sc, f64(f"{k}_logits"), f64(f"{k}_feats")) for k in names + ["ood"]}
    Y = {k: z[f"{k}_labels"] for k in names}
    G = {k: MS.groups_of(ds, z[f"{k}_keys"]) for k in names}
    res = {"ds": ds, "arch": args.arch, "seed": args.seed}
    if ds == "isic2019":
        ks, ku = ~np.char.startswith(G["test_seen"].astype(str), "IMG:"), ~np.char.startswith(G["test_unseen"].astype(str), "IMG:")
        res["dropped_no_lesion_id"] = {"seen": int((~ks).sum()), "unseen": int((~ku).sum())}
        res["with_lesion_id"] = gap_block({m: S["test_seen"][m][ks] for m in MS.ORDER},
                                          {m: S["test_unseen"][m][ku] for m in MS.ORDER},
                                          Y["test_seen"][ks], Y["test_unseen"][ku], G["test_seen"][ks],
                                          G["test_unseen"][ku], S["ood"])
        gt = MS.groups_of(ds, z["train_keys"])
        fm = MS.fold_map(ds, gt, z["train_labels"])
        tf = np.array([fm[g] for g in gt])
        fs = np.array([fm.get(g, -1) for g in G["test_seen"]])
        same, dis = {m: [] for m in MS.FIT}, {m: [] for m in MS.FIT}
        for f in (0, 1):
            sel = tf == f
            scf = MS.scorer(f64("train_feats")[sel], f64("train_logits")[sel], w, b)
            si = MS.all_scores(scf, f64("test_seen_logits"), f64("test_seen_feats"))
            so = MS.all_scores(scf, f64("ood_logits"), f64("ood_feats"))
            for m in MS.FIT:
                same[m].append(MS.auroc(si[m][(fs == f) & ks], so[m]))
                dis[m].append(MS.auroc(si[m][(fs == 1 - f) & ks], so[m]))
        res["with_lesion_id"]["delta_fit"] = {m: float(np.mean(same[m]) - np.mean(dis[m])) for m in MS.FIT}
    else:
        for nm, u in (("unseen_v2_only", "test_unseen"), ("unseen_v3_only", "unseen_extra")):
            res[nm] = gap_block(S["test_seen"], S[u], Y["test_seen"], Y[u], G["test_seen"], G[u], S["ood"])
            res[nm]["underpowered"] = res[nm]["n_unseen"] < 200 or res[nm]["groups_unseen"] < 20
    out = MC.REPO / f"outputs/reports/rigor_pack/medbench/{ds}/sens_{tag}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v.get("gap", v) if isinstance(v, dict) else v for k, v in res.items()})[:1500], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
