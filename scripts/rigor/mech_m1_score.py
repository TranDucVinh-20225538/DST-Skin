#!/usr/bin/env python3
"""M1 / F3c scoring of decisions/precommit_leak_mechanism_fix_2026-10-06.md for one (dataset, arch, variant).

Published model, ablated features: Δ = AUROC_same − AUROC_disjoint of the 2-fold fit protocol for the
4 fit scores (same folds / groups as mech_cpu.py), standard-protocol AUROCs of all 7 scores (= F3c) and
ID accuracy. Retrained seed-42 models (folds 0 / 1), ablated ID + OOD: within-model gap
AUROC(seen-group ID) − AUROC(unseen-group ID) for MSP / Energy and ID accuracy, fold mean.
Writes outputs/reports/rigor_pack/mechanism_fix/cells/m1_{ds}_{arch}_{variant}.json
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
import mech_cpu as MC  # noqa: E402

M1 = MC.REPO / "outputs/rigor_pack/mechanism_fix/m1"


def published(ds, arch, variant, ref):
    z = np.load(M1 / f"{ds}_published_{arch}_{variant}.npz")
    for k in ("train", "id", "ood"):
        if len(z[f"{k}_feats"]) != len(ref[f"{k}_feats"]):
            raise SystemExit(f"STOP: {k} size differs from the published features")
    f64 = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
    w, b = f64("fc_weight"), f64("fc_bias")
    tf, fi = ref["train_fold"], ref["id_fold"]
    keep = fi >= 0
    res = {"fails": {k: int(z[f"{k}_fails"]) for k in ("train", "id", "ood")},
           "acc_id": float((f64("id_logits").argmax(1) == z["id_labels"])[keep].mean()),
           "acc_id_original": float((ref["id_logits"].argmax(1) == z["id_labels"])[keep].mean())}
    sc = MC.fit(f64("train_feats"), f64("train_logits"), w, b)
    si, so = MC.scores(sc, f64("id_logits"), f64("id_feats")), MC.scores(sc, f64("ood_logits"), f64("ood_feats"))
    res["standard"] = {m: MC.auroc(si[m][keep], so[m]) for m in C.METHODS_ORDER}
    del sc
    same, dis = {m: [] for m in MC.FIT}, {m: [] for m in MC.FIT}
    for f in (0, 1):
        sel = tf == f
        sc = MC.fit(f64("train_feats")[sel], f64("train_logits")[sel], w, b)
        si, so = MC.scores(sc, f64("id_logits"), f64("id_feats")), MC.scores(sc, f64("ood_logits"), f64("ood_feats"))
        for m in MC.FIT:
            same[m].append(MC.auroc(si[m][fi == f], so[m]))
            dis[m].append(MC.auroc(si[m][fi == 1 - f], so[m]))
        del sc
    res["same_2fold"] = {m: float(np.mean(same[m])) for m in MC.FIT}
    res["disjoint_2fold"] = {m: float(np.mean(dis[m])) for m in MC.FIT}
    res["delta"] = {m: res["same_2fold"][m] - res["disjoint_2fold"][m] for m in MC.FIT}
    return res


def retrained(ds, arch, variant):
    from src.utils.scoring import OODScorer
    fn = {"MSP": OODScorer.score_msp, "Energy": OODScorer.score_energy}
    per = {m: [] for m in fn}
    per_o = {m: [] for m in fn}
    acc, acc_o = [], []
    for f in (0, 1):
        p = M1 / f"{ds}_retrained_{arch}_f{f}_{variant}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        if ds == "camelyon":
            from logit_retrain_slide_disjoint_v2 import fold_assignment
            _, iv, _, vf = fold_assignment(C.load_camelyon_metadata(MC.REPO))
            if not np.array_equal(np.asarray(z["id_idx"]), iv):
                raise SystemExit("STOP: Camelyon id order differs from the v2 retrain")
            o = np.load(MC.REPO / "outputs/rigor_pack/logit_retrain_slide_disjoint_v2" / f"{arch}_s42_f{f}.npz")
            fold, oid, ood_o = o["id_val_fold"], o["val_logits"], o["ood_logits"]
        else:
            import multibench_common as M
            o = np.load(M.feat_dir(ds) / f"{arch}_s42_f{f}.npz")
            if not np.array_equal(o["id_labels"], z["id_labels"]):
                raise SystemExit("STOP: id order differs from the multibench retrain")
            fold, oid, ood_o = o["id_fold"], o["id_logits"], o["ood_logits"]
        y = z["id_labels"]
        acc.append(float((z["id_logits"].argmax(1) == y).mean()))
        acc_o.append(float((oid.argmax(1) == y).mean()))
        for m, g in fn.items():
            so, so_o = g(z["ood_logits"]), g(ood_o)
            per[m].append(MC.auroc(g(z["id_logits"][fold == f]), so) - MC.auroc(g(z["id_logits"][fold == 1 - f]), so))
            per_o[m].append(MC.auroc(g(oid[fold == f]), so_o) - MC.auroc(g(oid[fold == 1 - f]), so_o))
    return {"gap": {m: float(np.mean(v)) for m, v in per.items()},
            "gap_original": {m: float(np.mean(v)) for m, v in per_o.items()},
            "acc_id": float(np.mean(acc)), "acc_id_original": float(np.mean(acc_o))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["camelyon", "iwildcam", "rxrx1"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--variant", required=True, choices=["macenko", "colour", "gray"])
    args = ap.parse_args()
    t0 = time.time()
    ref = MC.load_cell(args.ds, args.arch)
    res = {"ds": args.ds, "arch": args.arch, "variant": args.variant,
           "published": published(args.ds, args.arch, args.variant, ref)}
    del ref
    res["retrained"] = retrained(args.ds, args.arch, args.variant)
    res["seconds"] = round(time.time() - t0, 1)
    out = MC.OUT / f"m1_{args.ds}_{args.arch}_{args.variant}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res)[:2000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
