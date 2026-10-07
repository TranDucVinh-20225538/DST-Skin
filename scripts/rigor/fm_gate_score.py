#!/usr/bin/env python3
"""Camelyon17 scoring of one FM for decisions/precommit_foundation_leakage_gate_2026-10-07.md (L3, CPU).

Probe = fm_ood_pilot.fit_linear_head on the raw training embeddings (binary logits [-z, z], fc = [[-W], [W]]).
- standard: scorer + probe on all training embeddings; ID = id_val patches with an H11c fold; OOD = hospital 2.
- A-fit (as mech_cpu.run_cell 2-fold part): scorer fitted on fold-f training slides with the full-probe logits;
  same = id_val of fold-f slides, disjoint = id_val of fold-(1-f) slides; fold mean; Δ_fit = same - disjoint.
- logit within-model gap: probe refitted on fold-f training embeddings; gap = AUROC(id fold f) - AUROC(id fold
  1-f) for MSP / Energy / ELogitNorm, fold mean; fold-probe accuracy on seen / unseen id_val.
- cluster bootstrap B = 2000, default_rng(2): id_val slides and OOD images resampled (same draw for every
  score); 95% percentile CIs of Δ_fit and of the logit gaps.
- near-chance: AUROC in [0.45, 0.55] under both same and disjoint (flag only; used by the gate report).
Writes outputs/reports/rigor_pack/foundation_gate/cells/camelyon_{fm}[_smoke].json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from mech_cpu import auroc, fit, scores  # noqa: E402
from medbench_scores import cluster_resample, wauroc  # noqa: E402

REPO = C.REPO
FIT = ("Mahalanobis", "kNN", "ViM", "ReAct")
LOGIT = ("MSP", "Energy", "ELogitNorm")
ORDER = list(C.METHODS_ORDER)
NEAR = (0.45, 0.55)
OUT = REPO / "outputs/reports/rigor_pack/foundation_gate/cells"


def probe(X, y):
    from fm_ood_pilot import fit_linear_head
    t = time.time()
    _, W, b, _, acc = fit_linear_head(X, y)
    W2 = np.vstack([-W, W]).astype(np.float64)
    b2 = np.concatenate([-b, b]).astype(np.float64)
    return W2, b2, acc, round(time.time() - t, 1)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nb", type=int, default=2000)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(16)
    from leakfree_fit_scores import REPRO_FOLD_SIZES, folds
    t0 = time.time()
    name = "camelyon_%s%s" % (args.fm, "_smoke" if args.smoke else "")
    z = np.load(REPO / "outputs/rigor_pack/foundation_gate/feats" / ("%s.npz" % name))
    f64 = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
    Xtr, Xid, Xood = f64("train_feats"), f64("id_feats"), f64("ood_feats")
    ytr, yid = z["train_labels"], z["id_labels"]
    tsl, vsl = z["train_slide"].astype(str), z["id_slide"].astype(str)
    meta = C.load_camelyon_metadata(REPO)
    tf, fi = folds(z["train_slide"], meta.center.to_numpy()[z["train_idx"]], z["id_slide"], 42)
    if not args.smoke and ((tf == 0).sum(), (tf == 1).sum()) != REPRO_FOLD_SIZES:
        raise SystemExit("STOP: H11c fold sizes differ")
    keep = fi >= 0
    res = {"ds": "camelyon", "fm": args.fm, "seed": 42, "smoke": bool(args.smoke), "feat_dim": int(Xtr.shape[1]),
           "n_train": int(len(ytr)), "n_id": int(keep.sum()), "n_ood": int(len(Xood)),
           "n_id_excluded": int((~keep).sum()), "n_train_fold": [int((tf == k).sum()) for k in (0, 1)],
           "n_id_fold": [int((fi == k).sum()) for k in (0, 1)]}

    W, b, acc_tr, sec = probe(Xtr, ytr)
    res["probe"] = {"train_acc": acc_tr, "seconds": sec,
                    "id_acc": float(((Xid @ W.T + b).argmax(1) == yid)[keep].mean())}
    Ltr, Lid, Lood = Xtr @ W.T + b, Xid @ W.T + b, Xood @ W.T + b
    sc = fit(Xtr, Ltr, W, b)
    si, so = scores(sc, Lid, Xid), scores(sc, Lood, Xood)
    res["standard"] = {m: auroc(si[m][keep], so[m]) for m in ORDER}
    del sc
    print("standard %.0fs %s" % (time.time() - t0, res["standard"]), flush=True)

    same, dis, per_fold = {m: [] for m in ORDER}, {m: [] for m in ORDER}, []
    for f in (0, 1):
        sel = tf == f
        sc = fit(Xtr[sel], Ltr[sel], W, b)
        si, so = scores(sc, Lid, Xid), scores(sc, Lood, Xood)
        for m in ORDER:
            same[m].append(auroc(si[m][fi == f], so[m]))
            dis[m].append(auroc(si[m][fi == 1 - f], so[m]))
        per_fold.append(({m: si[m] for m in FIT}, {m: so[m] for m in FIT}))
        del sc
    res["same_2fold"] = {m: float(np.mean(same[m])) for m in ORDER}
    res["disjoint_2fold"] = {m: float(np.mean(dis[m])) for m in ORDER}
    res["delta"] = {m: res["same_2fold"][m] - res["disjoint_2fold"][m] for m in FIT}
    print("A-fit %.0fs %s" % (time.time() - t0, res["delta"]), flush=True)

    from src.utils.scoring import OODScorer
    fns = {"MSP": OODScorer.score_msp, "Energy": OODScorer.score_energy}
    lg = {m: {"seen": [], "unseen": []} for m in LOGIT}
    acc = {"seen": [], "unseen": []}
    per_fold_logit = []
    for f in (0, 1):
        sel = tf == f
        Wf, bf, _, _ = probe(Xtr[sel], ytr[sel])
        Lf_tr, Lf_id, Lf_ood = Xtr[sel] @ Wf.T + bf, Xid @ Wf.T + bf, Xood @ Wf.T + bf
        scf = fit(Xtr[sel], Lf_tr, Wf, bf)
        sid, sood = scores(scf, Lf_id, Xid), scores(scf, Lf_ood, Xood)
        del scf
        for m in LOGIT:
            if m in fns:
                assert np.allclose(fns[m](Lf_id), sid[m]) and np.allclose(fns[m](Lf_ood), sood[m])
            lg[m]["seen"].append(auroc(sid[m][fi == f], sood[m]))
            lg[m]["unseen"].append(auroc(sid[m][fi == 1 - f], sood[m]))
        p = Lf_id.argmax(1) == yid
        acc["seen"].append(float(p[fi == f].mean()))
        acc["unseen"].append(float(p[fi == 1 - f].mean()))
        per_fold_logit.append(({m: sid[m] for m in LOGIT}, {m: sood[m] for m in LOGIT}))
    res["logit_gap"] = {m: {"seen": float(np.mean(v["seen"])), "unseen": float(np.mean(v["unseen"])),
                            "gap": float(np.mean(v["seen"]) - np.mean(v["unseen"]))} for m, v in lg.items()}
    res["fold_probe_acc"] = {k: float(np.mean(v)) for k, v in acc.items()}
    res["confound_unseen_acc_lt_0.8"] = bool(res["fold_probe_acc"]["unseen"] < 0.8)
    print("logit gap %.0fs" % (time.time() - t0), flush=True)

    # cluster bootstrap: id_val slides + OOD images, one draw for every score
    brng = np.random.default_rng(2)
    nood = len(Xood)
    sorted_ = []
    for pf in per_fold + per_fold_logit:
        so = pf[1]
        sorted_.append({m: (np.argsort(so[m], kind="mergesort"), np.sort(so[m], kind="mergesort")) for m in so})
    bd = np.zeros((args.nb, len(FIT)))
    bg = np.zeros((args.nb, len(LOGIT)))
    for bi in range(args.nb):
        wg = cluster_resample(vsl, brng)
        wo = np.bincount(brng.integers(0, nood, nood), minlength=nood).astype(float)
        for fam, ms, arr, off in ((per_fold, FIT, bd, 0), (per_fold_logit, LOGIT, bg, 2)):
            for j, m in enumerate(ms):
                d = 0.0
                for f, (si, _) in enumerate(fam):
                    o, s = sorted_[off + f][m]
                    cum = np.cumsum(wo[o])
                    a_s = wauroc(si[m][fi == f], wg[fi == f], s, cum, cum[-1])
                    a_d = wauroc(si[m][fi == 1 - f], wg[fi == 1 - f], s, cum, cum[-1])
                    d += (a_s - a_d) / 2
                arr[bi, j] = d
    res["delta_ci"] = {m: ci(bd[:, j]) for j, m in enumerate(FIT)}
    res["logit_gap_ci"] = {m: ci(bg[:, j]) for j, m in enumerate(LOGIT)}
    res["near_chance"] = {m: bool(all(NEAR[0] <= res[k][m] <= NEAR[1] for k in ("same_2fold", "disjoint_2fold")))
                          for m in FIT}
    res["near_chance_logit"] = {m: bool(all(NEAR[0] <= res["logit_gap"][m][k] <= NEAR[1] for k in ("seen", "unseen")))
                                for m in LOGIT}
    from scipy.stats import kendalltau
    res["kendall_tau_b_same_vs_disjoint"] = float(kendalltau([res["same_2fold"][m] for m in ORDER],
                                                             [res["disjoint_2fold"][m] for m in ORDER]).statistic)
    res["winner"] = {k: max(ORDER, key=lambda m: res[k][m]) for k in ("standard", "same_2fold", "disjoint_2fold")}
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("%s.json" % name)).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
