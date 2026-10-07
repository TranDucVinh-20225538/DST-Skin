#!/usr/bin/env python3
"""MICCAI campaign Tracks C and E on medbench run caches (decisions/precommit_miccai_full_campaign_2026-10-07.md, L4).

Per run npz outputs/rigor_pack/medbench/{ds}/{model}_s42_{arm}.npz (CNN archs or fm_{fm}), std-type arms only:
- Track C: scorer fitted on all training images (as medbench_scores); τ = 5th percentile of the leaky-ID
  (test_seen / seen) scores (95% TPR); on new-group ID (test_unseen [+ unseen_extra] / unseen): realized TPR
  = P(s >= τ), false-alarm rate = 1 − realized TPR; OOD pass rate P(s_ood >= τ) (primary OOD set).
- Track E: medbench_scores.fold_map group folds; scorer fitted on fold-f training groups; ID = seen images
  with a fold. F1 = mean_f AUROC(seen of fold 1−f | fit f) for all 7 scores; F2 = cross-fitted (seen image of
  fold g scored by fit 1−g, OOD scores averaged over the two fits) for the 4 fit scores, standard (full fit)
  AUROC on the same seen images for the 3 logit scores.
Writes outputs/reports/rigor_pack/miccai_campaign/cpu_cells/{ds}_{model}_{arm}.json
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
import medbench_common as MC  # noqa: E402
from medbench_scores import all_scores, auroc, fold_map, groups_of, scorer  # noqa: E402

FIT = ("Mahalanobis", "kNN", "ViM", "ReAct")
ORDER = list(C.METHODS_ORDER)
OUT = MC.REPO / "outputs/reports/rigor_pack/miccai_campaign/cpu_cells"


def tau_metrics(S_leak, S_new, S_ood):
    out = {}
    for m in ORDER:
        t = float(np.percentile(S_leak[m], 5))
        out[m] = {"tau": t, "tpr_leaky": float(np.mean(S_leak[m] >= t)), "tpr_new": float(np.mean(S_new[m] >= t)),
                  "fa_new": float(np.mean(S_new[m] < t)), "ood_pass": float(np.mean(S_ood[m] >= t))}
    return out


def run(ds: str, model: str, armn: str) -> dict:
    t0 = time.time()
    z = np.load(MC.REPO / f"outputs/rigor_pack/medbench/{ds}/{model}_s42_{armn}.npz")
    f64 = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
    w, b = f64("fc_weight"), f64("fc_bias")
    sets = sorted({k[:-6] for k in z.files if k.endswith("_feats")})
    seen = ["test_seen"] if "test_seen" in sets else ["seen"]
    unseen = (["test_unseen"] + (["unseen_extra"] if "unseen_extra" in sets else [])) if "test_seen" in sets else ["unseen"]
    ood = sorted(k for k in sets if k.startswith("ood"))[0]
    cat = lambda names, f: np.concatenate([f(n) for n in names])  # noqa: E731
    sc = scorer(f64("train_feats"), f64("train_logits"), w, b)
    S = {k: all_scores(sc, f64(f"{k}_logits"), f64(f"{k}_feats")) for k in seen + unseen + [ood]}
    del sc
    Ss = {m: cat(seen, lambda n: S[n][m]) for m in ORDER}
    Su = {m: cat(unseen, lambda n: S[n][m]) for m in ORDER}
    res = {"ds": ds, "model": model, "arm": armn, "seed": 42, "ood_set": ood, "n_seen": int(len(Ss["MSP"])),
           "n_unseen": int(len(Su["MSP"])), "n_ood": int(len(S[ood]["MSP"])),
           "trackC": tau_metrics(Ss, Su, S[ood])}

    ks = cat(seen, lambda n: z[f"{n}_keys"])
    gs = groups_of(ds, ks)
    gt = groups_of(ds, z["train_keys"])
    fm = fold_map(ds, gt, z["train_labels"])
    tf = np.array([fm[g] for g in gt])
    fs = np.array([fm.get(g, -1) for g in gs])
    keep = fs >= 0
    f1 = {m: [] for m in ORDER}
    f2_id = {m: np.full(len(fs), np.nan) for m in FIT}
    f2_ood = {m: np.zeros(res["n_ood"]) for m in FIT}
    for f in (0, 1):
        sel = tf == f
        scf = scorer(f64("train_feats")[sel], f64("train_logits")[sel], w, b)
        si = {m: np.concatenate([all_scores(scf, f64(f"{n}_logits"), f64(f"{n}_feats"))[m] for n in seen]) for m in ORDER}
        so = all_scores(scf, f64(f"{ood}_logits"), f64(f"{ood}_feats"))
        del scf
        for m in ORDER:
            f1[m].append(auroc(si[m][fs == 1 - f], so[m]))
        for m in FIT:
            f2_id[m][fs == 1 - f] = si[m][fs == 1 - f]
            f2_ood[m] += so[m] / 2.0
    F1 = {m: float(np.mean(v)) for m, v in f1.items()}
    F2 = {m: auroc(Ss[m][keep], S[ood][m]) for m in ORDER}
    F2.update({m: auroc(f2_id[m][keep], f2_ood[m]) for m in FIT})
    res["trackE"] = {"F1": F1, "F2": F2, "n_seen_fold": [int((fs == k).sum()) for k in (0, 1)],
                     "n_train_fold": [int((tf == k).sum()) for k in (0, 1)]}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(16)
    OUT.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for armn in args.arms:
            dst = OUT / f"{args.ds}_{model}_{armn}.json"
            p = MC.REPO / f"outputs/rigor_pack/medbench/{args.ds}/{model}_s42_{armn}.npz"
            if dst.exists():
                continue
            if not p.exists():
                print("MISSING", p.name, flush=True)
                continue
            r = run(args.ds, model, armn)
            dst.write_text(json.dumps(r, indent=2) + "\n")
            print(dst.name, "%.0fs" % r["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
