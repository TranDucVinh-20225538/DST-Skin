#!/usr/bin/env python3
"""Slide-level leakage check for the feature-space scores (needs extract_features_indexed.py).

WILDS id_val (the ID side of every AUROC in the paper) is drawn from the SAME slides as the
training patches. Mahalanobis/kNN are fit on training features, so ID-val patches can have
near-duplicate neighbours from their own slide, inflating ID-side scores (and thus AUROC).
Hospital-2 patches never share a slide with training data. Two checks per cell:
  (a) slide-excluded kNN: for each id_val patch, the k=50 cosine neighbours are searched among
      training patches from OTHER slides only (OOD side unchanged). Compared with the standard kNN
      computed from the same indexed features.
  (b) slide-disjoint 2-fold: training slides split in two folds (stratified by hospital);
      OODScorer (Maha + kNN) fit on fold-A training patches, ID evaluated on id_val patches of
      fold-B slides only, OOD = all hospital 2; then swapped; AUROCs averaged.
GPU used for (a) if available (torch matmul + topk); (b) reuses src.utils.scoring.OODScorer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def knn_scores(train_n, q_n, k, train_slide=None, q_slide=None, device="cpu", chunk=1024):
    import torch

    T = torch.as_tensor(train_n, dtype=torch.float32, device=device)
    ts = torch.as_tensor(train_slide, device=device) if train_slide is not None else None
    out = np.empty(len(q_n))
    for i in range(0, len(q_n), chunk):
        q = torch.as_tensor(q_n[i:i + chunk], dtype=torch.float32, device=device)
        sim = q @ T.T
        if ts is not None:
            qs = torch.as_tensor(q_slide[i:i + chunk], device=device)
            sim = sim.masked_fill(qs[:, None] == ts[None, :], float("-inf"))
        top = torch.topk(sim, k, dim=1).values
        out[i:i + chunk] = -(1.0 - top).mean(dim=1).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--feat-root", default=None)
    ap.add_argument("--reports", default=None)
    ap.add_argument("--archs", nargs="+", default=list(C.ARCHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--skip-2fold", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    feat_root = Path(args.feat_root) if args.feat_root else root
    rep = Path(args.reports) if args.reports else C.default_reports(root) / "leakage"
    C.ensure_dir(rep)
    import torch
    from src.utils.benchmark_metrics import calc_auroc
    from src.utils.scoring import OODScorer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = C.load_camelyon_metadata(feat_root)
    rows = []
    for s in args.seeds:
        for a in args.archs:
            p = C.feature_path(feat_root, "camelyon17", a, s, indexed=True)
            if not p.exists():
                print("missing indexed cache %s (run stage extract)" % p, flush=True)
                continue
            try:
                d = torch.load(p, map_location="cpu", weights_only=False)
            except TypeError:
                d = torch.load(p, map_location="cpu")
            g = lambda k: np.asarray(d[k].numpy() if hasattr(d[k], "numpy") else d[k])  # noqa: E731
            if "train_slide" in d:
                tsl, vsl, osl = g("train_slide"), g("val_slide"), g("ood_slide")
                thosp = None
            elif meta is not None:
                tsl, vsl, osl = (meta.slide.to_numpy()[g(k)] for k in ("train_idx", "val_idx", "ood_idx"))
            else:
                print("no slide info for %s" % p)
                continue
            thosp = meta.center.to_numpy()[g("train_idx")] if meta is not None else np.zeros(len(tsl), int)
            tr = OODScorer.l2_normalize(g("train_feats"))
            iv = OODScorer.l2_normalize(g("val_feats"))
            oo = OODScorer.l2_normalize(g("ood_feats"))
            rec = {"arch": a, "seed": s, "n_train": len(tr), "n_id": len(iv), "n_ood": len(oo),
                   "ood_slides_in_train": int(np.isin(np.unique(osl), np.unique(tsl)).sum()),
                   "id_val_frac_same_slide_in_train": float(np.isin(vsl, np.unique(tsl)).mean())}
            k_std_id = knn_scores(tr, iv, args.k, device=device)
            k_std_ood = knn_scores(tr, oo, args.k, device=device)
            k_ex_id = knn_scores(tr, iv, args.k, tsl, vsl, device=device)
            rec["auroc_knn_standard"] = calc_auroc(k_std_id, k_std_ood)
            rec["auroc_knn_slide_excluded"] = calc_auroc(k_ex_id, k_std_ood)
            rec["delta_knn"] = rec["auroc_knn_slide_excluded"] - rec["auroc_knn_standard"]
            rec["median_id_score_shift"] = float(np.median(k_ex_id - k_std_id))
            if not args.skip_2fold:
                rng = np.random.default_rng(1000 + s)
                fold_of = {}
                for h in np.unique(thosp):
                    sl = np.unique(tsl[thosp == h])
                    rng.shuffle(sl)
                    for j, x in enumerate(sl):
                        fold_of[x] = j % 2
                tf = np.array([fold_of[x] for x in tsl])
                vf = np.array([fold_of.get(x, -1) for x in vsl])
                res = {"mahalanobis": [], "knn": [], "mahalanobis_full": [], "knn_full": []}
                for f in (0, 1):
                    sc = OODScorer(k_nearest=args.k)
                    sc.fit(g("train_feats")[tf == f])
                    idm = vf == 1 - f
                    fi = sc.l2_normalize(g("val_feats")[idm])
                    fo = sc.l2_normalize(g("ood_feats"))
                    res["mahalanobis"].append(calc_auroc(sc.score_mahalanobis(fi), sc.score_mahalanobis(fo)))
                    res["knn"].append(calc_auroc(sc.score_knn(fi), sc.score_knn(fo)))
                    fs = sc.l2_normalize(g("val_feats")[vf == f])  # same-fold (leaky) contrast
                    res["mahalanobis_full"].append(calc_auroc(sc.score_mahalanobis(fs), sc.score_mahalanobis(fo)))
                    res["knn_full"].append(calc_auroc(sc.score_knn(fs), sc.score_knn(fo)))
                rec["auroc_maha_2fold_slide_disjoint"] = float(np.mean(res["mahalanobis"]))
                rec["auroc_knn_2fold_slide_disjoint"] = float(np.mean(res["knn"]))
                rec["auroc_maha_2fold_same_slides"] = float(np.mean(res["mahalanobis_full"]))
                rec["auroc_knn_2fold_same_slides"] = float(np.mean(res["knn_full"]))
                rec["delta_maha_2fold"] = rec["auroc_maha_2fold_slide_disjoint"] - rec["auroc_maha_2fold_same_slides"]
                rec["delta_knn_2fold"] = rec["auroc_knn_2fold_slide_disjoint"] - rec["auroc_knn_2fold_same_slides"]
            rows.append(rec)
            print(rec, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(rep / "leakfree_knn.csv", index=False)
    C.write_text(rep / "leakfree_knn.txt", ["Slide-level leakage check (decision_precommit_rigor_pack.md H11)",
                                           df.round(4).to_string(index=False) if len(df) else "no indexed caches found"])


if __name__ == "__main__":
    main()
