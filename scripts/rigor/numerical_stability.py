#!/usr/bin/env python3
"""Numerical stability of Mahalanobis / ViM (post-precommit analysis, see
decisions/addendum_numerical_stability_2026-10-05.md).

Stage `score` (CPU array, one task = one (arch, seed, repeat)):
  old    : exactly the published path (src.utils.scoring.OODScorer.fit on the float32 cache,
           OODScorer.score_mahalanobis, -src.utils.ood_vim_react.vim_score with the default
           ViM dim = feat_dim - n_classes), as in camelyon17_pilot.analyze_backbone and
           build_score_cache.py.
  stable : Mahalanobis = L2-normalise + sklearn LedoitWolf fit, all in float64;
           ViM = fit_vim with d = smallest number of principal dims explaining >= 90% of the
           variance of (train_feats - u) (u = ViM origin), float64.
  The 4 original anchors get 10 repeats, the other 4 archs 3; repeat r runs with
  OMP/MKL/OPENBLAS_NUM_THREADS = THREADS[r % 3] (set before numpy is imported).
  Output: outputs/rigor_pack/stability/raw/{stem}_s{seed}_r{r}.npz (+ .json metrics).

Stage `aggregate` (CPU, seconds): spread tables, stable-vs-published table, Kendall W and the
  W REPRO numbers per old repeat and for the stable variant
  -> outputs/reports/rigor_pack/numerical_stability/.

Stage `variants`: builds one input tree per variant (old_r0..old_r9, stable) under
  outputs/rigor_pack/stability/variants/{v}/ with the anchor (old) or all-arch (stable)
  Mahalanobis/ViM AUROC/FPR95 swapped into the score CSVs + ranks file, and the per-sample
  arrays swapped into the score caches; everything else is a symlink to the real tree. The
  unchanged rigor stages are then run on each tree (run_stability_variants.sh).

Stage `verdicts`: reads each variant's reports with the precommit bars (verdict_reader.py)
  and writes the before/after verdict table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

THREADS = (1, 4, 8)
N_REP_ANCHOR = 10
N_REP_OTHER = 3
VAR_FRAC = 0.90
STAB_METHODS = ("mahalanobis", "vim")


def task_list():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common as C

    out = []
    for a in C.ARCHS:
        n = N_REP_ANCHOR if a in C.ANCHORS else N_REP_OTHER
        for s in C.SEEDS:
            for r in range(n):
                out.append((a, s, r, THREADS[r % len(THREADS)]))
    return out


def ensure_threads(t: int) -> None:
    """Re-exec with the BLAS thread env set, so it is in place before numpy loads."""
    if os.environ.get("DST_STAB_THREADS") == str(t):
        return
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "DST_STAB_THREADS"):
        env[k] = str(t)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def vim_dim_for_variance(train_feats, fc_weight, fc_bias, frac: float) -> int:
    import numpy as np
    from numpy.linalg import pinv

    w = np.asarray(fc_weight, dtype=np.float64)
    b = np.asarray(fc_bias, dtype=np.float64)
    u = -np.matmul(pinv(w), b)
    x = np.asarray(train_feats, dtype=np.float64) - u
    cov = x.T @ x / x.shape[0]
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = np.clip(ev, 0.0, None)
    cum = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(cum, frac) + 1)


def score_task(task_id: int, root: Path, out_root: Path, force: bool) -> None:
    a, s, r, t = task_list()[task_id]
    ensure_threads(t)
    import numpy as np
    import torch
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import average_precision_score

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common as C
    from src.utils.benchmark_metrics import calc_auroc, calc_fpr95
    from src.utils.ood_vim_react import fit_vim, vim_score
    from src.utils.scoring import OODScorer

    try:
        from threadpoolctl import threadpool_info

        blas = [(d.get("internal_api"), d.get("num_threads")) for d in threadpool_info()]
    except Exception:
        blas = []
    stem = C.file_stem(a)
    dst = out_root / "stability" / "raw" / ("%s_s%d_r%d.npz" % (stem, s, r))
    if dst.exists() and dst.with_suffix(".json").exists() and not force:
        print("skip (exists): %s" % dst, flush=True)
        return
    t0 = time.time()
    p = C.feature_path(root, "camelyon17", a, s)
    print("task %d: %s seed %d repeat %d threads %d blas %s" % (task_id, a, s, r, t, blas), flush=True)
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        d = torch.load(p, map_location="cpu")
    npy = lambda k: d[k].detach().cpu().numpy() if hasattr(d[k], "detach") else np.asarray(d[k])
    tr, trz = npy("train_feats"), npy("train_logits")
    va, oo = npy("val_feats"), npy("ood_feats")
    fw, fb = npy("fc_weight"), npy("fc_bias")
    del d

    res, arrays = {}, {}
    # old: published code path
    sc = OODScorer(k_nearest=50, use_react=False, use_vim=True, vim_dim=None)
    sc.fit(tr, train_labels=None, train_logits=trz, fc_weight=fw, fc_bias=fb)
    prec_eig = np.linalg.eigvalsh(np.asarray(sc.precision, dtype=np.float64))
    res["old_precision_min_eig"] = float(prec_eig.min())
    res["old_precision_n_neg_eig"] = int(np.sum(prec_eig < 0))
    arrays["old_mahalanobis"] = (sc.score_mahalanobis(sc.l2_normalize(va)), sc.score_mahalanobis(sc.l2_normalize(oo)))
    arrays["old_vim"] = (-vim_score(va, sc.vim_params), -vim_score(oo, sc.vim_params))
    res["old_vim_dim"] = int(sc.vim_params["d"])
    del sc

    # stable
    l2 = lambda x: (lambda x64: x64 / (np.linalg.norm(x64, axis=1, keepdims=True) + 1e-8))(np.asarray(x, dtype=np.float64))
    lw = LedoitWolf().fit(l2(tr))
    mu, prec = np.asarray(lw.location_, dtype=np.float64), np.asarray(lw.precision_, dtype=np.float64)
    res["stable_lw_shrinkage"] = float(lw.shrinkage_)
    ev = np.linalg.eigvalsh(np.asarray(lw.covariance_, dtype=np.float64))
    res["stable_cov_cond"] = float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")
    prec_eig = np.linalg.eigvalsh(prec)
    res["stable_precision_min_eig"] = float(prec_eig.min())
    res["stable_precision_n_neg_eig"] = int(np.sum(prec_eig < 0))

    def maha64(x):
        diff = l2(x) - mu
        d2 = np.sum((diff @ prec) * diff, axis=1)
        return -np.sqrt(np.maximum(d2, 0.0)), int(np.sum(d2 < 0))

    (mi, ni), (mo, no) = maha64(va), maha64(oo)
    arrays["stable_mahalanobis"] = (mi, mo)
    res["stable_mahalanobis_neg_d2"] = ni + no
    del lw
    dim = vim_dim_for_variance(tr, fw, fb, VAR_FRAC)
    vp = fit_vim(train_feats=tr, train_logits=trz, fc_weight=fw, fc_bias=fb, d=dim)
    arrays["stable_vim"] = (-vim_score(va, vp), -vim_score(oo, vp))
    res["stable_vim_dim"] = int(vp["d"])
    res["feat_dim"] = int(tr.shape[1])

    for k, (si, so) in arrays.items():
        si, so = np.asarray(si, dtype=np.float64), np.asarray(so, dtype=np.float64)
        y = np.r_[np.ones(len(si)), np.zeros(len(so))]
        sc_all = np.r_[si, so].astype(np.float32)
        res[k + "_auroc"] = calc_auroc(si, so)
        res[k + "_fpr95"] = calc_fpr95(si, so)[0]
        res[k + "_aupr_in"] = float(average_precision_score(y, sc_all))
        res[k + "_aupr_out"] = float(average_precision_score(1 - y, -sc_all))
        if k == "old_mahalanobis":
            # score_mahalanobis clips negative d^2 to 0 -> score exactly 0 (a real distance is never 0)
            res["old_mahalanobis_neg_d2"] = int(np.sum(si == 0) + np.sum(so == 0))
    res.update({"arch": a, "seed": s, "repeat": r, "threads": t, "blas": str(blas),
                "elapsed_s": round(time.time() - t0, 1)})
    C.ensure_dir(dst.parent)
    np.savez(dst, **{"id_" + k: np.asarray(v[0], dtype=np.float64) for k, v in arrays.items()},
             **{"ood_" + k: np.asarray(v[1], dtype=np.float64) for k, v in arrays.items()})
    dst.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print("RESULT " + " ".join("%s=%s" % (k, ("%.4f" % v) if isinstance(v, float) else v)
                               for k, v in res.items() if k != "blas"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["score", "list", "aggregate", "variants", "verdicts"])
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=None, help="default outputs/rigor_pack")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    out_root = Path(args.out) if args.out else root / "outputs/rigor_pack"
    if args.stage == "list":
        for i, c in enumerate(task_list()):
            print(i, *c)
    elif args.stage == "score":
        score_task(args.task_id, root, out_root, args.force)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import stability_readout as S

        getattr(S, args.stage)(root, out_root)


if __name__ == "__main__":
    main()
