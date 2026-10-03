#!/usr/bin/env python3
"""Per-sample 7-score cache for every arch x seed cell (CPU; one SLURM array task per cell).

Loads the EXISTING feature cache written by camelyon17_pilot.py extract stage
(outputs/features/camelyon17/frac1/seed{S}/{stem}_features.pt), fits src.utils.scoring.OODScorer
with exactly the settings of camelyon17_pilot.analyze_backbone (k=50, ReAct p90, ViM default dim),
and saves per-sample scores to outputs/rigor_pack/scores/{domain}/seed{S}/{stem}.npz:
    id_<method>, ood_<method>  (7 methods, float64; higher = more ID)
    id_logits, ood_logits      (float32, for temperature / accuracy checks)
    id_labels, ood_labels
    [train_idx, id_idx, ood_idx] when the cache came from rigor extract (indexed)

Reproduction check: recomputed AUROC per method vs the tracked *_score_comparison.csv
(seed 42 also vs architecture_invariance_ranks.csv). |diff| <= 2e-3 -> PASS.
Never writes into outputs/features or outputs/reports/camelyon17.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def cells(domains, seeds, archs):
    return [(d, s, a) for d in domains for s in seeds for a in archs]


def to_np(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def load_pt(path: Path) -> dict:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def reference_aurocs(root: Path, domain: str, arch: str, seed: int) -> dict:
    ref = {}
    v = C.metric_vector(root, domain, arch, seed, "AUROC")
    for m, a in zip(C.METHODS_ORDER, v):
        if not np.isnan(a):
            ref[C.INV_DISPLAY[m]] = float(a)
    return ref


def run_cell(root: Path, feat_root: Path, out_root: Path, domain: str, seed: int, arch: str, force: bool,
             prefer_indexed: bool, knn_k: int) -> dict:
    from src.utils.benchmark_metrics import calc_auroc
    from src.utils.scoring import OODScorer

    dst = C.score_cache_path(out_root, domain, arch, seed)
    if dst.exists() and not force:
        print("skip (exists): %s" % dst, flush=True)
        return {"cell": "%s/%d/%s" % (domain, seed, arch), "status": "exists"}
    p_orig = C.feature_path(feat_root, domain, arch, seed, indexed=False)
    p_idx = C.feature_path(feat_root, domain, arch, seed, indexed=True) if domain == "camelyon17" else None
    cand = [p_idx, p_orig] if prefer_indexed else [p_orig, p_idx]
    src = next((p for p in cand if p is not None and p.exists()), None)
    if src is None:
        print("MISSING features for %s seed %d %s (looked at %s)" % (domain, seed, arch, [str(p) for p in cand if p]),
              flush=True)
        return {"cell": "%s/%d/%s" % (domain, seed, arch), "status": "missing_features"}
    t0 = time.time()
    print("load %s" % src, flush=True)
    d = load_pt(src)
    need = ["train_feats", "train_logits", "val_logits", "val_feats", "ood_logits", "ood_feats", "fc_weight", "fc_bias"]
    miss = [k for k in need if k not in d]
    scorer = OODScorer(k_nearest=knn_k, use_react="fc_weight" in d, react_percentile=90.0,
                       use_vim="fc_weight" in d, vim_dim=None)
    scorer.fit(to_np(d["train_feats"]), train_labels=None, train_logits=to_np(d.get("train_logits")),
               fc_weight=to_np(d["fc_weight"]) if "fc_weight" in d else None,
               fc_bias=to_np(d["fc_bias"]) if "fc_bias" in d else None)
    id_z, ood_z = to_np(d["val_logits"]), to_np(d["ood_logits"])
    s_id = scorer.get_all_scores(id_z, to_np(d["val_feats"]))
    s_ood = scorer.get_all_scores(ood_z, to_np(d["ood_feats"]))
    save = {"id_logits": id_z.astype(np.float32), "ood_logits": ood_z.astype(np.float32),
            "id_labels": to_np(d["val_labels"]).astype(np.int64), "ood_labels": to_np(d["ood_labels"]).astype(np.int64),
            "source": np.array(str(src))}
    for k in ("train_idx", "val_idx", "ood_idx"):
        if k in d:
            save[k.replace("val_idx", "id_idx")] = to_np(d[k]).astype(np.int64)
    for m in C.METHOD_KEYS:
        if m in s_id:
            save["id_" + m] = np.asarray(s_id[m], dtype=np.float64)
            save["ood_" + m] = np.asarray(s_ood[m], dtype=np.float64)
    C.ensure_dir(dst.parent)
    np.savez_compressed(dst, **save)
    ref = reference_aurocs(root, domain, arch, seed)
    checks = []
    for m in C.METHOD_KEYS:
        if "id_" + m not in save:
            continue
        a = calc_auroc(save["id_" + m], save["ood_" + m])
        r = ref.get(m, np.nan)
        ok = (not np.isnan(r)) and abs(a - r) <= 2e-3
        checks.append("%s=%.4f(ref %s)%s" % (m, a, "nan" if np.isnan(r) else "%.4f" % r,
                                           "" if np.isnan(r) else (" ok" if ok else " MISMATCH")))
    status = "ok" if all("MISMATCH" not in c for c in checks) else "mismatch"
    print("REPRO %s %s seed %d %s [%s] missing_keys=%s (%.0fs)" % (status.upper(), domain, seed, arch,
                                                                   "; ".join(checks), miss, time.time() - t0), flush=True)
    return {"cell": "%s/%d/%s" % (domain, seed, arch), "status": status}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--feat-root", default=None)
    ap.add_argument("--out", default=None, help="default outputs/rigor_pack")
    ap.add_argument("--domains", nargs="+", default=["camelyon17"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.SEEDS))
    ap.add_argument("--archs", nargs="+", default=list(C.ARCHS))
    ap.add_argument("--task-id", type=int, default=None, help="SLURM_ARRAY_TASK_ID; default = all cells")
    ap.add_argument("--list", action="store_true", help="print the cell list and exit")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prefer-indexed", action="store_true")
    ap.add_argument("--knn-k", type=int, default=50)
    args = ap.parse_args()
    root = Path(args.root)
    feat_root = Path(args.feat_root) if args.feat_root else root
    out_root = Path(args.out) if args.out else C.default_out(root)
    todo = cells(args.domains, args.seeds, args.archs)
    if args.list:
        for i, c in enumerate(todo):
            print(i, *c)
        return
    if args.task_id is not None:
        todo = [todo[args.task_id]]
    res = [run_cell(root, feat_root, out_root, d, s, a, args.force, args.prefer_indexed, args.knn_k) for d, s, a in todo]
    bad = [r for r in res if r["status"] in ("missing_features",)]
    if bad:
        print("missing features for %d cell(s); run stage 'extract' (GPU) first" % len(bad), flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
