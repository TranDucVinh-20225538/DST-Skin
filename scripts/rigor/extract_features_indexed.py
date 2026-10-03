#!/usr/bin/env python3
"""GPU: re-extract per-sample logits + penultimate features WITH sample indices (opt-in).

Why: the existing caches (camelyon17_pilot.py extract stage) store train features in
SHUFFLED order without indices, so train patches cannot be mapped to slides/patients.
That blocks the slide-level leakage check (leakfree_knn.py). Also used to fill any cell
whose original cache is missing.

Reuses camelyon17_pilot.paths / get_model / resolve_recipe, src.datasets.camelyon_ood.get_dataloaders
and src.utils.feature_extractor.extract_features_and_logits. Loads the EXISTING checkpoint
data/models/camelyon17/frac1/seed{S}/{stem}_best.pth; never trains, never overwrites anything:
output goes to outputs/features/camelyon17/frac1/seed{S}/rigor_indexed/{stem}_features.pt with
the same keys as the original cache plus
    train_idx, val_idx, ood_idx      WILDS dataset indices (= metadata.csv row numbers)
    train_slide, val_slide, ood_slide, train_patient, val_patient, ood_patient
    train_transform                  'aug' (as the original pipeline) or 'eval'
--train-transform aug (default) reproduces the original pipeline, where Maha/kNN/ViM/ReAct are
fit on train features extracted under the random TRAIN augmentation; 'eval' is the
deterministic sensitivity variant (written to rigor_indexed_evaltf/).
--verify compares val/ood logits against the original cache (max abs diff, argmax agreement).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def load_pilot():
    spec = importlib.util.spec_from_file_location("camelyon17_pilot", C.REPO / "scripts" / "camelyon17_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def input_size_for(arch: str):
    return 224 if arch == "efficientnet_v2_s" else None


def out_path(feat_root: Path, arch: str, seed: int, train_tf: str) -> Path:
    d = "rigor_indexed" if train_tf == "aug" else "rigor_indexed_evaltf"
    return feat_root / "outputs/features/camelyon17/frac1" / ("seed%d" % seed) / d / ("%s_features.pt" % C.file_stem(arch))


def ckpt_path(feat_root: Path, arch: str, seed: int) -> Path:
    return feat_root / "data/models/camelyon17/frac1" / ("seed%d" % seed) / ("%s_best.pth" % C.file_stem(arch))


def run(arch: str, seed: int, feat_root: Path, num_workers: int, train_tf: str, verify: bool, force: bool) -> str:
    import torch
    from src.datasets.camelyon_ood import build_transform, get_dataloaders
    from src.models.cnn_family import fc_params as cnn_fc_params
    from src.utils.feature_extractor import extract_features_and_logits

    pilot = load_pilot()
    dst = out_path(feat_root, arch, seed, train_tf)
    if dst.exists() and not force:
        print("skip (exists) %s" % dst, flush=True)
        return "exists"
    ck = ckpt_path(feat_root, arch, seed)
    if not ck.exists():
        print("MISSING checkpoint %s -> needs training (stage 'train-missing')" % ck, flush=True)
        return "missing_ckpt"
    isz = input_size_for(arch)
    rec = pilot.resolve_recipe(arch)
    transform = build_transform(arch, train=False, input_size=isz) if train_tf == "eval" else None
    loaders = get_dataloaders(arch, batch_size=int(rec["batch_size"]), num_workers=num_workers, include_ood=True,
                              download=False, train_frac=1.0, seed=seed, transform=transform, shuffle_train=False,
                              input_size=isz)
    device = pilot.get_device()
    model = pilot.get_model(arch, pretrained=False)
    model.load_state_dict(torch.load(ck, map_location=device))
    model.to(device)
    meta = C.load_camelyon_metadata(feat_root)
    t0 = time.time()
    out = {}
    for name, key in (("train", "train"), ("id_val", "val"), ("ood", "ood")):
        loader = loaders[name]
        idx = np.asarray(loader.dataset.subset.indices, dtype=np.int64)
        print("--- %s %s seed %d: %d samples" % (name, arch, seed, len(idx)), flush=True)
        logits, feats = extract_features_and_logits(model, loader, device)
        out["%s_logits" % key] = logits
        out["%s_feats" % key] = feats
        ds = loader.dataset.subset.dataset
        out["%s_labels" % key] = np.asarray(ds.y_array[idx]).astype(np.int64)
        out["%s_idx" % key] = idx
        if meta is not None:
            out["%s_slide" % key] = meta.slide.to_numpy()[idx]
            out["%s_patient" % key] = meta.patient.to_numpy()[idx].astype(str)
    w, b = cnn_fc_params(model)
    out.update({"fc_weight": w, "fc_bias": b, "ood_split": "test", "train_frac": 1.0,
                "input_size": int(isz) if isz else int(pilot.default_input_size(arch)), "train_transform": train_tf})
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print("Saved %s (%.0fs)" % (dst, time.time() - t0), flush=True)
    if verify:
        orig = C.feature_path(feat_root, "camelyon17", arch, seed)
        if orig.exists():
            try:
                d = torch.load(orig, map_location="cpu", weights_only=False)
            except TypeError:
                d = torch.load(orig, map_location="cpu")
            for k in ("val", "ood"):
                a, bb = np.asarray(d["%s_logits" % k]), out["%s_logits" % k]
                print("VERIFY %s %s seed %d %s: max|dlogit|=%.2e argmax agree=%.5f labels equal=%s" % (
                    arch, k, seed, orig.name, float(np.max(np.abs(a - bb))), float(np.mean(a.argmax(1) == bb.argmax(1))),
                    bool(np.array_equal(np.asarray(d["%s_labels" % k]), out["%s_labels" % k]))), flush=True)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-root", default=str(C.REPO))
    ap.add_argument("--archs", nargs="+", default=list(C.ARCHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(C.SEEDS))
    ap.add_argument("--task-id", type=int, default=None, help="SLURM_ARRAY_TASK_ID over arch x seed")
    ap.add_argument("--only-missing", action="store_true", help="only cells whose ORIGINAL cache is missing")
    ap.add_argument("--train-transform", choices=("aug", "eval"), default="aug")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list cells + checkpoint/cache status, no torch/wilds")
    args = ap.parse_args()
    fr = Path(args.feat_root)
    todo = [(a, s) for a in args.archs for s in args.seeds]
    if args.only_missing:
        todo = [(a, s) for a, s in todo if not C.feature_path(fr, "camelyon17", a, s).exists()]
    if args.dry_run:
        for i, (a, s) in enumerate(todo):
            print(i, a, s, "ckpt=%s" % ckpt_path(fr, a, s).exists(),
                  "orig_cache=%s" % C.feature_path(fr, "camelyon17", a, s).exists(),
                  "indexed=%s" % out_path(fr, a, s, args.train_transform).exists())
        return
    if args.task_id is not None:
        if args.task_id >= len(todo):
            print("task %d >= %d cells, nothing to do" % (args.task_id, len(todo)))
            return
        todo = [todo[args.task_id]]
    status = [run(a, s, fr, args.num_workers, args.train_transform, args.verify, args.force) for a, s in todo]
    if "missing_ckpt" in status:
        sys.exit(4)


if __name__ == "__main__":
    main()
