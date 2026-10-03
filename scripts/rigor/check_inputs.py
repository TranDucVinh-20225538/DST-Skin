#!/usr/bin/env python3
"""Inventory: which checkpoints / feature caches / score caches / metadata exist on this machine.

Decides whether the GPU extract stage is needed. Prints one line per missing cell and writes
outputs/reports/rigor_pack/inputs_status.csv. --missing-features prints 'arch seed' lines only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def ckpt(feat_root: Path, domain: str, arch: str, seed: int) -> Path:
    stem = C.file_stem(arch, domain)
    if domain == "camelyon17":
        return feat_root / "data/models/camelyon17/frac1" / ("seed%d" % seed) / ("%s_best.pth" % stem)
    if domain == "skin_isic_pad":
        return feat_root / "data/models/skin" / ("" if seed == 42 else "seed%d" % seed) / ("%s_best.pth" % stem)
    return feat_root / "data/models/midog" / ("seed%d" % seed) / ("%s_best.pth" % stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--feat-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reports", default=None)
    ap.add_argument("--domains", nargs="+", default=list(C.DOMAINS))
    ap.add_argument("--extra-archs", nargs="*", default=["vit_b_16"])
    ap.add_argument("--missing-features", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    fr = Path(args.feat_root) if args.feat_root else root
    out_root = Path(args.out) if args.out else C.default_out(root)
    rows = []
    for d in args.domains:
        archs = list(C.ARCHS) + (list(args.extra_archs) if d == "camelyon17" else [])
        for a in archs:
            for s in C.SEEDS:
                rows.append({"domain": d, "arch": a, "seed": s,
                             "score_csv": C.score_csv_path(root, d, a, s).exists() if a in C.ARCHS or d == "camelyon17" else False,
                             "checkpoint": ckpt(fr, d, a, s).exists(),
                             "features": C.feature_path(fr, d, a, s).exists(),
                             "features_indexed": C.feature_path(fr, d, a, s, indexed=True).exists() if d == "camelyon17" else False,
                             "score_cache": C.score_cache_path(out_root, d, a, s).exists()})
    df = pd.DataFrame(rows)
    if args.missing_features:
        for _, r in df[(df.domain == "camelyon17") & df.arch.isin(C.ARCHS) & ~df.features].iterrows():
            print(r.arch, r.seed)
        return
    rep = Path(args.reports) if args.reports else C.default_reports(root)
    C.ensure_dir(rep)
    df.to_csv(rep / "inputs_status.csv", index=False)
    print("metadata.csv present:", C.metadata_path(fr).exists())
    print(df.groupby(["domain"])[["score_csv", "checkpoint", "features", "features_indexed", "score_cache"]].sum().to_string())
    cam = df[(df.domain == "camelyon17") & df.arch.isin(C.ARCHS)]
    miss = cam[~cam.features]
    print("Camelyon 8x5 cells without feature cache: %d" % len(miss))
    for _, r in miss.iterrows():
        print("  need extract: %s seed %d (checkpoint %s)" % (r.arch, r.seed, "present" if r.checkpoint else "MISSING -> retrain"))


if __name__ == "__main__":
    main()
