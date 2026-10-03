#!/usr/bin/env python
"""H13: ViT-B/16 over seeds (reads score CSVs only; the GPU training is a separate stage).

Inputs : outputs/reports/camelyon17/frac1/seed{42..46}/vit_b_16_score_comparison.csv
         (seed 42 exists; 43-46 come from `run_rigor_pack.sh --vit-seeds`).
Output : outputs/reports/rigor_pack/vit_seeds/{vit_seeds.csv, vit_seeds_summary.txt}

Verdict rule (precommit H13): locked prediction MSP AUROC >= 0.6947.
  >= 4/5 seeds hit -> "seed-robust"; 3/5 -> "seed-dependent"; <= 2/5 -> "did not replicate".
Also: per-seed MSP rank among the 7 scores, and the ViT cross-seed W (k = n seeds, n = 7),
to sit next to the CNN cross-seed W values from w_from_csv.py (same k -> comparable).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rigor import common as C  # noqa: E402

LOCKED = 0.6947


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--reports", default=None)
    ap.add_argument("--arch", default="vit_b_16")
    args = ap.parse_args()
    root = Path(args.root)
    out = C.ensure_dir(Path(args.reports) if args.reports else C.default_reports(root) / "vit_seeds")

    rows, mats = [], []
    for s in C.SEEDS:
        v = C.metric_vector(root, "camelyon17", args.arch, s)
        have = bool(np.isfinite(v).all())
        msp = v[C.METHODS_ORDER.index("MSP")]
        rank = int(C.ranks_rows(v[None, :])[0, C.METHODS_ORDER.index("MSP")]) if have else None
        rows.append({"seed": s, "complete": have, "msp_auroc": msp,
                     "msp_ge_locked": bool(np.isfinite(msp) and msp >= LOCKED), "msp_rank_of_7": rank,
                     "best_method": C.METHODS_ORDER[int(np.nanargmax(v))] if np.isfinite(v).any() else None})
        if have:
            mats.append(v)
    df = pd.DataFrame(rows)
    df.to_csv(out / "vit_seeds.csv", index=False)

    n = int(df.complete.sum())
    hits = int(df.msp_ge_locked.sum())
    if n < 5:
        verdict = "INCOMPLETE (%d/5 seeds present) - run `bash run_rigor_pack.sh --vit-seeds` first" % n
    elif hits >= 4:
        verdict = "seed-robust locked prediction"
    elif hits == 3:
        verdict = "seed-dependent"
    else:
        verdict = "seed-42 hit did not replicate"
    w = float(C.kendall_w(C.ranks_rows(np.stack(mats)))) if len(mats) >= 2 else float("nan")
    C.write_text(out / "vit_seeds_summary.txt", [
        "H13 ViT-B/16 over seeds (locked MSP >= %.4f)" % LOCKED,
        "seeds complete: %d/5, hits: %d" % (n, hits),
        "verdict: %s" % verdict,
        "ViT cross-seed W (k=%d seeds, n=7 scores): %.3f" % (len(mats), w),
    ])
    print((out / "vit_seeds_summary.txt").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
