#!/usr/bin/env python3
"""Collect every p-value the rigor pack produced and apply Holm (FWER) and BH (FDR).

Families are those declared in decisions/decision_precommit_rigor_pack.md (H14). Holm is
applied within each family AND over all tests together ("global"). Only results that stay
significant under the within-family Holm correction may be called significant in the paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=str(C.default_reports(C.REPO)))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    rep = Path(args.reports)
    files = sorted(p for p in rep.rglob("pvalues_family.csv"))
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if len(d):
            d["file"] = str(f.relative_to(rep))
            frames.append(d)
    if not frames:
        print("no p-values found under %s" % rep)
        return
    df = pd.concat(frames, ignore_index=True)
    df["p_holm_family"] = float("nan")
    df["p_bh_family"] = float("nan")
    for fam, g in df.groupby(["family", "domain"]):
        df.loc[g.index, "p_holm_family"] = C.holm(g.p.to_numpy())
        df.loc[g.index, "p_bh_family"] = C.bh(g.p.to_numpy())
    df["p_holm_global"] = C.holm(df.p.to_numpy())
    df["p_bh_global"] = C.bh(df.p.to_numpy())
    df["sig_holm_family"] = df.p_holm_family < args.alpha
    out = rep / "multiple_testing"
    C.ensure_dir(out)
    df.to_csv(out / "pvalues_adjusted.csv", index=False)
    s = df.groupby(["family", "domain"]).agg(n=("p", "size"), n_raw_sig=("p", lambda x: int((x < args.alpha).sum())),
                                             n_holm_sig=("sig_holm_family", "sum"),
                                             n_bh_sig=("p_bh_family", lambda x: int((x < args.alpha).sum()))).reset_index()
    s.to_csv(out / "pvalues_summary.csv", index=False)
    C.write_text(out / "pvalues_summary.txt", ["Multiple-testing summary (alpha=%.2f). Files: %s" % (args.alpha, [str(f) for f in files]),
                                               s.to_string(index=False)])
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
