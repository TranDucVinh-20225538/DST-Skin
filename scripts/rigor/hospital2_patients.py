#!/usr/bin/env python3
"""Per-patient patch counts / tumor fractions for Camelyon17 hospital-2 + leakage checks.

Resolves the manuscript wording question ("nine patients split in half" vs "9 per half",
and what "31,878 patches / 85% tumor" refers to). CPU, pandas only, Python 3.8.

Source of truth: WILDS metadata.csv (data/raw/wilds/camelyon17_v1.0/metadata.csv on HPC).
If metadata is absent (e.g. a laptop), falls back to the tracked frozen split file
outputs/reports/camelyon_coverage_val_split_patients.csv, which was written from the same
metadata by scripts/camelyon_coverage_val_split.py.

Leakage checks (need metadata):
  L1 patient IDs of hospital 2 vs hospitals 0/3/4 (train + id_val): must be disjoint.
  L2 slide IDs of hospital 2 vs hospitals 0/3/4: must be disjoint.
  L3 WILDS id_val (the ID test set of every AUROC) shares slides/patients with train:
     report the fraction (by WILDS design id_val patches come from the training slides).
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
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--feat-root", default=None, help="where data/raw/wilds lives (default = --root)")
    ap.add_argument("--reports", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    feat_root = Path(args.feat_root) if args.feat_root else root
    out = Path(args.reports) if args.reports else C.default_reports(root) / "hospital2"
    C.ensure_dir(out)
    lines = []
    frozen = C.frozen_val_patients(root) or []
    meta = C.load_camelyon_metadata(feat_root)
    leak = []
    if meta is not None:
        src = "WILDS metadata.csv"
        h2 = meta[meta.center == 2]
        rows = []
        for p, g in h2.groupby("patient"):
            rows.append({"patient": p, "slides": " ".join(str(s) for s in sorted(g.slide.unique())),
                         "n_patches": len(g), "n_tumor": int(g.tumor.sum()),
                         "tumor_frac": round(float(g.tumor.mean()), 4)})
        per = pd.DataFrame(rows)
        idm = meta[meta.center.isin([0, 3, 4])]
        trn = meta[meta.wilds_split == 0]
        idv = meta[meta.wilds_split == 1]
        leak.append({"check": "L1 patients hospital2 & hospitals034", "n_overlap":
                     len(set(h2.patient) & set(idm.patient)), "must_be": 0})
        leak.append({"check": "L2 slides hospital2 & hospitals034", "n_overlap":
                     len(set(h2.slide) & set(idm.slide)), "must_be": 0})
        leak.append({"check": "L3 id_val patches whose slide is in train (fraction)",
                     "n_overlap": round(float(idv.slide.isin(set(trn.slide)).mean()), 4), "must_be": "report"})
        leak.append({"check": "L3b id_val patches whose patient is in train (fraction)",
                     "n_overlap": round(float(idv.patient.isin(set(trn.patient)).mean()), 4), "must_be": "report"})
        leak.append({"check": "id_val n / train n", "n_overlap": "%d / %d" % (len(idv), len(trn)), "must_be": "33560 / 302436"})
        leak.append({"check": "hospital-2 n", "n_overlap": len(h2), "must_be": 85054})
        leak.append({"check": "id_val patients / slides", "n_overlap": "%d / %d" % (idv.patient.nunique(), idv.slide.nunique()),
                     "must_be": "report"})
    else:
        src = "frozen split CSV (metadata.csv not found; leakage checks skipped)"
        p = root / "outputs/reports/camelyon_coverage_val_split_patients.csv"
        per = pd.read_csv(p, dtype={"patient": str})
        per["n_tumor"] = (per.n_patches * per.tumor_frac).round().astype(int)
        per = per[["patient", "slides", "n_patches", "n_tumor", "tumor_frac"]]
    per["half_frozen_split"] = ["val" if p in frozen else "test" for p in per.patient]
    tot = int(per.n_patches.sum())
    per["share_of_hospital2"] = (per.n_patches / tot).round(4)
    per.to_csv(out / "hospital2_patients.csv", index=False)
    pd.DataFrame(leak).to_csv(out / "leakage_checks.csv", index=False)

    halves = []
    for h, g in per.groupby("half_frozen_split"):
        halves.append({"half": h, "n_patients": len(g), "n_slides": sum(len(str(s).split()) for s in g.slides),
                       "n_patches": int(g.n_patches.sum()), "share_patches": round(g.n_patches.sum() / tot, 4),
                       "tumor_frac": round(float(g.n_tumor.sum() / g.n_patches.sum()), 4),
                       "patients": " ".join(g.patient)})
    hv = pd.DataFrame(halves)
    hv.to_csv(out / "hospital2_frozen_split_halves.csv", index=False)
    big = per.loc[per.n_patches.idxmax()]
    lines += ["Camelyon17 hospital-2 per-patient table. Source: %s" % src,
              per.to_string(index=False), "",
              "Total: %d patients, %d slides, %d patches, tumor fraction %.4f" % (
                  len(per), sum(len(str(s).split()) for s in per.slides), tot, per.n_tumor.sum() / tot), "",
              "Frozen split (StratifiedGroupKFold(2), seed 42, grouped by patient; file camelyon_coverage_val_split_patients.csv):",
              hv.to_string(index=False), "",
              "Largest patient: %s (slide %s): %d patches = %.1f%% of hospital 2, tumor fraction %.4f, frozen half = %s" % (
                  big.patient, big.slides, big.n_patches, 100 * big.share_of_hospital2, big.tumor_frac,
                  big.half_frozen_split),
              "",
              "Reading: hospital 2 has %d patients in TOTAL. The frozen split is by patient into %s; the two halves are"
              " ~half of the PATCHES each, not half of the patients and not 9 per half." % (
                  len(per), " / ".join("%s %d patients" % (r.half, r.n_patients) for r in hv.itertuples())),
              ""]
    if leak:
        lines += ["Leakage checks:", pd.DataFrame(leak).to_string(index=False)]
    C.write_text(out / "hospital2_patients.txt", lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
