#!/usr/bin/env python3
"""P3 budget, P4 and P5 readout of decisions/precommit_isbi_patch2_2026-10-06.md.

Reads leakage/logit_retrain_slide_disjoint_v2/{arch}_s{s}_f{f}.json (+ scores_*.json) and writes
outputs/reports/rigor_pack/isbi_patch2/{phaseB_per_fold.csv, phaseB_per_fold.md, same_protocol_h5a.csv, README.md}.
--projection-only: print the P3 seed-44 budget projection from the smoke run and exit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ARCHS = ("resnet50", "convnext_tiny", "densenet121")
SEEDS = (42, 43, 44)
F3 = ("Mahalanobis", "kNN", "ViM")
F2 = ("Mahalanobis", "kNN")
BAR = 0.02
SMOKE = ("resnet50", 42, 0)
ESTIMATE = "17-18 GPU-h (precommit: ~16 h training + ~1-2 h extraction)"


def projection(v2: Path, v1: Path, sizes) -> tuple[float, dict]:
    sm = json.loads((v2 / ("%s_s%d_f%d.json" % SMOKE)).read_text())
    per_patch = {}
    for a in ARCHS:
        js = [json.loads((v1 / ("%s_fold%d.json" % (a, f))).read_text()) for f in (0, 1)]
        per_patch[a] = np.mean([np.mean(j["epoch_seconds"]) / j["n_train"] for j in js])
    fac = {a: per_patch[a] / per_patch["resnet50"] for a in ARCHS}
    t_smoke = sm["train_seconds"] + sm["extract_seconds"]
    tot = sum(t_smoke * fac[a] * sizes[f] / sizes[SMOKE[2]] for a in ARCHS for _ in SEEDS for f in (0, 1))
    return tot / 3600, fac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection-only", action="store_true")
    args = ap.parse_args()
    rep = C.default_reports(C.REPO)
    v2, v1 = rep / "leakage/logit_retrain_slide_disjoint_v2", rep / "leakage/logit_retrain_slide_disjoint"
    out = C.ensure_dir(rep / "isbi_patch2")
    sizes = (172943, 129493)
    proj, fac = projection(v2, v1, sizes)
    print("P3 projection: %.1f GPU-h for 18 runs (arch factors %s) -> %s" % (
        proj, {a: round(x, 2) for a, x in fac.items()}, "drop seed 44" if proj > 30 else "keep seed 44"))
    if args.projection_only:
        return 0
    seeds = SEEDS if proj <= 30 else SEEDS[:2]

    pub = {a: dict(zip(C.METHODS_ORDER, C.metric_vector(C.REPO, "camelyon17", a, 42))) for a in ARCHS}
    rows, srows, missing = [], [], []
    for s in seeds:
        for a in ARCHS:
            for f in (0, 1):
                p = v2 / ("%s_s%d_f%d.json" % (a, s, f))
                ps = v2 / ("scores_%s_s%d_f%d.json" % (a, s, f))
                if not p.exists() or not ps.exists():
                    missing.append("%s s%d f%d" % (a, s, f))
                    continue
                j, k = json.loads(p.read_text()), json.loads(ps.read_text())
                r = {key: j[key] for key in ("arch", "seed", "fold", "n_train", "train_tumour_frac", "best_epoch",
                                             "acc_id_seen", "acc_id_unseen", "acc_ood", "pred_tumour_ood")}
                for sc, name in (("msp", "MSP"), ("energy", "Energy")):
                    r["auroc_%s_unseen" % sc] = j["auroc_%s_unseen" % sc]
                    r["auroc_%s_seen" % sc] = j["auroc_%s_seen" % sc]
                    r["gap_%s" % sc] = j["gap_%s" % sc]
                    r["delta_published_%s" % sc] = pub[a][name] - j["auroc_%s_unseen" % sc]
                    r["auroc_%s_unseen_last" % sc] = j["auroc_%s_unseen_last" % sc]
                r["anomalous"] = bool(a == "resnet50" and min(r["auroc_msp_unseen"], r["auroc_energy_unseen"]) < 0.5)
                r["gpu_seconds"] = j["train_seconds"] + j["extract_seconds"]
                rows.append(r)
                sr = {"arch": a, "seed": s, "fold": f}
                sr.update({m: k["auroc_%s_unseen" % m] for m in C.METHODS_ORDER})
                v = np.array([sr[m] for m in C.METHODS_ORDER])
                sr["winner"] = C.METHODS_ORDER[int(np.nanargmax(v))]
                sr["winner_auroc"] = float(np.nanmax(v))
                sr["feature_win_F3"], sr["feature_win_F2"] = sr["winner"] in F3, sr["winner"] in F2
                vs = np.array([k["auroc_%s_seen" % m] for m in C.METHODS_ORDER])
                sr["winner_seen_slides"] = C.METHODS_ORDER[int(np.nanargmax(vs))]
                srows.append(sr)
    df, sp = pd.DataFrame(rows), pd.DataFrame(srows)
    df.to_csv(out / "phaseB_per_fold.csv", index=False, float_format="%.4f")
    sp.to_csv(out / "same_protocol_h5a.csv", index=False, float_format="%.4f")

    fm = df.groupby(["arch", "seed"], sort=False)[["gap_msp", "gap_energy", "delta_published_msp",
                                                     "delta_published_energy", "acc_id_unseen"]].mean().reset_index()
    agg = fm.groupby("arch", sort=False).agg(
        n_seeds=("seed", "size"), gap_msp_mean=("gap_msp", "mean"), gap_msp_sd=("gap_msp", "std"),
        gap_energy_mean=("gap_energy", "mean"), gap_energy_sd=("gap_energy", "std"),
        delta_pub_msp_mean=("delta_published_msp", "mean"), delta_pub_msp_sd=("delta_published_msp", "std"),
        delta_pub_energy_mean=("delta_published_energy", "mean"),
        delta_pub_energy_sd=("delta_published_energy", "std")).reset_index()
    agg["over_bar"] = (agg.gap_msp_mean > BAR) | (agg.gap_energy_mean > BAR)
    n_over = int(agg.over_bar.sum())
    all_small = bool(((agg.gap_msp_mean.abs() <= BAR) & (agg.gap_energy_mean.abs() <= BAR)).all())
    verdict = ("PASS: logit scores also benefit from slide sharing (%d/3 archs > %.2f)" % (n_over, BAR) if n_over >= 2 else
               "MSP/Energy change by <= 0.02 on unseen slides" if all_small else
               "per-arch report only (%d/3 archs > %.2f), no sentence" % (n_over, BAR))
    low = df[df.acc_id_unseen < 0.8]
    acc_rng = df.groupby("fold").acc_id_unseen.agg(["min", "max"])

    def tbl(d, cols, fmt="%.3f"):
        L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in d[cols].iterrows():
            L.append("| " + " | ".join(fmt % x if isinstance(x, (float, np.floating)) else str(x) for x in r) + " |")
        return L

    md = ["# Phase B v2 per fold (balanced folds, seeds %s)" % ", ".join(map(str, seeds)), "",
          "Primary = within-model gap (seen-slide minus unseen-slide id_val AUROC, same model, OOD = hospital 2). "
          "Secondary = published seed-42 AUROC minus unseen-slide retrain AUROC. `anomalous` = ResNet50 cell "
          "below 0.5 (P1: no bug, classifier collapse on hospital 2).", ""]
    md += tbl(df, ["arch", "seed", "fold", "acc_id_seen", "acc_id_unseen", "acc_ood", "auroc_msp_unseen", "auroc_msp_seen",
                   "gap_msp", "auroc_energy_unseen", "auroc_energy_seen", "gap_energy", "delta_published_msp", "anomalous"])
    md += ["", "## Per arch: fold mean, then mean ± SD over seeds", ""]
    md += tbl(agg, list(agg.columns))
    (out / "phaseB_per_fold.md").write_text("\n".join(md) + "\n")

    cnt = sp.groupby("arch", sort=False).agg(cells=("winner", "size"), F3=("feature_win_F3", "sum"),
                                             F2=("feature_win_F2", "sum")).reset_index()
    win_counts = sp.winner.value_counts().to_dict()
    gpu_h = df.gpu_seconds.sum() / 3600
    R = ["# ISBI patch 2 readout", "",
         "Precommit: `decisions/precommit_isbi_patch2_2026-10-06.md` (6477b25), committed before any patch-2 number.", "",
         "## P1 ResNet50 0.27", "",
         "No technical bug (`resnet50_diagnosis.md`): recompute matches, best = last epoch, no constant "
         "predictions on ID; the classifier collapses to 'normal' on hospital 2. Patch-1 numbers kept, labelled "
         "anomalous fold. Precision note: `calc_auroc` casts scores to float32; saturated MSP ties move the "
         "ResNet50 AUROC by <= 0.001 (same convention for all published numbers, not changed).", "",
         "## P2 balanced folds", "",
         "5 + 5 slides per hospital, min train-patch difference, ties by default_rng(20261006): train %d / %d "
         "patches (patch 1: 97,099 / 205,337); id_val 19,099 / 14,461. Exact balance is impossible under 5 + 5 "
         "because hospitals 3 and 4 each have one slide with > 50k patches." % sizes, "",
         "## P3 grid and budget", "",
         "Seeds run: %s. Projection after the smoke (ResNet50 s42 f0): %.1f GPU-h for 18 runs -> %s." % (
             ", ".join(map(str, seeds)), proj, "seed 44 dropped (rule fixed in the precommit)" if proj > 30 else "seed 44 kept"),
         "GPU-h estimate (precommit) %s; actual %.1f GPU-h (train + extraction, %d runs).%s" % (
             ESTIMATE, gpu_h, len(df), "" if not missing else " Missing runs: " + ", ".join(missing)), "",
         "## P4 within-model gap (primary) and published delta (secondary)", ""]
    R += tbl(agg, ["arch", "n_seeds", "gap_msp_mean", "gap_msp_sd", "gap_energy_mean", "gap_energy_sd",
                   "delta_pub_msp_mean", "delta_pub_energy_mean", "over_bar"])
    R += ["", "Bar (gap > %.2f for MSP or Energy on >= 2/3 archs, mean over seeds): **%s**." % (BAR, verdict), "",
          "Unseen-slide accuracy per fold: " + "; ".join("fold %d %.3f-%.3f" % (f, r["min"], r["max"]) for f, r in acc_rng.iterrows()) + ".",
          ("Confound note: unseen-slide accuracy < 0.8 in %d fold(s) (%s); in those cells the gap mixes slide "
           "sharing with poor generalisation to new slides." % (len(low), ", ".join("%s s%d f%d %.3f" % (
               r.arch, r.seed, r.fold, r.acc_id_unseen) for r in low.itertuples()))) if len(low) else
          "No fold has unseen-slide accuracy < 0.8.",
          "Descriptive (not a precommitted bar): gap range per fold, MSP / Energy: " + "; ".join(
              "fold %d %.3f-%.3f / %.3f-%.3f (unseen acc %.2f-%.2f)" % (
                  f, x.gap_msp.min(), x.gap_msp.max(), x.gap_energy.min(), x.gap_energy.max(),
                  x.acc_id_unseen.min(), x.acc_id_unseen.max()) for f, x in df.groupby("fold")) + ".",
          "Anomalous (ResNet50 < 0.5) cells: %d of %d." % (int(df.anomalous.sum()), int((df.arch == "resnet50").sum())), "",
          "## P5 same-protocol H5(a) (all 7 scores, fit = fold train slides, ID = unseen slides)", ""]
    R += tbl(cnt, ["arch", "cells", "F3", "F2"], fmt="%d")
    R += ["", "Feature-space (F3) wins %d/%d cells; F2 %d/%d. Winners: %s. Per cell: `same_protocol_h5a.csv`." % (
        int(sp.feature_win_F3.sum()), len(sp), int(sp.feature_win_F2.sum()), len(sp),
        ", ".join("%s %d" % kv for kv in win_counts.items())), "",
        "## Bug fixes", "", "None in patch 2 (P1 found no bug).", ""]
    (out / "README.md").write_text("\n".join(R) + "\n")
    print("\n".join(R))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
