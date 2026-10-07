#!/usr/bin/env python3
"""Track D two-channel table (post-hoc, descriptive; requested 2026-10-07 after the preliminary Track D read).

Within each isbi_patch2 slide-disjoint retrained model (ResNet50 / ConvNeXt-T / DenseNet121 x seeds 42-44 x
2 folds, scorer fitted on the fold's train slides, float64): seen-slide id_val vs unseen-slide id_val, OOD =
hospital 2, all 7 scores. Seen slides were used by both the backbone and the scorer fit; unseen by neither, so
  logit scores (MSP / Energy / ELogitNorm / ReAct): gap = backbone channel
  feature scores (Mahalanobis / kNN / ViM):         gap = backbone + scorer-fit channels
Scorer-fit-only Δ_fit (Track A, published backbone) is shown beside it. Fold 1 has unseen-slide accuracy
< 0.8 in every cell (confound flag), so folds are also reported separately.

Writes outputs/reports/rigor_pack/miccai_campaign/trackD_backbone_leak/{two_channel_per_fold.csv,
two_channel.csv, two_channel.md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

V2 = C.REPO / "outputs/reports/rigor_pack/leakage/logit_retrain_slide_disjoint_v2"
OUT = C.REPO / "outputs/reports/rigor_pack/miccai_campaign/trackD_backbone_leak"
ARCHS, SEEDS = ("resnet50", "convnext_tiny", "densenet121"), (42, 43, 44)
LOGIT, FEAT = ("MSP", "Energy", "ELogitNorm", "ReAct"), ("Mahalanobis", "kNN", "ViM")
NEAR = (0.45, 0.55)


def main() -> None:
    rows = []
    for a in ARCHS:
        for s in SEEDS:
            for f in (0, 1):
                sc = json.loads((V2 / ("scores_%s_s%d_f%d.json" % (a, s, f))).read_text())
                tr = json.loads((V2 / ("%s_s%d_f%d.json" % (a, s, f))).read_text())
                r = {"arch": a, "seed": s, "fold": f, "acc_seen": tr["acc_id_seen"], "acc_unseen": tr["acc_id_unseen"],
                     "confound_acc_lt_0.8": tr["acc_id_unseen"] < 0.8}
                for m in LOGIT + FEAT:
                    se, un = sc["auroc_%s_seen" % m], sc["auroc_%s_unseen" % m]
                    r["seen_%s" % m], r["unseen_%s" % m] = se, un
                    r["gap_%s" % m] = se - un
                    r["near_chance_%s" % m] = all(NEAR[0] <= x <= NEAR[1] for x in (se, un))
                    r["below_chance_%s" % m] = max(se, un) < 0.5
                rows.append(r)
    P = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    P.to_csv(OUT / "two_channel_per_fold.csv", index=False)

    side = pd.read_csv(OUT / "side_by_side.csv")
    agg = []
    for (a, sel), g in [((a, "both folds"), P[P.arch == a]) for a in ARCHS] + \
                       [((a, "fold 0 (acc >= 0.8)"), P[(P.arch == a) & (P.fold == 0)]) for a in ARCHS] + \
                       [((a, "fold 1 (acc < 0.8)"), P[(P.arch == a) & (P.fold == 1)]) for a in ARCHS]:
        r = {"arch": a, "folds": sel, "n_cells": len(g), "acc_seen": g.acc_seen.mean(), "acc_unseen": g.acc_unseen.mean()}
        for m in LOGIT + FEAT:
            r["gap_%s" % m] = g["gap_%s" % m].mean()
        r["logit_gap_median"] = float(np.median([r["gap_%s" % m] for m in ("MSP", "Energy", "ELogitNorm")]))
        r["feature_gap_median"] = float(np.median([r["gap_%s" % m] for m in FEAT]))
        r["any_below_chance"] = ",".join(m for m in LOGIT + FEAT if g["below_chance_%s" % m].any())
        sa = side[side.arch == a]
        r["scorer_fit_dfit_feature_median_trackA"] = sa.a_feature_dfit_median.median()
        agg.append(r)
    A = pd.DataFrame(agg)
    A.to_csv(OUT / "two_channel.csv", index=False)

    cols = ["arch", "folds", "acc_unseen", "gap_MSP", "gap_Energy", "gap_ReAct", "gap_Mahalanobis", "gap_kNN", "gap_ViM",
            "logit_gap_median", "feature_gap_median", "scorer_fit_dfit_feature_median_trackA", "any_below_chance"]
    lines = ["# Track D two-channel table (post-hoc, descriptive)", "",
             "Within-model seen - unseen AUROC on the isbi_patch2 slide-disjoint retrained models (mean over seeds 42-44"
             " and the listed folds; scorer fitted on the fold's train slides, float64). Logit gap = backbone channel;"
             " feature gap = backbone + scorer-fit channels. Last numeric column: Track A scorer-fit-only Δ_fit (published"
             " backbone, median over available seeds). Fold 1: unseen-slide accuracy < 0.8 in every cell (confound).", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in A[cols].iterrows():
        lines.append("| " + " | ".join(("%+.3f" % v if isinstance(v, float) and not c.startswith("acc") else
                                        "%.3f" % v if isinstance(v, float) else str(v)) for c, v in r.items()) + " |")
    (OUT / "two_channel.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
