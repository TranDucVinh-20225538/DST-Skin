#!/usr/bin/env python3
"""Kermany Track C confound check (post-hoc, descriptive; requested 2026-10-07 after the preliminary Track C read).

Track C pools Kermany new-patient ID = v2 test_unseen (86) + v3 supplement unseen_extra (750). Per CNN (seed 42,
std arm, scorer fitted on all training images in float64 as Track C), separately for test_seen, test_unseen and
unseen_extra: accuracy, class histogram, realized TPR at τ = 5th percentile of test_seen scores, and the same
TPR after subsampling to the class histogram common with test_seen (20 draws, default_rng(0)).

Writes outputs/reports/rigor_pack/miccai_campaign/trackC_clinical_tau/kermany_confound.{csv,md}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medbench_scores as MS  # noqa: E402
import medbench_common as MC  # noqa: E402

ARCHS = ("resnet18", "resnet50", "densenet121", "convnext_tiny", "mobilenet_v3_large", "regnet_y_3_2gf", "effb3",
         "efficientnet_v2_s")
SETS = ("test_seen", "test_unseen", "unseen_extra")
OUT = MC.REPO / "outputs/reports/rigor_pack/miccai_campaign/trackC_clinical_tau"


def matched_idx(y, ref, rng):
    cls = np.unique(np.concatenate([y, ref]))
    k = {c: min((y == c).sum(), (ref == c).sum()) for c in cls}
    return np.concatenate([rng.choice(np.flatnonzero(y == c), k[c], replace=False) for c in cls if k[c] > 0])


def main() -> None:
    rows = []
    for a in ARCHS:
        p = MC.REPO / ("outputs/rigor_pack/medbench/kermany/%s_s42_std.npz" % a)
        if not p.exists():
            continue
        z = np.load(p)
        f64 = lambda k: np.asarray(z[k], dtype=np.float64)  # noqa: E731
        sc = MS.scorer(f64("train_feats"), f64("train_logits"), f64("fc_weight"), f64("fc_bias"))
        S = {k: MS.all_scores(sc, f64("%s_logits" % k), f64("%s_feats" % k)) for k in SETS}
        Y = {k: np.asarray(z["%s_labels" % k]) for k in SETS}
        P = {k: f64("%s_logits" % k).argmax(1) for k in SETS}
        for m in S["test_seen"]:
            tau = float(np.percentile(S["test_seen"][m], 5))
            for k in SETS:
                r = {"arch": a, "score": m, "set": k, "n": len(Y[k]), "acc": float((P[k] == Y[k]).mean()),
                     "class_hist": "/".join(str(int((Y[k] == c).sum())) for c in range(3)),
                     "tpr_at_tau": float(np.mean(S[k][m] >= tau))}
                rng = np.random.default_rng(0)
                r["tpr_at_tau_class_matched"] = float(np.mean(
                    [np.mean(S[k][m][matched_idx(Y[k], Y["test_seen"], rng)] >= tau) for _ in range(20)]))
                rows.append(r)
        print(a, "done", flush=True)
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "kermany_confound.csv", index=False)
    acc = D[D.score == D.score.iloc[0]].pivot_table(index="arch", columns="set", values="acc")
    hist = D[(D.score == D.score.iloc[0]) & (D.arch == D.arch.iloc[0])][["set", "n", "class_hist"]]
    med = D.groupby(["score", "set"])[["tpr_at_tau", "tpr_at_tau_class_matched"]].median().unstack("set")
    lines = ["# Kermany Track C confound check (post-hoc, descriptive)", "",
             "Class order CNV / DME / NORMAL. test_unseen = v2 test, patients absent from v2 train; unseen_extra = v3"
             " supplement (patients absent from v2 train).", "", "## Class histograms", "",
             "```\n" + hist.to_string(index=False) + "\n```", "", "## Accuracy per set (seed 42, std arm)", "", "```\n" + acc.round(3).to_string() + "\n```", "",
             "## Median over archs: realized TPR at τ (95% on test_seen)", "", "```\n" + med.round(3).to_string() + "\n```"]
    (OUT / "kermany_confound.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
