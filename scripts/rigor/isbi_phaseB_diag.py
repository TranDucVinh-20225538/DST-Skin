#!/usr/bin/env python3
"""Per-fold table and failure-mode diagnostics for ISBI patch Phase B (no retraining, no new bars).

Reads the saved logits of logit_retrain_slide_disjoint.py and the published seed-42 scores.
Writes outputs/reports/rigor_pack/isbi_patch/{phaseB_per_fold.csv, phaseB_diag.md}.

Columns per (arch, fold): accuracies (id same slides / id unseen slides / OOD), predicted-tumour
fraction, MSP and Energy AUROC on unseen vs seen id_val slides from the same retrained model
(within-model gap), the published model's AUROC on the same unseen-slide ID subset, and the
sign-flipped MSP AUROC (diagnostic only, not a valid score).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from leakfree_fit_scores import folds  # noqa: E402

ARCHS = ("resnet50", "convnext_tiny", "densenet121")


def msp(z):
    z = z.astype(np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).max(1)


def energy(z):
    z = z.astype(np.float64)
    m = z.max(1)
    return m + np.log(np.exp(z - m[:, None]).sum(1))


def auc(a, b):
    return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))


def main() -> int:
    meta = C.load_camelyon_metadata(C.REPO)
    sp = meta.wilds_split.to_numpy()
    tr, iv, te = (np.where(sp == s)[0] for s in (0, 1, 2))
    _, vf = folds(meta.slide.to_numpy()[tr], meta.center.to_numpy()[tr], meta.slide.to_numpy()[iv], 42)
    yi, yo = meta.tumor.to_numpy()[iv], meta.tumor.to_numpy()[te]
    ldir = C.default_out(C.REPO) / "logit_retrain_slide_disjoint"
    rows = []
    for a in ARCHS:
        pub = np.load(C.default_out(C.REPO) / ("scores/camelyon17/seed42/%s.npz" % a))
        if not (np.array_equal(pub["id_labels"], yi) and np.array_equal(pub["ood_labels"], yo)):
            raise SystemExit("STOP: published label order differs from metadata for %s" % a)
        for f in (0, 1):
            z = np.load(ldir / ("%s_fold%d_logits.npz" % (a, f)))
            if not np.array_equal(z["id_val_fold"], vf):
                raise SystemExit("STOP: fold assignment differs for %s fold %d" % (a, f))
            I, O = z["id_val"], z["ood"]
            d, s = vf == 1 - f, vf == f
            r = {"arch": a, "fold": f,
                 "acc_id_seen": float((I[s].argmax(1) == yi[s]).mean()),
                 "acc_id_unseen": float((I[d].argmax(1) == yi[d]).mean()),
                 "acc_ood": float((O.argmax(1) == yo).mean()),
                 "pred_tumour_id_unseen": float(I[d].argmax(1).mean()),
                 "pred_tumour_ood": float(O.argmax(1).mean()),
                 "true_tumour_ood": float(yo.mean())}
            for name, fn in (("msp", msp), ("energy", energy)):
                si, so = fn(I), fn(O)
                r["auroc_%s_unseen" % name] = auc(si[d], so)
                r["auroc_%s_seen" % name] = auc(si[s], so)
                r["within_gap_%s" % name] = r["auroc_%s_seen" % name] - r["auroc_%s_unseen" % name]
                r["published_%s_same_subset" % name] = auc(fn(pub["id_logits"])[d], fn(pub["ood_logits"]))
            r["auroc_msp_unseen_flipped_diag"] = auc(-msp(I)[d], -msp(O))
            rows.append(r)
    df = pd.DataFrame(rows)
    out = C.ensure_dir(C.default_reports(C.REPO) / "isbi_patch")
    df.to_csv(out / "phaseB_per_fold.csv", index=False, float_format="%.4f")

    pub_acc = {a: float((np.load(C.default_out(C.REPO) / ("scores/camelyon17/seed42/%s.npz" % a))["ood_logits"].argmax(1) == yo).mean())
               for a in ARCHS}
    g = df.groupby("arch", sort=False)[["within_gap_msp", "within_gap_energy"]].agg(["min", "max", "mean"])
    lines = ["# Phase B per-fold table and diagnostics", "",
             "Seed 42, 3 backbones, 2 H11c slide folds; a sensitivity check, not a general claim.",
             "Primary reading here: the within-model gap (same retrained model, id_val on seen slides minus "
             "id_val on unseen slides, same OOD set), because fold 0 trains on fewer slides and is weaker on "
             "unseen slides (published-vs-retrain delta mixes slide leakage with a weaker model).", "",
             "| arch | fold | acc ID seen | acc ID unseen | acc OOD | MSP unseen | MSP seen | gap MSP | "
             "Energy unseen | Energy seen | gap Energy | published MSP (same ID subset) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            r["arch"], r["fold"], r["acc_id_seen"], r["acc_id_unseen"], r["acc_ood"],
            r["auroc_msp_unseen"], r["auroc_msp_seen"], r["within_gap_msp"],
            r["auroc_energy_unseen"], r["auroc_energy_seen"], r["within_gap_energy"], r["published_msp_same_subset"]))
    lines += ["", "Within-model gap range over 6 (arch, fold) cells: MSP %.3f-%.3f, Energy %.3f-%.3f." % (
        df.within_gap_msp.min(), df.within_gap_msp.max(), df.within_gap_energy.min(), df.within_gap_energy.max()), "",
        "## ResNet50 below chance", "",
        "- Not a sign flip: MSP/Energy use the same code path (higher = ID) as ConvNeXt/DenseNet, which stay "
        "above 0.5; the published ResNet50 is already at chance (MSP %.3f on the same ID subsets)." % (
            df[df.arch == "resnet50"].published_msp_same_subset.mean()),
        "- Not a failed training run: loss falls monotonically to <0.006, checkpoint-selection accuracy 0.995-0.997, "
        "accuracy on seen id_val slides %.3f-%.3f." % (
            df[df.arch == "resnet50"].acc_id_seen.min(), df[df.arch == "resnet50"].acc_id_seen.max()),
        "- Failure mode: on hospital 2 the ResNet50 recipe collapses to the 'normal' class (predicts tumour on "
        "%.1f-%.1f%% of OOD patches vs %.0f%% true tumour; OOD accuracy %.3f-%.3f; published model %.3f) "
        "with near-saturated confidence, so OOD patches score as more confident than unseen ID slides. "
        "MSP/Energy AUROC < 0.5 is therefore a confident-collapse case of the classifier, not evidence about "
        "leakage by itself; the within-model gap (seen minus unseen slides) is still %.3f-%.3f." % (
            100 * df[df.arch == "resnet50"].pred_tumour_ood.min(), 100 * df[df.arch == "resnet50"].pred_tumour_ood.max(),
            100 * df.true_tumour_ood.iloc[0], df[df.arch == "resnet50"].acc_ood.min(), df[df.arch == "resnet50"].acc_ood.max(),
            pub_acc["resnet50"], df[df.arch == "resnet50"].within_gap_msp.min(), df[df.arch == "resnet50"].within_gap_msp.max()),
        "- The sign-flipped MSP AUROC (column auroc_msp_unseen_flipped_diag) is a diagnostic only, chosen after "
        "seeing labels, and is not reported as a detector.",
        "- ResNet50 is kept in the table (precommit: no dropping archs after seeing deltas). The precommitted "
        "Phase B bar (|delta| > 0.02 on >= 2/3 archs) is met by ConvNeXt and DenseNet alone.", ""]
    (out / "phaseB_diag.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(g.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
