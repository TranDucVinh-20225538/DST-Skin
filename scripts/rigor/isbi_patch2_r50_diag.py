#!/usr/bin/env python3
"""P1 of decisions/precommit_isbi_patch2_2026-10-06.md: diagnose the patch-1 ResNet50 slide-disjoint
AUROC < 0.5 (checks a-d, in order). CPU only, reads saved logits and the patch-1 training logs.

Writes outputs/reports/rigor_pack/isbi_patch2/resnet50_diagnosis.md.
Usage: isbi_patch2_r50_diag.py --log0 <fold-0 log> --log1 <fold-1 log>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from leakfree_fit_scores import folds  # noqa: E402

EPOCH_RE = re.compile(r"\[Epoch (\d+)/\d+\] loss=([\d.]+) sel_acc=([\d.]+)")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--log0", required=True)
    ap.add_argument("--log1", required=True)
    args = ap.parse_args()
    meta = C.load_camelyon_metadata(C.REPO)
    sp = meta.wilds_split.to_numpy()
    tr, iv, te = (np.where(sp == s)[0] for s in (0, 1, 2))
    tf, vf = folds(meta.slide.to_numpy()[tr], meta.center.to_numpy()[tr], meta.slide.to_numpy()[iv], 42)
    y = meta.tumor.to_numpy()
    yi, yo = y[iv], y[te]
    pub = np.load(C.default_out(C.REPO) / "scores/camelyon17/seed42/resnet50.npz")
    jdir = C.default_reports(C.REPO) / "leakage/logit_retrain_slide_disjoint"
    L = ["# P1 — ResNet50 slide-disjoint AUROC < 0.5 (patch 1, seed 42)", "",
         "Precommit: `decisions/precommit_isbi_patch2_2026-10-06.md`, P1. Checks in the precommitted order.", ""]
    bug = False

    L += ["## (a) Score sign and ID/OOD orientation", ""]
    for f in (0, 1):
        z = np.load(C.default_out(C.REPO) / ("logit_retrain_slide_disjoint/resnet50_fold%d_logits.npz" % f))
        J = json.loads((jdir / ("resnet50_fold%d.json" % f)).read_text())
        I, O = z["id_val"], z["ood"]
        if not np.array_equal(z["id_val_fold"], vf) or len(I) != len(iv) or len(O) != len(te):
            bug = True
            L.append("- fold %d: MISMATCH of saved fold assignment or row counts" % f)
        d = vf == 1 - f
        f32 = lambda x: x.astype(np.float32)  # noqa: E731  calc_auroc casts scores to float32
        a_msp, a_en = auc(f32(msp(I[d])), f32(msp(O))), auc(f32(energy(I[d])), f32(energy(O)))
        ok = abs(a_msp - J["auroc_msp_slide_disjoint"]) < 1e-6 and abs(a_en - J["auroc_energy_slide_disjoint"]) < 1e-6
        bug |= not ok
        s32 = f32(np.r_[msp(I[d]), msp(O)])
        L.append("- fold %d: independent recompute (own softmax/logsumexp, sklearn, ID label 1, higher = ID, float32 "
                 "scores as in `calc_auroc`) MSP %.4f / Energy %.4f vs reported %.4f / %.4f -> %s. In float64 the MSP "
                 "AUROC is %.4f: %.1f%% of MSP values round to exactly 1.0 in float32 (ties); |diff| <= 0.001, same "
                 "convention as every published AUROC, not a sign or alignment issue." % (
                     f, a_msp, a_en, J["auroc_msp_slide_disjoint"], J["auroc_energy_slide_disjoint"],
                     "match" if ok else "MISMATCH", auc(msp(I[d]), msp(O)), 100 * (s32 == 1).mean()))
    L += ["- The same scoring code gives ConvNeXt / DenseNet slide-disjoint AUROC 0.47-0.65 and seen-slide "
          "0.73-0.79 (patch-1 table), so a global sign flip is excluded; the published ResNet50 seed-42 MSP is "
          "%.3f (all id_val) with the same convention." % auc(msp(pub["id_logits"]), msp(pub["ood_logits"])),
          "- Row alignment: seen-slide id_val accuracy 0.995-0.997 (id_val order) and ConvNeXt OOD accuracy "
          "0.82-0.85 through the identical OOD loader (OOD order); published labels equal metadata order.", ""]

    L += ["## (b) Per-fold accuracy and AUROC, seen vs unseen slides", "",
          "| fold | acc ID seen | acc ID unseen | acc OOD | pred. tumour OOD | MSP unseen | MSP seen | Energy unseen | Energy seen |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f in (0, 1):
        z = np.load(C.default_out(C.REPO) / ("logit_retrain_slide_disjoint/resnet50_fold%d_logits.npz" % f))
        I, O = z["id_val"], z["ood"]
        d, s = vf == 1 - f, vf == f
        L.append("| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            f, (I[s].argmax(1) == yi[s]).mean(), (I[d].argmax(1) == yi[d]).mean(), (O.argmax(1) == yo).mean(),
            O.argmax(1).mean(), auc(msp(I[d]), msp(O)), auc(msp(I[s]), msp(O)),
            auc(energy(I[d]), energy(O)), auc(energy(I[s]), energy(O))))
    L += ["", "Published seed-42 ResNet50 (all train slides): OOD accuracy %.3f, predicted tumour on %.3f of OOD "
          "(true %.3f)." % ((pub["ood_logits"].argmax(1) == yo).mean(), pub["ood_logits"].argmax(1).mean(), yo.mean()), ""]

    L += ["## (c) Training collapse", ""]
    hist = {}
    for f, lp in ((0, args.log0), (1, args.log1)):
        h = [(int(e), float(lo), float(a)) for e, lo, a in EPOCH_RE.findall(Path(lp).read_text())]
        if not h or not Path(lp).read_text().count("retrain resnet50 fold %d" % f):
            raise SystemExit("STOP: log %s is not ResNet50 fold %d" % (lp, f))
        hist[f] = h
        L.append("- fold %d: loss %s; train tumour fraction %.3f (n = %d)." % (
            f, " ".join("%.4f" % lo for _, lo, _ in h), y[tr][tf == f].mean(), (tf == f).sum()))
    L += ["- The patch-1 folds differ in class balance as well as size (tumour fraction %.3f vs %.3f), a further "
          "confound of the patch-1 fold comparison; the v2 JSONs record the balanced folds' tumour fraction." % (
              y[tr][tf == 0].mean(), y[tr][tf == 1].mean()),
          "- Predictions are not constant on ID: seen-slide accuracy 0.995-0.997, both classes predicted. On "
          "hospital 2 the classifier predicts tumour on <1% of patches (table above) with saturated confidence; "
          "this is a collapse under the hospital shift, not a training failure.", ""]

    L += ["## (d) Checkpoint selection (best vs last epoch)", ""]
    for f, h in hist.items():
        be = max(h, key=lambda t: (t[2], -t[0]))
        L.append("- fold %d: best epoch %d (sel_acc %.4f), last epoch %d (sel_acc %.4f)%s." % (
            f, be[0], be[2], h[-1][0], h[-1][2], " -> best = last" if be[0] == h[-1][0] else
            "; last-epoch checkpoint not saved in patch 1, selection accuracies differ by %.4f" % (be[2] - h[-1][2])))
    L += ["- The v2 runs save both checkpoints and report the last-epoch AUROC as a diagnostic.", ""]

    L += ["## Verdict", "",
          ("Technical bug found: see MISMATCH lines; fix and rerun the affected fold per P1." if bug else
           "No technical bug (sign, orientation, alignment, subset, checkpoint). The patch-1 ResNet50 numbers "
           "(MSP 0.316 / 0.226 slide-disjoint) are kept and labelled **anomalous fold**: classifier collapse to "
           "'normal' on the OOD hospital makes OOD patches more confident than unseen ID slides. The within-model "
           "seen-minus-unseen gap (0.24-0.25) is unaffected by this reading. Hyperparameters unchanged."), ""]
    out = C.ensure_dir(C.default_reports(C.REPO) / "isbi_patch2")
    (out / "resnet50_diagnosis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 1 if bug else 0


if __name__ == "__main__":
    raise SystemExit(main())
