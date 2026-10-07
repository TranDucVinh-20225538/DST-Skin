#!/usr/bin/env python3
"""Item 7 (optional, post-hoc): split conformal tumour/normal classification under slide leakage.

Camelyon17 slide-disjoint backbone-retrain arm (logit_retrain_slide_disjoint_v2), fold 0, every arch x seed
cached. Patches: WILDS id_val. Seen slides = id_val slides of fold 0 (their train patches trained the model, their
id_val patches only selected the checkpoint); unseen slides = id_val slides of fold 1 (never used by the model).

Calibration sources, alpha = 0.05, qhat = ceil((n+1)(1-alpha))-th smallest score:
  (i)  all id_val patches of the seen slides
  (ii) id_val patches of a calibration half of the unseen slides
Test = the other half of the unseen slides (same halves for (i) and (ii)). 20 slide splits drawn from
numpy default_rng(0), stratified by slide tumour fraction (slides ranked, consecutive pairs, one of each pair to
calibration at random, odd slide to test). Scores: LAC (1 - p_true, primary) and APS (non-randomised,
E(x,y) = sum of p_k with p_k >= p_y, secondary).

Reads outputs/rigor_pack/logit_retrain_slide_disjoint_v2/{arch}_s{seed}_f0.npz (val_logits, id_val_fold) and the
WILDS metadata. Writes results/r3/7/{per_seed.csv, per_arch.csv, per_split.csv, REPORT.md}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
import common as C  # noqa: E402

FEAT = REPO / "outputs/rigor_pack/logit_retrain_slide_disjoint_v2"
OUT = REPO / "results/r3/7"
ARCHS, SEEDS, FOLD = ("resnet50", "convnext_tiny", "densenet121"), (42, 43, 44), 0
ALPHA, N_SPLITS, SPLIT_RNG = 0.05, 20, 0
CAL_NAMES = {"seen": "(i) seen-slide id_val", "unseen": "(ii) unseen-slide cal half"}


def softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def lac_all(p):
    return 1.0 - p


def aps_all(p):
    return (p[:, None, :] * (p[:, None, :] >= p[:, :, None])).sum(2)


def qhat(s_true):
    n = len(s_true)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    return np.inf if k > n else np.sort(s_true)[k - 1]


def splits(unseen_slides, tumour_frac, rng):
    order = sorted(unseen_slides, key=lambda x: (tumour_frac[x], x))
    out = []
    for _ in range(N_SPLITS):
        cal = []
        for i in range(0, len(order) - 1, 2):
            cal.append(order[i + int(rng.integers(2))])
        out.append(np.array(sorted(cal)))
    return out


def evaluate(sets, y, slide):
    cov = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    per_slide = pd.Series(cov).groupby(slide).mean()
    return {"coverage": cov.mean(), "cov_normal": cov[y == 0].mean(), "cov_tumour": cov[y == 1].mean() if (y == 1).any() else np.nan,
            "slide_mean_cov": per_slide.mean(), "slide_min_cov": per_slide.min(), "mean_set_size": size.mean(),
            "frac_empty": (size == 0).mean(), "frac_both": (size == 2).mean()}


def main() -> int:
    meta = C.load_camelyon_metadata(C.REPO)
    iv = meta[meta.wilds_split.to_numpy() == 1]
    y_all, sl_all = iv.tumor.to_numpy().astype(int), iv.slide.to_numpy()
    rows = []
    split_list = None
    for a in ARCHS:
        for s in SEEDS:
            p = FEAT / ("%s_s%d_f%d.npz" % (a, s, FOLD))
            if not p.exists():
                print("missing %s" % p.name, flush=True)
                continue
            z = np.load(p)
            logits, vf = z["val_logits"], z["id_val_fold"]
            if len(logits) != len(iv):
                raise SystemExit("STOP: %s val_logits rows %d != id_val rows %d" % (p.name, len(logits), len(iv)))
            seen, unseen = vf == FOLD, vf == 1 - FOLD
            if split_list is None:
                un_sl = sorted(np.unique(sl_all[unseen]).tolist())
                tf = iv[unseen].groupby("slide").tumor.mean().to_dict()
                split_list = splits(un_sl, tf, np.random.default_rng(SPLIT_RNG))
                split_vf = vf.copy()
            elif not np.array_equal(vf, split_vf):
                raise SystemExit("STOP: id_val_fold differs across cached files")
            prob = softmax(logits)
            acc_seen = (prob[seen].argmax(1) == y_all[seen]).mean()
            acc_unseen = (prob[unseen].argmax(1) == y_all[unseen]).mean()
            for score, fn in (("LAC", lac_all), ("APS", aps_all)):
                S = fn(prob)
                s_true = S[np.arange(len(y_all)), y_all]
                q_seen = qhat(s_true[seen])
                for k, cal_sl in enumerate(split_list):
                    in_cal = unseen & np.isin(sl_all, cal_sl)
                    test = unseen & ~np.isin(sl_all, cal_sl)
                    q_un = qhat(s_true[in_cal])
                    for src, q, cm in (("seen", q_seen, seen), ("unseen", q_un, in_cal)):
                        r = {"arch": a, "seed": s, "score": score, "cal_source": src, "split": k, "qhat": q,
                             "n_cal_patches": int(cm.sum()), "n_cal_slides": int(np.unique(sl_all[cm]).size),
                             "n_test_patches": int(test.sum()), "n_test_slides": int(np.unique(sl_all[test]).size),
                             "acc_seen": acc_seen, "acc_unseen": acc_unseen}
                        r.update(evaluate(S[test] <= q, y_all[test], sl_all[test]))
                        r["cov_seen_heldout"] = (s_true[seen] <= q).mean() if src == "unseen" else np.nan
                        rows.append(r)
            print("done %s s%d" % (a, s), flush=True)
    if split_list is None:
        raise SystemExit("STOP: no fold-0 cached logits found")
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "per_split.csv", index=False)

    keys = ["arch", "seed", "score", "cal_source"]
    g = D.groupby(keys, sort=False)
    P = g.agg(n_cal_patches_mean=("n_cal_patches", "mean"), n_cal_slides=("n_cal_slides", "mean"),
              n_test_slides=("n_test_slides", "mean"), n_test_patches_mean=("n_test_patches", "mean"),
              coverage_mean=("coverage", "mean"), coverage_min=("coverage", "min"), coverage_max=("coverage", "max"),
              coverage_sd=("coverage", "std"), slide_mean_cov=("slide_mean_cov", "mean"),
              slide_min_cov_min=("slide_min_cov", "min"), cov_normal=("cov_normal", "mean"), cov_tumour=("cov_tumour", "mean"),
              mean_set_size=("mean_set_size", "mean"), set_size_min=("mean_set_size", "min"), set_size_max=("mean_set_size", "max"),
              frac_empty=("frac_empty", "mean"), frac_both=("frac_both", "mean"),
              cov_seen_heldout=("cov_seen_heldout", "mean"), acc_unseen=("acc_unseen", "first")).reset_index()
    P.to_csv(OUT / "per_seed.csv", index=False)

    ga = D.groupby(["arch", "score", "cal_source"], sort=False)
    A = ga.agg(n_seeds=("seed", "nunique"), n_cal_patches_mean=("n_cal_patches", "mean"), n_cal_slides=("n_cal_slides", "mean"),
               n_test_slides=("n_test_slides", "mean"), coverage_mean=("coverage", "mean"), coverage_min=("coverage", "min"),
               coverage_max=("coverage", "max"), slide_mean_cov=("slide_mean_cov", "mean"), slide_min_cov_min=("slide_min_cov", "min"),
               cov_normal=("cov_normal", "mean"), cov_tumour=("cov_tumour", "mean"), mean_set_size=("mean_set_size", "mean"),
               frac_empty=("frac_empty", "mean"), frac_both=("frac_both", "mean"),
               cov_seen_heldout=("cov_seen_heldout", "mean"), acc_unseen=("acc_unseen", "mean")).reset_index()
    A.to_csv(OUT / "per_arch.csv", index=False)

    def md(T, cols):
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in T[cols].iterrows():
            cells = []
            for c, v in r.items():
                if c == "cal_source":
                    cells.append(CAL_NAMES[v])
                elif isinstance(v, (float, np.floating)):
                    cells.append("-" if np.isnan(v) else ("%.0f" % v if c.startswith("n_") else "%.4f" % v))
                else:
                    cells.append(str(v))
            out.append("| " + " | ".join(cells) + " |")
        return out

    def ver(score):
        x = A[A.score == score].groupby("cal_source")
        return {k: (v.coverage_mean.mean(), v.coverage_min.min(), v.coverage_max.max()) for k, v in x}

    vl = ver("LAC")
    verdict = ("verdict: LAC, alpha=0.05, fold 0, 3 archs x %d seeds x %d slide splits: calibrating on seen-slide patches gives "
               "coverage %.3f (range %.3f-%.3f) on unseen slides vs %.3f (range %.3f-%.3f) with unseen-slide calibration (target 0.95)"
               % (D.seed.nunique(), N_SPLITS, *vl["seen"], *vl["unseen"]))
    ca = ["arch", "score", "cal_source", "n_seeds", "n_cal_patches_mean", "n_cal_slides", "n_test_slides", "coverage_mean",
          "coverage_min", "coverage_max", "slide_mean_cov", "slide_min_cov_min", "cov_normal", "cov_tumour", "mean_set_size",
          "frac_empty", "frac_both", "cov_seen_heldout", "acc_unseen"]
    cp = ["arch", "seed", "score", "cal_source", "n_cal_patches_mean", "n_cal_slides", "n_test_slides", "coverage_mean",
          "coverage_min", "coverage_max", "coverage_sd", "slide_mean_cov", "slide_min_cov_min", "mean_set_size", "set_size_min",
          "set_size_max", "frac_empty", "frac_both", "cov_seen_heldout", "acc_unseen"]
    lines = ["# Item 7: split conformal (alpha = 0.05) under slide leakage, Camelyon17 slide-disjoint retrain, fold 0", "",
             "## Per arch (pooled over seeds and %d slide splits)" % N_SPLITS, ""] + md(A, ca) + \
            ["", "## Per arch x seed (over %d slide splits)" % N_SPLITS, ""] + md(P, cp) + \
            ["", "commit: <filled by parent>", "", verdict, "", "- post-hoc (not in precommit)", "- optional item"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
