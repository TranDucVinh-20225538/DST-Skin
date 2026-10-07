#!/usr/bin/env python3
"""Report of decisions/precommit_leak_mechanism_fix_2026-10-06.md from the per-cell JSONs
(mech_cpu.py, mech_cpu.py --m4-logit, mech_m1_score.py). Writes mechanism.md/.csv, predictor.md/.csv,
fixes.md/.csv, checklist.md and README.md in outputs/reports/rigor_pack/mechanism_fix/.

Δ = AUROC_same − AUROC_disjoint (> 0 = inflation). Reduction = 1 − effect_variant / effect_original.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ORDER = ["MSP", "Energy", "ELogitNorm", "ViM", "ReAct", "Mahalanobis", "kNN"]
FIT = ["Mahalanobis", "kNN", "ViM", "ReAct"]
DS = ["camelyon", "iwildcam", "rxrx1"]
M1_ARCHS = ["resnet18", "resnet50", "densenet121", "convnext_tiny"]
COLOUR = {"camelyon": "macenko", "iwildcam": "colour", "rxrx1": "colour"}
MIN_EFFECT = 0.01


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def cells(d: Path) -> dict:
    out = {}
    for ds in DS:
        for p in sorted(d.glob(f"{ds}_*.json")):
            j = load(p)
            out[(ds, j["arch"])] = j
    return out


def m1_table(d: Path, C: dict) -> pd.DataFrame:
    rows = []
    for p in sorted(d.glob("m1_*.json")):
        j = load(p)
        ds, arch, v = j["ds"], j["arch"], j["variant"]
        pub = j["published"]
        orig = C[(ds, arch)]["delta"]
        for s in FIT:
            rows.append({"dataset": ds, "arch": arch, "variant": v, "effect": f"delta_{s}", "model": "published",
                         "original": orig[s], "ablated": pub["delta"][s],
                         "acc_id_original": pub["acc_id_original"], "acc_id_ablated": pub["acc_id"],
                         "fails_train": pub["fails"]["train"], "fails_id": pub["fails"]["id"], "fails_ood": pub["fails"]["ood"]})
        r = j.get("retrained")
        if r:
            for s in ("MSP", "Energy"):
                rows.append({"dataset": ds, "arch": arch, "variant": v, "effect": f"gap_{s}", "model": "retrained",
                             "original": r["gap_original"][s], "ablated": r["gap"][s],
                             "acc_id_original": r["acc_id_original"], "acc_id_ablated": r["acc_id"]})
    T = pd.DataFrame(rows)
    T["reduction"] = 1 - T.ablated / T.original
    T["counted"] = T.original.abs() >= MIN_EFFECT
    return T


def m1_verdicts(T: pd.DataFrame) -> dict:
    out = {}
    for ds in DS:
        t = T[T.dataset == ds]
        if not len(t):
            continue
        med = {}
        for v in sorted(t.variant.unique()):
            x = t[t.variant == v]
            med[v] = {"median_reduction": float(x[x.counted].reduction.median()),
                      "median_reduction_all_cells": float(x.reduction.median()),
                      "n_cells": int(x.counted.sum()), "n_excluded_small": int((~x.counted).sum()),
                      "acc_pub_orig": float(x[x.model == "published"].acc_id_original.mean()),
                      "acc_pub_abl": float(x[x.model == "published"].acc_id_ablated.mean())}
        a, g = med.get(COLOUR[ds], {}).get("median_reduction", np.nan), med.get("gray", {}).get("median_reduction", np.nan)
        if a >= 0.5 or g >= 0.5:
            verdict = "group appearance is the main driver"
        elif a < 0.2 and g < 0.2:
            verdict = "not explained by colour / stain"
        else:
            verdict = "partial"
        out[ds] = {"variants": med, "verdict": verdict}
    return out


def m4(d: Path, C: dict) -> dict:
    out = {}
    for ds in DS:
        fit = [C[k]["M4_fit"][s]["R"] for k in C if k[0] == ds for s in FIT]
        lg = []
        for p in sorted(d.glob(f"m4logit_{ds}_*.json")):
            j = load(p)
            lg += [j[s]["R"] for s in ("MSP", "Energy")]
        f_ok, l_ok = [r for r in fit if r is not None], [r for r in lg if r is not None]
        out[ds] = {"fit_median_R": float(np.median(f_ok)) if f_ok else None, "fit_n": len(f_ok),
                   "fit_excluded": len(fit) - len(f_ok),
                   "logit_median_R": float(np.median(l_ok)) if l_ok else None, "logit_n": len(l_ok),
                   "logit_excluded": len(lg) - len(l_ok)}
        for k in ("fit", "logit"):
            r = out[ds][f"{k}_median_R"]
            out[ds][f"{k}_verdict"] = None if r is None else ("group-level effect" if r >= 0.5 else "instance memorisation")
    return out


def predictor(C: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (ds, arch), j in C.items():
        for s in FIT:
            rows.append({"dataset": ds, "arch": arch, "score": s, "delta": j["delta"][s],
                         "M2_balanced_acc": j["M2"]["balanced_acc"], "M2_chance": j["M2"]["chance"],
                         "M3_same_group_nn_frac": j["M3"]["same_group_nn_frac"], "M3_excess": j["M3"]["excess"],
                         "centroid_distance": j["centroid_distance"]})
    P = pd.DataFrame(rows)
    res = {}
    for c in ("M2_balanced_acc", "M3_same_group_nn_frac", "centroid_distance"):
        rho = float(spearmanr(P[c], P.delta).statistic)
        maes = {}
        for held in DS:
            tr, te = P[P.dataset != held], P[P.dataset == held]
            if tr.dataset.nunique() < 1 or not len(te):
                continue
            k, b0 = np.polyfit(tr[c], tr.delta, 1)
            maes[held] = float(np.mean(np.abs(te.delta - (k * te[c] + b0))))
        mae = float(np.mean(list(maes.values())))
        within = {d: float(spearmanr(P[P.dataset == d][c], P[P.dataset == d].delta).statistic) for d in DS}
        res[c] = {"spearman": rho, "lodo_mae": mae, "lodo_mae_per_heldout": maes, "spearman_within": within,
                  "verdict": f"Δ is predictable from {c}" if abs(rho) >= 0.6 and mae < 0.03 else "not predictive"}
    return P, res


def fixes(C: dict, d: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (ds, arch), j in C.items():
        std, f1 = j["standard"], j["F1"]
        vecs = {"F2": (j["F2"], FIT), "F3a": (j["F3a"], FIT), "F3b": (j["F3b"], FIT), "F4": (j["F4"], ["kNN"])}
        p = d / f"m1_{ds}_{arch}_{COLOUR[ds]}.json"
        if p.exists():
            vecs["F3c"] = (dict(std, **{s: load(p)["published"]["standard"][s] for s in FIT}), FIT)
        for fx, (v, changed) in vecs.items():
            a, b = np.array([v[s] for s in ORDER]), np.array([f1[s] for s in ORDER])
            for s in changed:
                rows.append({"dataset": ds, "arch": arch, "fix": fx, "score": s, "fix_auroc": v[s], "F1_auroc": f1[s],
                             "standard_auroc": std[s], "abs_diff_vs_F1": abs(v[s] - f1[s]),
                             "change_vs_standard": v[s] - std[s], "tau_vs_F1": kendalltau(a, b).statistic})
    F = pd.DataFrame(rows)
    res = {}
    for fx in sorted(F.fix.unique()):
        per = {}
        for ds in DS:
            x = F[(F.fix == fx) & (F.dataset == ds)]
            if not len(x):
                continue
            tau = x.groupby("arch").tau_vs_F1.first()
            per[ds] = {"median_abs_diff": float(x.abs_diff_vs_F1.median()), "median_tau": float(tau.median()),
                       "median_change_vs_standard": float(x.change_vs_standard.median()), "n_archs": int(len(tau))}
            per[ds]["works"] = per[ds]["median_abs_diff"] <= 0.02 and per[ds]["median_tau"] >= 0.8
        n_ok = sum(v["works"] for v in per.values())
        works = n_ok >= 2 / 3 * len(per) if len(per) > 1 else None
        res[fx] = {"per_dataset": per, "n_datasets_work": n_ok, "n_datasets": len(per),
                   "verdict": ("works" if works else "does not work") if works is not None
                   else ("works on Camelyon only" if per.get("camelyon", {}).get("works") else "does not work")}
    return F, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reports/rigor_pack/mechanism_fix")
    ap.add_argument("--gpu-hours", type=float, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    cd = out / "cells"
    C = cells(cd)

    T = m1_table(cd, C)
    V1 = m1_verdicts(T)
    M4 = m4(cd, C)
    mech_rows = []
    for (ds, arch), j in sorted(C.items()):
        r = {"dataset": ds, "arch": arch, **{f"delta_{s}": j["delta"][s] for s in FIT},
             "M2_balanced_acc": j["M2"]["balanced_acc"], "M2_chance": j["M2"]["chance"], "M2_groups": j["M2"]["n_groups"],
             "M3_same_group_nn_frac": j["M3"]["same_group_nn_frac"], "M3_chance": j["M3"]["chance"],
             "M3_excess": j["M3"]["excess"], "centroid_distance": j["centroid_distance"]}
        for s in FIT:
            r[f"M4_R_{s}"] = j["M4_fit"][s]["R"]
        mech_rows.append(r)
    MT = pd.DataFrame(mech_rows)
    MT.to_csv(out / "mechanism.csv", index=False)
    T.to_csv(out / "m1_cells.csv", index=False)

    md = ["# Mechanism (M1-M4)", "", "Precommit `decisions/precommit_leak_mechanism_fix_2026-10-06.md`. Δ = AUROC_same − AUROC_disjoint",
          "(2-fold fit protocol, seed 42); reduction = 1 − effect_ablated / effect_original.", "",
          "## M1 stain / colour ablation", "",
          f"Cells with |original effect| < {MIN_EFFECT} are excluded from the median (nothing to reduce; mostly ReAct) and counted.", "",
          "| dataset | variant | median reduction (cells) | all-cells median | excluded | published ID acc original → ablated |",
          "|---|---|---|---|---|---|"]
    for ds, v in V1.items():
        for var, m in v["variants"].items():
            md.append(f"| {ds} | {var} | {m['median_reduction']:+.0%} ({m['n_cells']}) | {m['median_reduction_all_cells']:+.0%} | "
                      f"{m['n_excluded_small']} | {m['acc_pub_orig']:.3f} → {m['acc_pub_abl']:.3f} |")
    md += ["", "Verdicts: " + "; ".join(f"{ds}: {v['verdict']}" for ds, v in V1.items()), "",
           "Per effect (median reduction over archs, counted cells):", "",
           "| dataset | variant | " + " | ".join(f"Δ{s}" for s in FIT) + " | gap MSP | gap Energy |", "|---" * 8 + "|"]
    for (ds, var), g in T[T.counted].groupby(["dataset", "variant"]):
        md.append(f"| {ds} | {var} | " + " | ".join(
            (f"{g[g.effect == e].reduction.median():+.0%}" if (g.effect == e).any() else "—")
            for e in [f"delta_{s}" for s in FIT] + ["gap_MSP", "gap_Energy"]) + " |")
    md += ["", "## M2 group decodability / M3 same-group neighbours / centroid distance", "",
           "| dataset | arch | ΔMaha | ΔkNN | ΔViM | ΔReAct | M2 bal. acc (chance) | M3 frac (chance) | centroid dist |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in MT.iterrows():
        md.append(f"| {r.dataset} | {r.arch} | {r.delta_Mahalanobis:+.3f} | {r.delta_kNN:+.3f} | {r.delta_ViM:+.3f} | "
                  f"{r.delta_ReAct:+.3f} | {r.M2_balanced_acc:.2f} ({r.M2_chance:.3f}) | {r.M3_same_group_nn_frac:.2f} "
                  f"({r.M3_chance:.3f}) | {r.centroid_distance:.4f} |")
    md += ["", "Spearman(M3 excess, Δ) per fit score over archs (descriptive):", ""]
    for ds in DS:
        x = MT[MT.dataset == ds]
        if len(x) > 2:
            md.append(f"- {ds}: " + ", ".join(f"{s} {spearmanr(x.M3_excess, x[f'delta_{s}']).statistic:+.2f}" for s in FIT[:3]))
    md += ["", "## M4 instance vs group", "", "| dataset | fit scores median R (n, excluded) | verdict | MSP/Energy median R (n, excluded) | verdict |",
           "|---|---|---|---|---|"]
    fr = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
    for ds, m in M4.items():
        md.append(f"| {ds} | {fr(m['fit_median_R'])} ({m['fit_n']}, {m['fit_excluded']}) | {m['fit_verdict'] or '—'} | "
                  f"{fr(m['logit_median_R'])} ({m['logit_n']}, {m['logit_excluded']}) | {m['logit_verdict'] or '—'} |")
    md += ["", "## Main-driver sentence", ""]
    for ds in DS:
        if ds in V1:
            md.append(f"- {ds}: M1 → {V1[ds]['verdict']}; M4 → fit scores {M4[ds]['fit_verdict']}"
                      + (f", logit scores {M4[ds]['logit_verdict']}" if M4[ds]['logit_verdict'] else "") + ".")
    (out / "mechanism.md").write_text("\n".join(md) + "\n")

    P, PR = predictor(C)
    P.to_csv(out / "predictor.csv", index=False)
    pm = ["# Predictor (P)", "", f"Unit = (dataset, arch, fit score), n = {len(P)} cells over {P.dataset.nunique()} datasets; target Δ (seed 42).",
          "Bar: |Spearman ρ| >= 0.6 AND leave-one-dataset-out MAE < 0.03.", "",
          "| candidate | ρ (all cells) | LODO MAE | MAE per held-out | ρ within dataset | verdict |", "|---|---|---|---|---|---|"]
    for c, r in PR.items():
        pm.append(f"| {c} | {r['spearman']:+.2f} | {r['lodo_mae']:.3f} | " + ", ".join(f"{k} {v:.3f}" for k, v in r["lodo_mae_per_heldout"].items())
                  + " | " + ", ".join(f"{k} {v:+.2f}" for k, v in r["spearman_within"].items()) + f" | {r['verdict']} |")
    (out / "predictor.md").write_text("\n".join(pm) + "\n")

    F, FR = fixes(C, cd)
    F.to_csv(out / "fixes.csv", index=False)
    fm = ["# Fixes (F)", "", "Reference F1 = group-disjoint ID (2-fold disjoint AUROC). Per dataset: median |AUROC_fix − AUROC_F1| over",
          "the cells the fix changes, median Kendall tau-b (fix 7-score vector vs F1) over archs, median change vs the",
          "standard protocol. Bar: median |diff| <= 0.02 AND median tau >= 0.8 on >= 2/3 of datasets.", "",
          "| fix | dataset | median |diff| vs F1 | median tau | median change vs standard | archs | works |", "|---|---|---|---|---|---|---|"]
    for fx, r in FR.items():
        for ds, p in r["per_dataset"].items():
            fm.append(f"| {fx} | {ds} | {p['median_abs_diff']:.3f} | {p['median_tau']:.2f} | {p['median_change_vs_standard']:+.3f} | "
                      f"{p['n_archs']} | {'yes' if p['works'] else 'no'} |")
    fm += ["", "Verdicts: " + "; ".join(f"{fx} {r['verdict']} ({r['n_datasets_work']}/{r['n_datasets']})" for fx, r in FR.items()),
           "", "F2 = cross-fitted scorer (full ID kept); F3a = per-group centring; F3b = per-batch centring (256);",
           "F3c = Macenko / colour-standardised features (4 archs); F4 = kNN with same-group exclusion (kNN only)."]
    (out / "fixes.md").write_text("\n".join(fm) + "\n")

    pred_ok = [c for c, r in PR.items() if r["verdict"] != "not predictive"]
    ck = ["# Checklist for post-hoc OOD benchmarks with grouped data", "",
          "1. Use a group-disjoint ID split (no slide / patient / location / experiment shared between the scorer's fit set and ID eval), or the cross-fitted scorer (F2) if the full ID set must be kept.",
          "2. Ledoit-Wolf covariance in float64 for Mahalanobis.",
          "3. Fix the BLAS thread count (OMP / OPENBLAS / MKL) and report it; results move at the 1e-6 level otherwise.",
          "4. Report >= 1 seed × >= 6 archs (sample_size_disjoint: min reliable disjoint partition seed 1, arch 6; the Maha/ViM 8-score tree did not stabilise within 6 archs).",
          "5. " + (f"Report the group-leakage statistic {', '.join(pred_ok)} next to the AUROCs." if pred_ok
                   else "No candidate statistic (M2, M3, centroid distance) predicts Δ across datasets; report the group-disjoint AUROC itself.")]
    (out / "checklist.md").write_text("\n".join(ck) + "\n")

    rd = ["# Mechanism / predictor / fix: README", "",
          "Precommit `decisions/precommit_leak_mechanism_fix_2026-10-06.md` (commit 57d8035). Files: mechanism.md/.csv,",
          "m1_cells.csv, predictor.md/.csv, fixes.md/.csv, checklist.md, per-cell JSON in `cells/`.", "",
          "## Pass / fail per bar", "",
          "| item | result |", "|---|---|"]
    for ds, v in V1.items():
        acc = "; ".join(f"{var} ID acc {m['acc_pub_orig']:.3f} → {m['acc_pub_abl']:.3f}" for var, m in v["variants"].items())
        weak = any(m["acc_pub_abl"] < 0.8 * m["acc_pub_orig"] for m in v["variants"].values())
        rd.append(f"| M1 {ds} | {v['verdict']} ({acc}{'; ablation destroys accuracy → reading weakened' if weak else ''}) |")
    for ds, m in M4.items():
        rd.append(f"| M4 {ds} | fit {m['fit_verdict']} (R {fr(m['fit_median_R'])}); logit {m['logit_verdict'] or '—'} (R {fr(m['logit_median_R'])}) |")
    for c, r in PR.items():
        rd.append(f"| P {c} | {r['verdict']} (ρ {r['spearman']:+.2f}, LODO MAE {r['lodo_mae']:.3f}) |")
    for fx, r in FR.items():
        rd.append(f"| F {fx} | {r['verdict']} ({r['n_datasets_work']}/{r['n_datasets']}) |")
    rd += ["", "## Budget", "", "Estimate: ~5-6 GPU-h (M1 re-extraction). Actual: "
           + (f"{args.gpu_hours:.1f} GPU-h." if args.gpu_hours is not None else "n/a."), "",
           "## Technical choices / skipped", "",
           f"- M1 reduction median excludes cells with |original effect| < {MIN_EFFECT} (ratio undefined; counted per row); the all-cells median is shown too.",
           "- M1 within-model gap: Camelyon retrained ISBI-patch-2 v2 seed-42 models exist for ResNet50 / DenseNet121 / ConvNeXt-T only (no ResNet18).",
           "- MIDOG not used (precommit). Medbench datasets are not part of this precommit.", ""]
    (out / "README.md").write_text("\n".join(rd) + "\n")
    print("\n".join(rd))


if __name__ == "__main__":
    main()
