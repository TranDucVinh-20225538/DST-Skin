#!/usr/bin/env python3
"""Medbench group-leakage report (decisions/precommit_group_leakage_medbench_2026-10-06.md, L4-L5).

Reads outputs/reports/rigor_pack/medbench/<ds>/{train,scores}_*.json and writes
model_quality.csv, <ds>/{A_gap,A_fit,B_gap,bootstrap_ci,ranking}.csv, summary.csv/.md, README.md.

Aggregation (technical choices, recorded in the README):
- gates per run (G1 train json, G2 on the arm's ID_seen set, G3 on the matched AUROCs); failing runs are
  dropped, an arch counts if >= 1 of its runs passes; per arch = mean over passing seeds (and, for
  BreakHis, over the 5 repeats = the pooled cell of deviation 4).
- sample-size floor on the arch cell: A-gap / B-gap use ID_unseen images and groups (summed over the
  BreakHis repeats); Δ_fit uses the smaller fold of the seen set.
- robust bar: point estimate > 0.02 and every passing run's 95% CI excludes 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ORDER = ["MSP", "Energy", "ELogitNorm", "ViM", "ReAct", "Mahalanobis", "kNN"]
FIT = ["Mahalanobis", "kNN", "ViM", "ReAct"]
LOGIT = ["MSP", "Energy", "ELogitNorm"]
FEAT = ["Mahalanobis", "kNN", "ViM"]
CORE = ["resnet18", "resnet50", "densenet121", "convnext_tiny"]
CORE_DS = ["dermamnist", "isic2019", "kermany", "breakhis"]
ALL_DS = CORE_DS + ["brain_cheng"]
BAR = 0.02
G2 = {"dermamnist": ("acc", 0.70), "isic2019": ("bacc", 0.50), "kermany": ("acc", 0.90),
      "breakhis": ("bacc", 0.50), "brain_cheng": ("acc", 0.85)}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def family(arm: str) -> str:
    if arm == "std" or arm.startswith("std_r"):
        return "A"
    if arm in ("b0", "b1"):
        return "B"
    return "other"


def seen_set(j: dict) -> str | None:
    for k in ("test_seen", "seen", "c_test", "test_unseen", "unseen"):
        if f"acc_{k}" in j:
            return k
    return None


def run_rows(rep: Path) -> pd.DataFrame:
    rows = []
    for ds in ALL_DS:
        for p in sorted((rep / ds).glob("scores_*.json")):
            j = load(p)
            t = load(rep / ds / p.name.replace("scores_", "train_"))
            r = {"dataset": ds, "arch": j["arch"], "seed": j["seed"], "arm": j["arm"], "family": family(j["arm"]),
                 "final_train_acc": t["final_train_acc"], "final_loss": t["final_loss"],
                 "epoch1_loss": t["epoch1_loss"], "train_seconds": t["train_seconds"],
                 "n_unseen": j.get("n_unseen"), "groups_unseen": j.get("groups_unseen"),
                 "n_seen": j.get("n_seen"), "groups_seen": j.get("groups_seen")}
            for k in j:
                if k.startswith(("acc_", "bacc_")):
                    r[k] = j[k]
            r["G1"] = bool(t["final_train_acc"] >= 0.80 and t["final_loss"] <= 0.5 * t["epoch1_loss"])
            metric, thr = G2[ds]
            ss = seen_set(j)
            r["G2_set"] = ss
            r["G2_value"] = j.get(f"{metric}_{ss}")
            r["G2"] = bool(r["G2_value"] is not None and r["G2_value"] >= thr)
            if "matched_ood" in j:
                m = j["matched_ood"]
                for s in ORDER:
                    r[f"seen_{s}"], r[f"unseen_{s}"] = m[s]["seen"], m[s]["unseen"]
                    r[f"gap_{s}"] = j["gap_ood"][s]
                    lo, hi = j["bootstrap_ood"]["gap_ci"][s]
                    r[f"gap_lo_{s}"], r[f"gap_hi_{s}"] = lo, hi
                near = [s for s in ORDER if all(0.45 <= m[s][k] <= 0.55 for k in ("seen", "unseen"))]
                r["pair_ci_seen"] = j["bootstrap_ood"]["pair_ci_seen"]
                r["pair_ci_unseen"] = j["bootstrap_ood"]["pair_ci_unseen"]
            else:
                a = j["auroc_all"]
                near = [s for s in ORDER if all(0.45 <= a[k][s] <= 0.55 for k in a)]
                for k in a:
                    for s in ORDER:
                        r[f"auroc_{k}_{s}"] = a[k][s]
            r["near_chance"] = ",".join(near)
            r["G3_inadequate"] = len(near) >= 5
            if "afit" in j:
                f = j["afit"]
                r["n_seen_fold_min"] = min(f["n_seen_fold"])
                for s in FIT:
                    r[f"dfit_{s}"] = f["delta_fit"][s]
                    r[f"dfit_lo_{s}"], r[f"dfit_hi_{s}"] = f["delta_fit_ci"][s]
                    r[f"same_{s}"], r[f"disjoint_{s}"] = f["same"][s], f["disjoint"][s]
            r["passes"] = r["G1"] and r["G2"] and not r["G3_inadequate"]
            rows.append(r)
    return pd.DataFrame(rows)


def diff_ci(pairs: dict, x: str, y: str) -> tuple[float, float]:
    """CI of AUROC(x) - AUROC(y); stored pairs are (first - second) in ORDER."""
    if f"{x}-{y}" in pairs:
        lo, hi = pairs[f"{x}-{y}"]
        return lo, hi
    lo, hi = pairs[f"{y}-{x}"]
    return -hi, -lo


def ranking(R: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in R[R.family.isin(["A", "B"]) & R.passes].iterrows():
        near = set(filter(None, r.near_chance.split(",")))
        sc = [s for s in ORDER if s not in near]
        a = np.array([r[f"seen_{s}"] for s in sc])
        b = np.array([r[f"unseen_{s}"] for s in sc])
        wl, wd = sc[int(a.argmax())], sc[int(b.argmax())]
        sig = False
        if wl != wd:
            sig = bool(diff_ci(r.pair_ci_seen, wl, wd)[0] > 0 and diff_ci(r.pair_ci_unseen, wd, wl)[0] > 0)
        out.append({"dataset": r.dataset, "arch": r.arch, "seed": r.seed, "arm": r.arm, "family": r.family,
                    "winner_leaky": wl, "winner_disjoint": wd, "winner_changed": wl != wd,
                    "swap_ci_significant": sig, "kendall_tau_b": kendalltau(a, b).statistic})
    return pd.DataFrame(out)


def arch_cells(R: pd.DataFrame, fam: str) -> pd.DataFrame:
    """Per (dataset, arch): mean over passing runs (seeds, folds, BreakHis repeats)."""
    P = R[(R.family == fam) & R.passes]
    if fam == "B":
        P = P[P.arch.isin(CORE)]
    cells = []
    for (ds, arch), g in P.groupby(["dataset", "arch"]):
        c = {"dataset": ds, "arch": arch, "n_runs": len(g)}
        if ds == "breakhis":
            per_seed = g.groupby("seed").agg(n=("n_unseen", "sum"), gr=("groups_unseen", "sum"))
            c["n_unseen"], c["groups_unseen"] = int(per_seed.n.min()), int(per_seed.gr.min())
        else:
            c["n_unseen"], c["groups_unseen"] = int(g.n_unseen.min()), int(g.groups_unseen.min())
        c["underpowered_gap"] = c["n_unseen"] < 200 or c["groups_unseen"] < 20
        for s in ORDER:
            c[f"gap_{s}"] = g[f"gap_{s}"].mean()
            c[f"gap_ci_excl0_{s}"] = bool((g[f"gap_lo_{s}"] > 0).all() or (g[f"gap_hi_{s}"] < 0).all())
            c[f"seen_{s}"], c[f"unseen_{s}"] = g[f"seen_{s}"].mean(), g[f"unseen_{s}"].mean()
        if fam == "A" and "dfit_kNN" in g:
            nf = g.n_seen_fold_min.min()
            c["n_seen_fold_min"] = int(nf)
            c["underpowered_fit"] = nf < 200
            for s in FIT:
                c[f"dfit_{s}"] = g[f"dfit_{s}"].mean()
                c[f"dfit_ci_excl0_{s}"] = bool((g[f"dfit_lo_{s}"] > 0).all() or (g[f"dfit_hi_{s}"] < 0).all())
        sk = "acc_test_seen" if "acc_test_seen" in g and g["acc_test_seen"].notna().all() else "acc_seen"
        uk = "acc_test_unseen" if "acc_test_unseen" in g and g["acc_test_unseen"].notna().all() else "acc_unseen"
        c["acc_seen"], c["acc_unseen"] = g[sk].mean(), g[uk].mean()
        c["confound_unseen_acc_lt_0.8"] = bool(c["acc_unseen"] < 0.8)
        cells.append(c)
    return pd.DataFrame(cells)


def a_bar(A: pd.DataFrame, ds: str) -> dict:
    a = A[A.dataset == ds]
    n = len(a)
    res = {}
    for s in ORDER:
        okg = ~a.underpowered_gap
        okf = ~a.underpowered_fit if s in FIT and "underpowered_fit" in a else pd.Series(False, index=a.index)
        hit = (okg & (a[f"gap_{s}"] > BAR))
        rob = (okg & (a[f"gap_{s}"] > BAR) & a[f"gap_ci_excl0_{s}"])
        if s in FIT and f"dfit_{s}" in a:
            hit = hit | (okf & (a[f"dfit_{s}"] > BAR))
            rob = rob | (okf & (a[f"dfit_{s}"] > BAR) & a[f"dfit_ci_excl0_{s}"])
        counted = int((okg | okf).sum())
        res[s] = {"hits": int(hit.sum()), "robust_hits": int(rob.sum()), "counted": counted, "n_archs": n,
                  "present": counted > 0 and hit.sum() >= counted / 2,
                  "present_robust": counted > 0 and rob.sum() >= counted / 2}
    return res


def b_bar(B: pd.DataFrame, ds: str) -> dict:
    b = B[(B.dataset == ds) & ~B.underpowered_gap]
    n = len(b)
    res = {}
    for s in ORDER:
        hit = int((b[f"gap_{s}"] > BAR).sum())
        rob = int(((b[f"gap_{s}"] > BAR) & b[f"gap_ci_excl0_{s}"]).sum())
        res[s] = {"hits": hit, "robust_hits": rob, "n": n,
                  "present": n > 0 and hit >= 2 * n / 3, "present_robust": n > 0 and rob >= 2 * n / 3}
    return res


def sensitivity(rep: Path) -> list[str]:
    """ISIC 2019 no-lesion-id drop (L1) and Kermany v2 / v3 split of ID_unseen (descriptive, no bar)."""
    out = ["## Sensitivity readouts (reported only, not in any bar)", ""]
    rows = []
    for p in sorted(rep.glob("*/sens_*.json")):
        j = load(p)
        for blk in ("with_lesion_id", "unseen_v2_only", "unseen_v3_only"):
            if blk in j:
                b = j[blk]
                r = {"dataset": j["ds"], "arch": j["arch"], "seed": j["seed"], "readout": blk,
                     "n_seen": b["n_seen"], "n_unseen": b["n_unseen"], "groups_unseen": b["groups_unseen"]}
                for s in ORDER:
                    r[f"gap_{s}"] = b["gap"][s]
                    r[f"gap_lo_{s}"], r[f"gap_hi_{s}"] = b["gap_ci"][s]
                for s, v in b.get("delta_fit", {}).items():
                    r[f"dfit_{s}"] = v
                rows.append(r)
    if not rows:
        return out + ["(not run)", ""]
    D = pd.DataFrame(rows)
    D.to_csv(rep / "sensitivity.csv", index=False)
    out += ["Median over archs × seeds (CI-excluding-0 runs / runs in brackets):", "",
            "| dataset | readout | n unseen (median) | " + " | ".join(f"gap {s}" for s in ORDER) + " | Δfit Maha / kNN / ViM |",
            "|---" * (len(ORDER) + 4) + "|"]
    for (ds, rd_), g in D.groupby(["dataset", "readout"]):
        cells = " | ".join(f"{g[f'gap_{s}'].median():+.3f} ({int((g[f'gap_lo_{s}'] > 0).sum())}/{len(g)})" for s in ORDER)
        fz = " / ".join(f"{g[f'dfit_{s}'].median():+.3f}" for s in FEAT) if "dfit_kNN" in g else "—"
        out.append(f"| {ds} | {rd_} | {int(g.n_unseen.median())} | {cells} | {fz} |")
    return out + [""]


def extra_arms(R: pd.DataFrame) -> list[str]:
    """Descriptive arms: Kermany M_std(v3), DermaMNIST-C/-E external arm, BreakHis M_gd (secondary)."""
    out = ["## Descriptive arms (not in bars)", ""]
    v3 = R[(R.dataset == "kermany") & (R.arm == "v3std")]
    if len(v3):
        out.append("- Kermany M_std(v3) on the official v3 test (patient-disjoint), AUROC vs v3 DRUSEN, mean of "
                   f"{len(v3)} runs: " + ", ".join(f"{s} {v3[f'auroc_test_unseen_{s}'].mean():.3f}" for s in ORDER)
                   + f"; acc {v3.acc_test_unseen.mean():.3f}.")
    dmc = R[(R.dataset == "dermamnist") & (R.arm == "dmc")]
    if len(dmc):
        out.append(f"- DermaMNIST-C arm ({len(dmc)} runs), AUROC vs PAD-UFES-20, C test (lesion-disjoint) / E test "
                   "(ISIC 2018 test): " + ", ".join(f"{s} {dmc[f'auroc_c_test_{s}'].mean():.3f} / {dmc[f'auroc_e_test_{s}'].mean():.3f}"
                                                    for s in ORDER)
                   + f"; acc {dmc.acc_c_test.mean():.3f} / {dmc.acc_e_test.mean():.3f}.")
    std = R[(R.dataset == "breakhis") & R.arm.str.startswith("std_r") & R.arch.isin(CORE)]
    gd = R[(R.dataset == "breakhis") & R.arm.str.startswith("gd_r")]
    if len(gd):
        sec = {s: std[f"seen_{s}"].mean() - gd[f"auroc_unseen_{s}"].mean() for s in ORDER}
        out.append(f"- BreakHis secondary AUROC(M_std, leaky ID_seen) − AUROC(M_gd, P_out), all {len(gd)} M_gd runs "
                   f"({int(gd.passes.sum())} pass the gates; M_gd balanced acc on P_out {gd.bacc_unseen.mean():.3f}): "
                   + ", ".join(f"{s} {v:+.3f}" for s, v in sec.items()) + ".")
    br = R[R.dataset == "brain_cheng"]
    if len(br):
        out.append(f"- Brain (optional, M_std only): ID_unseen {int(br.n_unseen.max())} images / {int(br.groups_unseen.max())} "
                   "patients → underpowered, counts toward no bar; (B) cvind arm not run.")
    return out + [""]


def confounds(rep: Path) -> list[str]:
    """Population differences between ID_seen and ID_unseen that the A-gap cannot separate from leakage."""
    out = ["## Population confounds of the A-gap (descriptive)", ""]
    spl = rep / "splits"
    d = pd.read_csv(spl / "isic2019.csv.gz", low_memory=False)
    ct = pd.crosstab(d.role_std, d.source)
    fr = lambda r: ", ".join(f"{c} {ct.loc[r, c] / ct.loc[r].sum():.0%}" for c in ct.columns)  # noqa: E731
    out.append(f"- ISIC 2019 source mix: test_seen {fr('test_seen')}; test_unseen {fr('test_unseen')}; OOD {fr('ood')}. "
               "ID_seen is mostly BCN, ID_unseen mostly HAM / no-id, so the A-gap mixes leakage with a source shift; "
               "dropping no-id images does not remove it (sensitivity table).")
    out.append("- Kermany: ID_unseen = 86 v2-test + 750 v3-test images. The v3 supplement drives the A-gap "
               "(sensitivity table); M_std(v3) also scores its own official v3 test below the v3 DRUSEN OOD set "
               "(descriptive arms), i.e. the v3 test images are shifted from training images independently of group sharing.")
    out.append("- BreakHis: ID_unseen = P_out patients, unseen-patient accuracy ≈ 0.65-0.68 vs ≈ 0.9 on ID_seen "
               "(confound flag); the A-gap includes the patient-shift drop in accuracy.")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="outputs/reports/rigor_pack/medbench")
    ap.add_argument("--gpu-hours", type=float, default=None)
    args = ap.parse_args()
    rep = Path(args.reports)

    R = run_rows(rep)
    mq_cols = ["dataset", "arch", "seed", "arm", "family", "final_train_acc", "final_loss", "epoch1_loss", "G1",
               "G2_set", "G2_value", "G2", "near_chance", "G3_inadequate", "passes"]
    acc_cols = sorted(c for c in R.columns if c.startswith(("acc_", "bacc_")))
    R[mq_cols + acc_cols].to_csv(rep / "model_quality.csv", index=False)

    RK = ranking(R)
    A = arch_cells(R, "A")
    B = arch_cells(R, "B")
    # BreakHis (B) = within-model gap of M_std(r) (L3), core archs
    B = pd.concat([B[B.dataset != "breakhis"], A[(A.dataset == "breakhis") & A.arch.isin(CORE)]], ignore_index=True)

    for ds in ALL_DS:
        d = rep / ds
        a = A[A.dataset == ds]
        if len(a):
            a[["dataset", "arch", "n_runs", "n_unseen", "groups_unseen", "underpowered_gap", "acc_seen", "acc_unseen",
               "confound_unseen_acc_lt_0.8"] + [f"gap_{s}" for s in ORDER]
              + [f"seen_{s}" for s in ORDER] + [f"unseen_{s}" for s in ORDER]].to_csv(d / "A_gap.csv", index=False)
            if "dfit_kNN" in a:
                a[["dataset", "arch", "n_seen_fold_min", "underpowered_fit"] + [f"dfit_{s}" for s in FIT]
                  + [f"dfit_ci_excl0_{s}" for s in FIT]].to_csv(d / "A_fit.csv", index=False)
        b = B[B.dataset == ds]
        if len(b):
            b[["dataset", "arch", "n_runs", "n_unseen", "groups_unseen", "underpowered_gap", "acc_seen", "acc_unseen",
               "confound_unseen_acc_lt_0.8"] + [f"gap_{s}" for s in ORDER]].to_csv(d / "B_gap.csv", index=False)
        ci_cols = ["dataset", "arch", "seed", "arm"] + [c for s in ORDER for c in (f"gap_{s}", f"gap_lo_{s}", f"gap_hi_{s}")]
        ci_cols += [c for s in FIT for c in (f"dfit_{s}", f"dfit_lo_{s}", f"dfit_hi_{s}")]
        r = R[(R.dataset == ds) & R.family.isin(["A", "B"])]
        r[[c for c in ci_cols if c in r]].to_csv(d / "bootstrap_ci.csv", index=False)
        if len(RK):
            RK[RK.dataset == ds].to_csv(d / "ranking.csv", index=False)

    # verdicts
    V = {}
    for ds in ALL_DS:
        ab = a_bar(A, ds) if (A.dataset == ds).any() else {}
        bb = b_bar(B, ds) if (B.dataset == ds).any() else {}
        a_present = [s for s in ORDER if ab.get(s, {}).get("present")]
        a_present_rob = [s for s in ORDER if ab.get(s, {}).get("present_robust")]
        b_present = [s for s in LOGIT if bb.get(s, {}).get("present")]
        b_present_rob = [s for s in LOGIT if bb.get(s, {}).get("present_robust")]
        rk = {}
        for fam in ("A", "B"):
            x = RK[(RK.dataset == ds) & (RK.family == fam)] if len(RK) else RK
            if len(x):
                per_arch = x.groupby("arch").agg(ch=("winner_changed", "mean"), sig=("swap_ci_significant", "mean"))
                n = len(per_arch)
                rk[fam] = {"n_archs": n, "winner_changed_archs": int((per_arch.ch >= 0.5).sum()),
                           "sig_swap_archs": int((per_arch.sig >= 0.5).sum()),
                           "median_tau": float(x.kendall_tau_b.median())}
                rk[fam]["changed_plain"] = rk[fam]["winner_changed_archs"] >= n / 2 or rk[fam]["median_tau"] < 0.5
                rk[fam]["changed_sig"] = rk[fam]["sig_swap_archs"] >= n / 2 or rk[fam]["median_tau"] < 0.5
        a_ds = A[A.dataset == ds]
        fam_d = {}
        if len(a_ds):
            med = {s: float(np.median(a_ds[f"gap_{s}"])) for s in ORDER}
            fam_d = {"feature": float(np.median([med[s] for s in FEAT])),
                     "logit": float(np.median([med[s] for s in LOGIT])), "ReAct": med["ReAct"]}
        n_pass = int(a_ds.shape[0])
        V[ds] = {"A": ab, "B": bb, "a_present": a_present, "a_present_robust": a_present_rob,
                 "b_present": b_present, "b_present_robust": b_present_rob, "ranking": rk, "family": fam_d,
                 "n_pass_archs_A": n_pass,
                 "inconclusive": n_pass < 3,
                 "present": (bool(a_present) or bool(b_present)) and n_pass >= 3,
                 "present_robust": (bool(a_present_rob) or bool(b_present_rob)) and n_pass >= 3}

    tested = [d for d in CORE_DS if not V[d]["inconclusive"]]
    n_present = sum(V[d]["present"] for d in tested)
    n_present_rob = sum(V[d]["present_robust"] for d in tested)

    def gen(n):
        if len(tested) < 3:
            return "dataset-dependent (cap: < 3 core datasets)"
        if n >= 3:
            return "group leakage in standard medical splits inflates post-hoc OOD AUROC across benchmarks"
        if n == 2:
            return "dataset-dependent"
        return "Camelyon/WILDS-specific; medical datasets are negative controls"

    dir_hits = [d for d in tested if V[d]["family"] and V[d]["family"]["feature"] > V[d]["family"]["logit"]]
    react_ok = [d for d in tested if V[d]["family"] and abs(V[d]["family"]["ReAct"]) <= BAR]

    # summary.csv
    S = []
    for ds in ALL_DS:
        v = V[ds]
        row = {"dataset": ds, "verdict": "INCONCLUSIVE" if v["inconclusive"] else ("present" if v["present"] else "absent"),
               "verdict_robust": "present" if v["present_robust"] else "absent",
               "A_present_scores": ";".join(v["a_present"]), "A_present_robust": ";".join(v["a_present_robust"]),
               "B_present_logit_scores": ";".join(v["b_present"]), "B_present_robust": ";".join(v["b_present_robust"]),
               "n_pass_archs_A": v["n_pass_archs_A"]}
        for fam in ("A", "B"):
            if fam in v["ranking"]:
                for k, x in v["ranking"][fam].items():
                    row[f"rank{fam}_{k}"] = x
        for k, x in v["family"].items():
            row[f"median_gap_{k}"] = x
        S.append(row)
    S += [{"dataset": "camelyon17 (earlier)", "verdict": "present"},
          {"dataset": "iwildcam (multibench)", "verdict": "present"},
          {"dataset": "rxrx1 (multibench)", "verdict": "present (A only; weak detectors)"},
          {"dataset": "midog (multibench)", "verdict": "absent by construction"}]
    pd.DataFrame(S).to_csv(rep / "summary.csv", index=False)

    f3 = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"  # noqa: E731
    md = ["# Medbench group-leakage check: summary", "",
          "Precommit `decisions/precommit_group_leakage_medbench_2026-10-06.md` (+ deviations file). Gap = AUROC(OOD vs",
          "seen-group ID) − AUROC(OOD vs unseen-group ID), class-matched; Δ_fit = same-group − group-disjoint 2-fold",
          "scorer-fit AUROC; positive = inflation. Per arch = mean over gate-passing seeds (BreakHis: and repeats).",
          "`*` = every run's 95% cluster-bootstrap CI excludes 0; `u` = underpowered cell (excluded from bars).", ""]
    for ds in ALL_DS:
        a = A[A.dataset == ds]
        if not len(a):
            continue
        md += [f"## {ds}", "", "(A) frozen M_std:", "",
               "| arch | " + " | ".join(f"gap {s}" for s in ORDER) + " | " + " | ".join(f"Δfit {s}" for s in FIT)
               + " | acc seen / unseen |", "|---" * (len(ORDER) + len(FIT) + 2) + "|"]
        for _, c in a.iterrows():
            u = "u" if c.underpowered_gap else ""
            g = " | ".join(f"{f3(c[f'gap_{s}'])}{'*' if c[f'gap_ci_excl0_{s}'] else ''}{u}" for s in ORDER)
            uf = "u" if c.get("underpowered_fit") is True else ""
            fz = " | ".join(f"{f3(c.get(f'dfit_{s}'))}{'*' if c.get(f'dfit_ci_excl0_{s}') is True else ''}{uf}" for s in FIT)
            md.append(f"| {c.arch} | {g} | {fz} | {c.acc_seen:.3f} / {c.acc_unseen:.3f} |")
        md.append("")
        b = B[B.dataset == ds]
        if len(b) and ds != "breakhis":
            md += ["(B) retrained group-disjoint, within-model gap (mean over seeds × folds):", "",
                   "| arch | " + " | ".join(ORDER) + " | acc seen / unseen |", "|---" * (len(ORDER) + 2) + "|"]
            for _, c in b.iterrows():
                u = "u" if c.underpowered_gap else ""
                md.append(f"| {c.arch} | " + " | ".join(f"{f3(c[f'gap_{s}'])}{'*' if c[f'gap_ci_excl0_{s}'] else ''}{u}"
                                                       for s in ORDER) + f" | {c.acc_seen:.3f} / {c.acc_unseen:.3f} |")
            md.append("")
        x = RK[RK.dataset == ds] if len(RK) else RK
        if len(x):
            for fam in ("A", "B"):
                y = x[x.family == fam]
                if len(y):
                    w = y.groupby(["winner_leaky", "winner_disjoint"]).size().sort_values(ascending=False)
                    md.append(f"Ranking ({fam}, {len(y)} runs): " + ", ".join(f"{i[0]}→{i[1]} {n}" for i, n in w.items())
                              + f"; median tau-b {y.kendall_tau_b.median():.2f}; CI-significant swaps {int(y.swap_ci_significant.sum())}")
            md.append("")
    (rep / "summary.md").write_text("\n".join(md) + "\n")

    # README
    rd = ["# Medbench group-leakage check: README", "",
          "Precommit `decisions/precommit_group_leakage_medbench_2026-10-06.md` (commit 0fa52af, DEPOSIT 163151d),",
          "deviations in `decisions/precommit_group_leakage_medbench_2026-10-06.deviations.md`. Bar = 0.02 AUROC.", "",
          "## Bars per dataset", "",
          "| dataset | archs passing gates (A) | A-bar present for (robust) | B-bar logit scores (robust) | ranking A: winner changed / sig swaps / median tau | ranking B | verdict (robust) |",
          "|---|---|---|---|---|---|---|"]
    for ds in ALL_DS:
        v = V[ds]
        ra = v["ranking"].get("A", {})
        rb = v["ranking"].get("B", {})
        fr = lambda r: (f"{r['winner_changed_archs']}/{r['sig_swap_archs']} of {r['n_archs']}, {r['median_tau']:.2f} → "  # noqa: E731
                        f"{'changed' if r['changed_plain'] else 'stable'} ({'changed' if r['changed_sig'] else 'stable'})") if r else "—"
        verdict = "INCONCLUSIVE" if v["inconclusive"] else ("present" if v["present"] else "absent")
        if A[A.dataset == ds].underpowered_gap.all():
            verdict += ", gap underpowered: Δ_fit only"
        rd.append(f"| {ds} | {v['n_pass_archs_A']} | {', '.join(v['a_present']) or 'none'} ({', '.join(v['a_present_robust']) or 'none'}) | "
                  f"{', '.join(v['b_present']) or 'none'} ({', '.join(v['b_present_robust']) or 'none'}) | {fr(ra)} | {fr(rb)} | "
                  f"{verdict} ({'present' if v['present_robust'] else 'absent'}) |")
    rd += ["", "A-bar hit counts (archs with gap > 0.02 or Δ_fit > 0.02 / archs counted):", ""]
    for ds in ALL_DS:
        if V[ds]["A"]:
            rd.append(f"- {ds}: " + ", ".join(f"{s} {V[ds]['A'][s]['hits']}/{V[ds]['A'][s]['counted']}" for s in ORDER))
    rd += ["", "B-bar hit counts (core archs with within-model gap > 0.02 / counted):", ""]
    for ds in ALL_DS:
        if V[ds]["B"]:
            rd.append(f"- {ds}: " + ", ".join(f"{s} {V[ds]['B'][s]['hits']}/{V[ds]['B'][s]['n']}" for s in ORDER))
    rd += ["", "Δ_fit alone (scorer fit on same vs other groups inside the seen set; no seen/unseen population difference),",
           "archs with Δ_fit > 0.02 / counted, median Δ_fit:", ""]
    for ds in ALL_DS:
        a = A[A.dataset == ds]
        if "dfit_kNN" in a:
            a = a[~a.underpowered_fit]
            rd.append(f"- {ds}: " + ", ".join(f"{s} {int((a[f'dfit_{s}'] > BAR).sum())}/{len(a)} ({a[f'dfit_{s}'].median():+.3f})" for s in FIT))
    rd += ["", "## Generalization verdict (locked rule, with Camelyon / multibench rows)", "",
           f"Core medical datasets tested: {len(tested)} ({', '.join(tested)}); leakage present on {n_present} "
           f"(robust bar: {n_present_rob}) → **{gen(n_present)}** (robust: {gen(n_present_rob)}).",
           "Earlier rows: Camelyon17 present; multibench iWildCam present, RxRx1 present (A only), MIDOG absent by construction.", "",
           "## Secondary directional prediction", ""]
    for ds in tested:
        f = V[ds]["family"]
        rd.append(f"- {ds}: median A-gap feature family {f['feature']:+.3f} vs logit family {f['logit']:+.3f}; ReAct {f['ReAct']:+.3f}")
    rd += [f"Feature > logit on {len(dir_hits)}/{len(tested)} (prediction needs >= 3/4): "
           f"{'holds' if len(dir_hits) >= 0.75 * len(tested) else 'fails'}; ReAct |gap| <= 0.02 on {len(react_ok)}/{len(tested)}.", "",
           "## Model-quality gates", ""]
    for ds in ALL_DS:
        r = R[R.dataset == ds]
        fails = r[~r.passes]
        rd.append(f"- {ds}: {int(r.passes.sum())}/{len(r)} runs pass"
                  + (": failing " + "; ".join(f"{x.arch} s{x.seed} {x.arm} (G1 {x.G1}, G2 {x.G2} = {x.G2_value:.3f}, G3 near-chance {x.near_chance or '-'})"
                                               for _, x in fails.iterrows()) if len(fails) else ""))
    rd += ["", "Confound notes (unseen-group accuracy < 0.8):", ""]
    for nm, T in (("A", A), ("B", B)):
        c = T[T["confound_unseen_acc_lt_0.8"]]
        if len(c):
            rd.append(f"- ({nm}) " + ", ".join(f"{x.dataset}/{x.arch} {x.acc_unseen:.3f}" for _, x in c.iterrows()))
    rd += ["", "## Budget", "",
           "Estimate (precommit L6): ≈ 70 GPU-h core + ≈ 3 brain. Actual (sum of medbench GPU job elapsed, smokes included): "
           + (f"{args.gpu_hours:.1f} GPU-h." if args.gpu_hours is not None else "n/a."), "No cut was needed.", "",
           *sensitivity(rep), *extra_arms(R), *confounds(rep),
           "## Aggregation choices (technical; not in the precommit text)", "",
           "- Gates per run; failing runs dropped; an arch counts if >= 1 run passes; arch value = mean over passing runs.",
           "- G2 set: ID_seen of the arm (`test_seen` / `seen`); DermaMNIST-C arm: `c_test`; v3std and BreakHis M_gd: their only ID set.",
           "- G3 on class-matched seen / unseen AUROCs (arms without them: on the unmatched AUROCs).",
           "- Floor: A-gap / B-gap cells use ID_unseen (BreakHis: summed over repeats, the pooled cell of deviation 4);",
           "  Δ_fit cells use the smaller seen-set fold.",
           "- Robust bar: point > 0.02 and every passing run's CI excludes 0.",
           "- Ranking: leaky = class-matched AUROC on ID_seen, disjoint = on ID_unseen; near-chance scores dropped from the",
           "  ranking of that run; a swap is CI-significant if the leaky winner beats the disjoint winner under the leaky",
           "  protocol and the reverse holds under the disjoint protocol (both pair CIs exclude 0); per arch = majority of runs.",
           "- BreakHis (B) = within-model gap of M_std(r), core archs (L3); M_gd(r) reported in model_quality.csv only.", ""]
    (rep / "README.md").write_text("\n".join(rd) + "\n")
    print("\n".join(rd))


if __name__ == "__main__":
    main()
