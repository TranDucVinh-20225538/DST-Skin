"""Multibench group-leakage report: summary.csv/.md and README.md (bars as in
decisions/precommit_group_leakage_multibench_2026-10-06.md)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ORDER = ["MSP", "Energy", "ELogitNorm", "ViM", "ReAct", "Mahalanobis", "kNN"]
FIT = ["Mahalanobis", "kNN", "ViM", "ReAct"]
A_ARCHS = {"iwildcam": ["resnet18", "resnet50", "densenet121", "convnext_tiny", "mobilenet_v3_large",
                        "regnet_y_3_2gf", "effb3", "efficientnet_v2_s"],
           "rxrx1": ["resnet18", "resnet50", "densenet121", "convnext_tiny"]}
B_ARCHS = ["resnet18", "resnet50", "densenet121", "convnext_tiny"]
SEEDS, FOLDS, BAR = [42, 43], [0, 1], 0.02


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def a_rows(rep: Path, ds: str) -> list[dict]:
    rows = []
    for a in A_ARCHS[ds]:
        j = load(rep / ds / f"A_{a}.json")
        std = np.array([j["standard"][s] for s in ORDER])
        dis = np.array([j["disjoint_2fold"][s] for s in ORDER])
        tau = kendalltau(std, dis).statistic
        r = {"dataset": ds, "arch": a, "winner_standard": ORDER[int(std.argmax())],
             "winner_disjoint": ORDER[int(dis.argmax())], "kendall_tau": tau,
             "n_id_fold0": j["n_id_fold"][0], "n_id_fold1": j["n_id_fold"][1], "n_ood": j["n_ood"]}
        for s in ORDER:
            r[f"std_{s}"] = j["standard"][s]
            r[f"dis_{s}"] = j["disjoint_2fold"][s]
        for s in FIT:
            r[f"same_{s}"] = j["same_2fold"][s]
            r[f"delta_{s}"] = j["delta"][s]
        rows.append(r)
    return rows


def b_rows(rep: Path, ds: str) -> list[dict]:
    rows = []
    for a in B_ARCHS:
        for s in SEEDS:
            for f in FOLDS:
                t = load(rep / ds / f"retrain_{a}_s{s}_f{f}.json")
                c = load(rep / ds / f"scores_{a}_s{s}_f{f}.json")
                r = {"dataset": ds, "arch": a, "seed": s, "fold": f, "n_train": t["n_train"],
                     "n_id_seen": t["n_id_seen"], "n_id_unseen": t["n_id_unseen"],
                     "acc_id_seen": t["acc_id_seen"], "acc_id_unseen": t["acc_id_unseen"],
                     "acc_ood": t["acc_ood"], "gap_msp": t["gap_msp"], "gap_energy": t["gap_energy"],
                     "train_seconds": t["train_seconds"]}
                for sc in ORDER:
                    r[f"unseen_{sc}"] = c[f"auroc_{sc}_unseen"]
                    r[f"seen_{sc}"] = c[f"auroc_{sc}_seen"]
                unseen = np.array([r[f"unseen_{sc}"] for sc in ORDER])
                r["winner_same_protocol"] = ORDER[int(unseen.argmax())]
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="outputs/reports/rigor_pack/multibench")
    ap.add_argument("--gpu-hours", type=float, default=None)
    args = ap.parse_args()
    rep = Path(args.reports)

    A = pd.DataFrame([r for ds in A_ARCHS for r in a_rows(rep, ds)])
    B = pd.DataFrame([r for ds in A_ARCHS for r in b_rows(rep, ds)])
    A.to_csv(rep / "A_all.csv", index=False)
    B.to_csv(rep / "B_all.csv", index=False)

    # per arch: fold mean, then mean +- SD over seeds
    Bf = B.groupby(["dataset", "arch", "seed"]).mean(numeric_only=True).reset_index()
    Ba = Bf.groupby(["dataset", "arch"]).agg(
        gap_msp=("gap_msp", "mean"), gap_msp_sd=("gap_msp", "std"),
        gap_energy=("gap_energy", "mean"), gap_energy_sd=("gap_energy", "std"),
        acc_id_seen=("acc_id_seen", "mean"), acc_id_unseen=("acc_id_unseen", "mean"),
        acc_ood=("acc_ood", "mean")).reset_index()

    summ = []
    for ds in A_ARCHS:
        for _, r in A[A.dataset == ds].iterrows():
            row = {"dataset": ds, "arch": r.arch, "winner_standard": r.winner_standard,
                   "winner_disjoint": r.winner_disjoint, "winner_changed": r.winner_standard != r.winner_disjoint,
                   "kendall_tau": r.kendall_tau}
            for s in FIT:
                row[f"delta_{s}"] = r[f"delta_{s}"]
            b = Ba[(Ba.dataset == ds) & (Ba.arch == r.arch)]
            if len(b):
                b = b.iloc[0]
                for k in ["gap_msp", "gap_msp_sd", "gap_energy", "gap_energy_sd", "acc_id_seen",
                          "acc_id_unseen", "acc_ood"]:
                    row[k] = b[k]
                row["confound_unseen_acc_lt_0.8"] = bool(b.acc_id_unseen < 0.8)
            summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(rep / "summary.csv", index=False)

    verdict, lines = {}, []
    for ds in A_ARCHS:
        a = A[A.dataset == ds]
        n_a = len(a)
        a_pass = {s: int((-a[f"delta_{s}"] > BAR).sum()) for s in FIT}
        a_present = [s for s in FIT if a_pass[s] >= n_a / 2]
        b = Ba[Ba.dataset == ds]
        b_msp, b_en = int((b.gap_msp > BAR).sum()), int((b.gap_energy > BAR).sum())
        b_either = int(((b.gap_msp > BAR) | (b.gap_energy > BAR)).sum())
        b_pass = b_msp >= 3 or b_en >= 3
        verdict[ds] = bool(a_present) or b_pass
        n_change = int((a.winner_standard != a.winner_disjoint).sum())
        lines.append((ds, n_a, a_pass, a_present, b_msp, b_en, b_either, b_pass, n_change,
                      float(a.kendall_tau.median()), b))

    n_present = sum(verdict.values())  # MIDOG counts as absent by construction
    if n_present >= 2:
        claim = "group-level leakage in WILDS-style ID splits (present on RxRx1 and iWildCam; MIDOG absent by construction)"
    elif n_present == 1:
        only = [d for d, v in verdict.items() if v][0]
        claim = f"claim names Camelyon17 and {only} only; the others are negative controls; no general claim"
    else:
        claim = "claim stays Camelyon-specific; MIDOG, RxRx1 and iWildCam are negative controls"

    fmt = lambda x: f"{x:+.3f}"
    md = ["# Multibench group-leakage check: summary", "",
          "Precommit `decisions/precommit_group_leakage_multibench_2026-10-06.md`. (A): Δ = disjoint − same-group",
          "2-fold scorer-fit AUROC (negative = inflation from group sharing), seed 42. (B): within-model gap",
          "= AUROC(seen-group ID) − AUROC(unseen-group ID), fold mean, mean ± SD over seeds 42/43. (C): winner",
          "of the 7 scores under standard vs group-disjoint protocol, Kendall tau-b.", ""]
    for ds in A_ARCHS:
        md += [f"## {ds}", "", "| arch | ΔMaha | ΔkNN | ΔViM | ΔReAct | gap MSP | gap Energy | acc seen / unseen | winner std → disjoint | tau |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in S[S.dataset == ds].iterrows():
            g = (f"{r.gap_msp:+.3f} ± {r.gap_msp_sd:.3f} | {r.gap_energy:+.3f} ± {r.gap_energy_sd:.3f} | "
                 f"{r.acc_id_seen:.3f} / {r.acc_id_unseen:.3f}") if pd.notna(r.get("gap_msp")) else "— | — | —"
            md.append(f"| {r.arch} | {fmt(r.delta_Mahalanobis)} | {fmt(r.delta_kNN)} | {fmt(r.delta_ViM)} | "
                      f"{fmt(r.delta_ReAct)} | {g} | {r.winner_standard} → {r.winner_disjoint} | {r.kendall_tau:.2f} |")
        md.append("")
        sp = B[B.dataset == ds].winner_same_protocol.value_counts()
        md += ["Same-protocol winner on retrained models (unseen-group ID, 16 runs): "
               + ", ".join(f"{k} {v}" for k, v in sp.items()), ""]
        std_tab = A[A.dataset == ds][["arch"] + [f"std_{s}" for s in ORDER]]
        md += ["Standard-protocol AUROC (7 scores):", "", "| arch | " + " | ".join(ORDER) + " |",
               "|---" * (len(ORDER) + 1) + "|"]
        for _, r in std_tab.iterrows():
            md.append(f"| {r.arch} | " + " | ".join(f"{r[f'std_{s}']:.3f}" for s in ORDER) + " |")
        md.append("")
    (rep / "summary.md").write_text("\n".join(md) + "\n")

    rd = ["# Multibench group-leakage check: README", "",
          "Precommit: `decisions/precommit_group_leakage_multibench_2026-10-06.md` (commit 909f710), committed",
          "before any number. Bar = 0.02 AUROC throughout. Files: `summary.md/.csv` (per arch), `A_all.csv`,",
          "`B_all.csv` (per run), per-model JSON in `iwildcam/`, `rxrx1/`; inventory in `inventory.md`.", "",
          "## Pass / fail per bar", "",
          "| dataset | (A) archs with loss > 0.02: Maha / kNN / ViM / ReAct | (A) present for | (B) archs gap > 0.02: MSP / Energy (either) | (B) | (C) winner changed | median tau | verdict |",
          "|---|---|---|---|---|---|---|---|",
          "| MIDOG | not run (no seen-group ID split exists) | — | not run | — | — | — | absent by construction |"]
    for ds, n_a, a_pass, a_present, b_msp, b_en, b_either, b_pass, n_change, tau, b in lines:
        rd.append(f"| {ds} | {a_pass['Mahalanobis']} / {a_pass['kNN']} / {a_pass['ViM']} / {a_pass['ReAct']} of {n_a} | "
                  f"{', '.join(a_present) or 'none'} | {b_msp} / {b_en} ({b_either}) of 4 | "
                  f"{'PASS' if b_pass else 'fail'} | {n_change} of {n_a} | {tau:.2f} | "
                  f"{'present' if verdict[ds] else 'absent'} |")
    rd += ["", "(A) bar: loss (−Δ) > 0.02 on >= half of the archs. (B) bar: gap > 0.02 (mean over seeds) on",
           ">= 3 of 4 archs for MSP or for Energy (the per-arch \"MSP or Energy\" count is shown in brackets).", "",
           "## Generalization verdict (locked rule)", "",
           f"Leakage present on {n_present} of 3 datasets (MIDOG counts as absent by construction) → **{claim}**.", "",
           "## Caveats (descriptive, not used to change any bar)", ""]
    for ds, *_ , b in lines:
        conf = b[b.acc_id_unseen < 0.8].arch.tolist()
        rd.append(f"- {ds}: unseen-group accuracy < 0.8 (confound flag) for {', '.join(conf) or 'no arch'}.")
    rx = A[A.dataset == "rxrx1"]
    rd.append(f"- rxrx1: standard-protocol AUROCs range {rx[[f'std_{s}' for s in ORDER]].values.min():.2f}-"
              f"{rx[[f'std_{s}' for s in ORDER]].values.max():.2f}, i.e. detectors are near chance; leakage on RxRx1 "
              "is measured on weak detectors (1,139-class models after 10 epochs).")
    rd += ["", "## Budget", "",
           "Estimate (precommit): 15-22 GPU-h. Actual (sum of GPU job elapsed time, extract + base + retrain): "
           + (f"{args.gpu_hours:.1f} GPU-h." if args.gpu_hours is not None else "n/a."),
           "No seed dropped (projection stayed under 40 GPU-h).", "",
           "## Skipped", "",
           "- MIDOG: (A)/(B) not run; standard ID split is case-disjoint from train with a single ID scanner, so",
           "  no seen-group ID set exists (recorded in the precommit; no substitute).",
           "- Nothing else skipped; RxRx1 was downloaded (WILDS v1.0, public) and used.", ""]
    (rep / "README.md").write_text("\n".join(rd) + "\n")
    print("\n".join(rd))


if __name__ == "__main__":
    main()
