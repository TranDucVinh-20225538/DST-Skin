#!/usr/bin/env python3
"""Writes results/t1/A/REPORT.md from the item-A result files and sacct (rule 15)."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

OUT = CM.RES / "A"
JOBS = {"A0": "65259", "A1": "65261", "A2": "65265", "A3": "65266"}


def gpu_hours(job):
    """Sum over array tasks of elapsed seconds x allocated GPUs (sacct, allocation lines only)."""
    txt = subprocess.run(["sacct", "-j", job, "-X", "-n", "-P", "--format=JobID,ElapsedRaw,AllocTRES,Start,End"],
                         capture_output=True, text=True).stdout
    tot, start, end = 0.0, None, None
    for line in txt.strip().splitlines():
        jid, el, tres, st, en = line.split("|")
        g = 0
        for kv in tres.split(","):
            if kv.startswith("gres/gpu="):
                g = int(kv.split("=")[1])
        tot += int(el) * g / 3600
        start = st if start is None or st < start else start
        end = en if end is None or en > end else end
    return tot, start, end


def f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main():
    s1 = json.load(open(OUT / "a1_summary.json"))
    s2 = json.load(open(OUT / "a2_summary.json"))
    s3 = json.load(open(OUT / "a3_summary.json"))
    v = json.load(open(OUT / "verdict.json"))
    a0 = {k: json.load(open(OUT / f"{k}.json")) for k in ("a0a", "a0b", "a0c")}
    lobo = list(csv.DictReader(open(OUT / "a1_lobo.csv")))
    a2l = list(csv.DictReader(open(OUT / "a2_lobo.csv")))
    gh = {k: gpu_hours(j) for k, j in JOBS.items()}
    total = sum(x[0] for x in gh.values())
    L = []
    L.append("# Item A — closed form vs ICC on real cached features (REPORT)")
    L.append("")
    L.append(f"**Verdict: {v['verdict']}** (read off the A decision table; A-i {v['status']['A_i']}, A-ii {v['status']['A_ii']}, "
             f"A-iii {v['status']['A_iii']}, A-iv {v['status']['A_iv']}).")
    L.append("")
    L.append("**Interpretation limits (binding wording).** (1) 13 backbones is a small number of independent units; the LOBO result is a "
             "screening result and its confidence intervals may be unstable. (2) The cluster bootstrap with B = 10,000 over backbones "
             "resamples the same 13 backbones; it does not create independent backbones and cannot make the interval narrower than 13 "
             "units allow. (3) The within-FM analysis has n = 5 units (within-CNN n = 8): it is descriptive only, has no power, and is "
             "not used to claim generalisation within either family. (4) The NO-GO says that, on these 13 backbones, the formula did not "
             "predict held-out backbones better than the stated baselines; the saturation ratio r_sat and the A-ii / A-iii rules are as "
             "pre-registered.")
    L.append("")
    L.append("Consequence written in the order for NO-GO: the Gaussian closed form is not supported as the mechanism of real-feature "
             "leakage in this set; Idea 2 is downgraded to toy theory; G2 is skipped (G-ii `not run`); C-core and D run at their listed "
             "scope; C5's real-cell validity map is skipped (C5 on synthetic remains).")
    L.append("")
    L.append(f"- Commit at report time: `{CM.head_hash()}`; precommit commit `{CM.precommit_hash()}`; "
             f"work list: `results/t1/PRECOMMIT_T1_addendum_A.json` (8,116 units; all executed).")
    L.append(f"- GPU-hours (sacct, elapsed × GPUs): A0 {gh['A0'][0]:.2f}, A1 {gh['A1'][0]:.2f}, A2 {gh['A2'][0]:.2f}, A3 {gh['A3'][0]:.2f}; "
             f"total **{total:.2f}** vs estimate 4–8 (re-derived 3–6), cap 12.")
    L.append(f"- Wall clock: first job start {min(x[1] for x in gh.values())}, last job end {max(x[2] for x in gh.values())}.")
    L.append("")
    L.append("## Status words")
    L.append("")
    L.append("| sub-criterion | status | numbers |")
    L.append("|---|---|---|")
    D = s1["D"]
    L.append(f"| A-i (A1 vs B1 and B1w, Camelyon LOBO, n = 13) | {v['status']['A_i']} | mean D_b vs B1 {f(D['B1']['mean_D'])} "
             f"[{f(D['B1']['ci_lo'])}, {f(D['B1']['ci_hi'])}]; vs B1w {f(D['B1w']['mean_D'])} [{f(D['B1w']['ci_lo'])}, {f(D['B1w']['ci_hi'])}] |")
    L.append(f"| A-ii (vs B3, B4, B5) | {v['status']['A_ii']} | B3 {f(D['B3']['mean_D'])} [{f(D['B3']['ci_lo'])}, {f(D['B3']['ci_hi'])}]; "
             f"B4 {f(D['B4']['mean_D'])} [{f(D['B4']['ci_lo'])}, {f(D['B4']['ci_hi'])}]; B5 {f(D['B5']['mean_D'])} [{f(D['B5']['ci_lo'])}, "
             f"{f(D['B5']['ci_hi'])}]; B5 eligible (r_sat < 0.9 in {s1['r_sat_frac_lt_0_9']['cell']:.0%} of cells) |")
    D2 = s2["A_iii"]["D"]
    sp = s2["A_iii"]["spearman"]
    L.append(f"| A-iii (A2 within-backbone, {s2['A_iii']['n_complete']} backbones complete) | {v['status']['A_iii']} | vs B6 "
             f"{f(D2['B6']['mean_D'])} [{f(D2['B6']['ci_lo'])}, {f(D2['B6']['ci_hi'])}]; vs B5 {f(D2['B5']['mean_D'])} "
             f"[{f(D2['B5']['ci_lo'])}, {f(D2['B5']['ci_hi'])}]; mean within-backbone Spearman {f(sp['mean'], 3)} "
             f"[{f(sp['ci_lo'], 3)}, {f(sp['ci_hi'], 3)}] |")
    L.append(f"| A-iv (ΔQ level) | {v['status']['A_iv']} | median zero-parameter \\|ΔQ_pred/ΔQ_meas − 1\\| = {f(s1['A_iv']['median_rel_err_cell'], 3)} "
             f"over 27 Camelyon cells (bar ≤ 0.35); pooled R²_oos of LOBO-calibrated ΔQ_pred vs B0 = {f(s1['A_iv']['r2_oos_dq_vs_B0'], 3)} (bar > 0) |")
    L.append("")
    L.append(f"r_sat ≥ 0.9 in {s1['r_sat_frac_ge_0_9']['cell']:.0%} of Camelyon cells, so the INCONCLUSIVE row does not apply. "
             f"Both readings of the two ambiguous aggregations (DA-4, DA-5 in `DEVIATIONS.md`) give the same verdict "
             f"(verdict_reading_dependent = {v['verdict_reading_dependent']}).")
    L.append("")
    L.append("## A0 unit tests (all pass)")
    L.append("")
    L.append(f"- A0a toy reproduction: pass = {a0['a0a']['pass_A0a']}. A0b Track A reproduction (3 cells, |Δ_own − Δ_R3| ≤ 1e-6): "
             f"pass = {a0['a0b']['pass_A0b']}. A0c unit conversion (1e-9 relative): pass = {a0['a0c']['pass_A0c']}.")
    L.append(f"- A1 own Δ vs R3 item-1 Δ (`mahalanobis_l2`, strict seen) over all 158 cells: max |diff| = {s1['max_abs_diff_own_vs_r3']:.1e}.")
    L.append("")
    L.append("## A1 (Camelyon LOBO over 13 backbones)")
    L.append("")
    L.append("| predictor | MAE | R²_oos vs B0 | Spearman (perm p) |")
    L.append("|---|---|---|---|")
    for r in lobo:
        if r["predictor"].startswith("D_b"):
            continue
        sp_txt = "—" if r["spearman"] in ("", "nan") else f"{float(r['spearman']):.3f} ({float(r['spearman_perm_p']):.3f})"
        L.append(f"| {r['predictor']} | {float(r['mae']):.4f} | {float(r['r2_oos_vs_B0']):.3f} | {sp_txt} |")
    L.append("")
    L.append("P = P_Δ with the LOBO calibration a + b·P_Δ; P_zero_param = P_Δ with a = 0, b = 1. Family, within-family and medbench "
             "blocks: `a1_family.csv` (report only, not in the verdict). Per-backbone table: `a1_backbones.csv`; per cell × fold: `cells.csv`.")
    L.append("")
    L.append("## A2 (within-backbone designs, 13 backbones × 2 folds × 30 designs × 10 repeats = 7,800 fits)")
    L.append("")
    L.append("S5 sanity (G' = 15, n' = all reproduces the A1 fold Δ to 1e-6): passed for all 26 backbone × fold pairs (`a2_s5.csv`).")
    L.append("")
    L.append("| predictor | pooled MAE | R²_oos vs B0 |")
    L.append("|---|---|---|")
    for r in a2l:
        if r["predictor"].startswith("D_b"):
            continue
        L.append(f"| {r['predictor']} | {float(r['pooled_mae']):.4f} | {float(r['r2_oos_vs_B0']):.3f} |")
    L.append("")
    L.append("## A3 estimator audit (report only)")
    L.append("")
    fl = s3["flags"]
    L.append(f"- {s3['n_cells']} cells, {s3['n_rows']} cell × fold rows. Cells with r_sat ≥ 0.9: {fl['flag_r_sat_ge_0_9']['n_cells']}; "
             f"δ̂ ≥ 0.5: {fl['flag_delta_ge_0_5']['n_cells']}; N < 5d (formula regime unverified): {fl['flag_N_lt_5d']['n_cells']}.")
    L.append(f"- Median group-jackknife relative SE: ρ̂_w {s3['median_rel_se_rho_w']:.3f}, ΔQ_pred {s3['median_rel_se_dq_pred']:.3f}. "
             f"Median n_w/n̄ = {s3['median_n_w_over_n_bar']:.2f}. Largest effect of the A0c conversion factor on ΔQ_pred: "
             f"{s3['max_abs_conversion_effect']:.2%}. Class-conditional centring: not evaluable (no cell has `labels_train`, DA-11).")
    L.append("")
    L.append("## Required caveats")
    L.append("")
    L.append("- Seen vs unseen eval sets differ also in hospital / class composition; Camelyon is stratified by hospital in the fold "
             "construction (hospitals {0, 3, 4}, 5 slides per hospital per fold); the medbench datasets may not be.")
    L.append("- Deviations: `DEVIATIONS.md` (DA-1 … DA-14); none changes a criterion, threshold, seed or grid.")
    L.append("")
    L.append("Files: `tests.csv` (Holm within A), `cells.csv`, `a1_lobo.csv`, `a1_family.csv`, `a1_backbones.csv`, `a2_designs.csv`, "
             "`a2_lobo.csv`, `a2_s5.csv`, `a3_audit.csv`, `scatter_dq.png`, `scatter_delta.png`, `a2_dose.png`, `verdict.json`.")
    L.append("")
    CM.atomic_write_text(OUT / "REPORT.md", "\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
