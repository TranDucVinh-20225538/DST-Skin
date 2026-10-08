#!/usr/bin/env python3
"""R3 item 2 report: results/r3/2/REPORT.md plus coverage_{main,d768,d2560}.md / .json.

Reads ~/r3work/item2/{main,realdim_768,realdim_2560}/ through the unchanged v3 analyzer
(analyze_coverage_paper2fold.py) and results/r3/1/paper_ci_v2.csv (item-1 real Δ, seen = strict; run item1_report.py
first) for the "coverage not verified at this Δ" marks. Usage: item2_report.py [<commit>]
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CF = Path.home() / "handoff/crossfit-ood_v3/scripts"
W = Path.home() / "r3work/item2"
OUT = REPO / "results/r3/2"
sys.path.insert(0, str(CF))
import analyze_coverage_paper2fold as A  # noqa: E402

RUNS = {"main": "main", "d768": "realdim_768", "d2560": "realdim_2560"}
TH = A.THRESHOLD


def main() -> int:
    commit = sys.argv[1] if len(sys.argv) > 1 else "uncommitted"
    OUT.mkdir(parents=True, exist_ok=True)
    S = {}
    for k, d in RUNS.items():
        p = W / d
        if not (p / "cover.jsonl").exists():
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sys.argv = ["x", str(p)]
            A.main()
        (OUT / f"coverage_{k}.md").write_text(buf.getvalue())
        _, cells = A.summarize(str(p))
        json.dump(cells, open(OUT / f"coverage_{k}.json", "w"), indent=1)
        S[k] = cells
    M = S["main"]
    jk = [(c["design"], c["scorer"], c["level"], c["methods"]["jackknife"]["coverage"], c["methods"]["jackknife"]["n"])
          for c in M]
    bad = [x for x in jk if x[3] < TH]
    nmin = min(x[4] for x in jk)
    dec = ("jackknife coverage >= 0.93 in every cell for every scorer: the jackknife CI is used in the paper beside the old CI"
           if not bad else f"{len(bad)} cell(s) with jackknife coverage < 0.93: reported, stop (no method change this round)")
    L = ["# R3 item 2: coverage of the paper_2fold jackknife CI on designs read from the real splits", "", f"Commit: {commit}", "",
         f"Verdict: {dec}.", "",
         "Design: crossfit-ood v3 coverage_study_paper2fold.py (ICC grid, decision rule, defaults unchanged) on the real "
         "per-group train sizes, ID-eval sizes and OOD sizes of every item-1 dataset (scripts/r3/item2_coverage_real.py). "
         "Scorers: Track A definitions (mahalanobis_l2, knn_mean_cosine); seen = strict only.", "",
         f"Datasets per cell: min {nmin} (target 100).", "",
         "## Real designs", "", "```"]
    L += subprocess.run([sys.executable, str(REPO / "scripts/r3/item2_coverage_real.py"), "summary"], capture_output=True,
                        text=True, check=True).stdout.strip().splitlines()
    L += ["```", "", "## Pre-registered decision (jackknife, 0.93)", "",
          f"Cells: {len(jk)}; jackknife coverage min = {min(x[3] for x in jk):.2f}; cells < 0.93: {len(bad)}.", ""]
    L += [f"- {x[0]} / {x[1]} / {x[2]}: {x[3]:.2f} (n = {x[4]})" for x in bad]
    L += ["", "## Coverage per cell (d = 128, v3 default)", "",
          "| design | scorer | level | true Δ | n | jackknife coverage ± MC SE | old bootstrap coverage ± MC SE | jackknife width | "
          "P(lo > 0) | P(lo > 0.02) |", "|---|---|---|---|---|---|---|---|---|---|"]
    def cov(v):
        return "n/a" if v is None else f"{v['coverage']:.2f} ± {v['mcse']:.2f}"

    for c in M:
        j = c["methods"]["jackknife"]
        L.append(f"| {c['design']} | {c['scorer']} | {c['level']} | {c['true_delta']:+.4f} | {j['n']} | {cov(j)} | "
                 f"{cov(c['methods'].get('bootstrap'))} | {j['width']:.4f} | {j['p_lo_gt_0']:.2f} | {j['p_lo_gt_002']:.2f} |")
    L += ["", "## Power at true Δ ≈ 0.03 (level d03; jackknife)", "",
          "| design | scorer | true Δ | P(lo > 0) | P(lo > 0.02) | n |", "|---|---|---|---|---|---|"]
    for c in M:
        if c["level"] == "d03":
            j = c["methods"]["jackknife"]
            L.append(f"| {c['design']} | {c['scorer']} | {c['true_delta']:+.4f} | {j['p_lo_gt_0']:.2f} | {j['p_lo_gt_002']:.2f} | {j['n']} |")
    miss = sorted({(c["design"], c["scorer"]) for c in M} - {(c["design"], c["scorer"]) for c in M if c["level"] == "d03"})
    if miss:
        L.append("")
        L.append("Level d03 not reached (no power row): " + ", ".join(f"{a} / {b}" for a, b in miss))
    L += ["", "## Real dimension check (Camelyon design)", "",
          "ICC levels from the d = 128 calibration; true Δ re-estimated at each d.", "",
          "| scorer | level | d | true Δ | n | jackknife coverage ± MC SE | P(lo > 0) |", "|---|---|---|---|---|---|---|"]
    for k, dd in (("main", 128), ("d768", 768), ("d2560", 2560)):
        for c in S.get(k, []):
            if c["design"] == "camelyon":
                j = c["methods"]["jackknife"]
                L.append(f"| {c['scorer']} | {c['level']} | {dd} | {c['true_delta']:+.4f} | {j['n']} | {j['coverage']:.2f} ± {j['mcse']:.2f} | {j['p_lo_gt_0']:.2f} |")
    for k in ("d768", "d2560"):
        if k not in S:
            L.append(f"\n{k}: not run.")
        else:
            b2 = [c for c in S[k] if c["methods"]["jackknife"]["coverage"] < TH]
            nn = min(c["methods"]["jackknife"]["n"] for c in S[k])
            L.append(f"\n{k}: {len(S[k])} cells, min n {nn}; jackknife coverage < 0.93 in {len(b2)}"
                     + (": " + ", ".join(f"{c['scorer']}/{c['level']} {c['methods']['jackknife']['coverage']:.2f}" for c in b2) if b2 else "") + ".")
    L += ["", "## Real cells beyond the simulated Δ range", "",
          "Real Δ (item 1, seen = strict) above the largest simulated true Δ of its design and scorer: coverage not verified at this Δ.", ""]
    p1 = REPO / "results/r3/1/paper_ci_v2.csv"
    if p1.exists():
        T = pd.read_csv(p1)
        T = T[T.seen == "strict"]
        mx = {}
        for c in M:
            k = (c["design"], c["scorer"])
            mx[k] = max(mx.get(k, -1), c["true_delta"])
        L += ["| design | scorer | max simulated true Δ | real rows | rows above (coverage not verified at this Δ) |", "|---|---|---|---|---|"]
        flag = []
        for (ds, sc), v in sorted(mx.items()):
            t = T[(T.dataset == ds) & (T.scorer == sc)]
            a = t[t.delta > v]
            flag += [f"{r.cell} / {sc}: Δ {r.delta:.4f}" for r in a.itertuples()]
            L.append(f"| {ds} | {sc} | {v:+.4f} | {len(t)} | {len(a)} |")
        if flag:
            L += ["", "Coverage not verified at this Δ:", ""] + [f"- {x}" for x in flag]
    else:
        L.append("Item 1 table not available: not run.")
    L += ["", "## Caveats", "",
          "1. The local-v3 vs HPC comparison is dropped: the local run used the package scorers, this run the Track A "
          "scorers, so the cells are not comparable.",
          "2. Coverage is verified for seen = strict only; item-1 rows with seen = tracka are not covered by this study.",
          "3. ViM is not studied: Track A ViM is not reproduced (item 1 match table).",
          "4. v3 scaling kept: Camelyon sizes 1/30; Kermany train 1/2 and OOD 1/8."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
