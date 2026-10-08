#!/usr/bin/env python3
"""R3 item 8 / P2-b report: results/r3/8/p2b/REPORT.md from cells/, cells_ood_clean/ (post hoc) and
results/r3/8/ood_overlap/audit.json. Usage: item8_p2b_report.py [<commit>]

Cells counted in medians: medbench quality gate passed (model_quality.csv G1 / G2 as recorded) and not near-chance,
at ceiling or below chance (P2-a rules on leaky / truth). Robust = median of the per-cell jackknife lower bounds > 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
P2B = REPO / "results/r3/8/p2b"
DS = ("kermany", "isic2019")
ARCH = ("resnet18", "resnet50", "densenet121", "convnext_tiny")
SCORERS = ("mahalanobis_l2", "knn_mean_cosine")
NAME = {"kermany": "Kermany OCT", "isic2019": "ISIC 2019"}


def load(d):
    out = {}
    for ds in DS:
        for a in ARCH:
            for s in (42, 43):
                for f in ("b0", "b1"):
                    p = P2B / d / f"{ds}_{a}_s{s}_{f}.json"
                    out[(ds, a, s, f)] = json.loads(p.read_text()) if p.exists() else None
    return out


def counted(c, sc):
    x = c["scorers"][sc]
    return c["passes_quality"] and not any(x["flags"].values())


def med(C, ds, sc, stat, fold=None):
    v = [c["scorers"][sc][stat] for k, c in C.items() if c and k[0] == ds and (fold is None or k[3] == fold) and counted(c, sc)]
    if not v:
        return None
    return {"n": len(v), "median": float(np.median([x["est"] for x in v])), "median_lo": float(np.median([x["ci95"][0] for x in v]))}


def f4(x):
    return "n/a" if x is None else f"{x:+.4f}"


def main() -> int:
    commit = sys.argv[1] if len(sys.argv) > 1 else "uncommitted"
    C, CC = load("cells"), load("cells_ood_clean")
    audit = json.loads((REPO / "results/r3/8/ood_overlap/audit.json").read_text())
    missing = [k for k, v in C.items() if v is None]
    pr, ident, sec = {}, {}, {}
    for ds in DS:
        for sc in SCORERS:
            for stat in ("scorer_channel", "model_channel"):
                pr[(ds, sc, stat)] = med(C, ds, sc, stat)
            m0 = [med(C, ds, sc, s, "b0") for s in ("scorer_channel", "model_channel")]
            m1 = [med(C, ds, sc, s, "b1") for s in ("scorer_channel", "model_channel")]
            ident[(ds, sc)] = None if None in m0 + m1 else [abs(a["median"] - b["median"]) for a, b in zip(m0, m1)]
            fc = [c["scorers"][sc]["fraction_closed"] for k, c in C.items() if c and k[0] == ds and counted(c, sc)]
            sec[(ds, sc)] = (float(np.median([x for x in fc if x is not None])) if any(x is not None for x in fc) else None,
                             sum(x is None for x in fc), len(fc))

    def holds(ds, stat):
        r = [pr[(ds, sc, stat)] for sc in SCORERS]
        if None in r:
            return "not run (no counted cells)", None
        p = all(x["median"] > 0 for x in r)
        rob = all(x["median_lo"] > 0 for x in r)
        txt = "; ".join(f"{sc} median {x['median']:+.4f} (median lower bound {x['median_lo']:+.4f}, n = {x['n']})" for sc, x in zip(SCORERS, r))
        return f"{'holds' if p else 'fails'} (robust: {'holds' if rob else 'fails'}): {txt}", p

    p1, p1b = holds("kermany", "scorer_channel")
    p2, p2b = holds("kermany", "model_channel")
    i1, i1b = holds("isic2019", "scorer_channel")
    i2, i2b = holds("isic2019", "model_channel")
    p3 = ("not run" if None in (p1b, p2b, i1b, i2b) else
          f"{'holds' if (i1b == p1b and i2b == p2b) else 'fails'} (ISIC scorer channel {'> 0' if i1b else 'not > 0'} for both "
          f"scorers vs Kermany {'> 0' if p1b else 'not > 0'}; model channel ISIC {'> 0' if i2b else 'not > 0'} vs Kermany "
          f"{'> 0' if p2b else 'not > 0'})")
    idt = {}
    for ds in DS:
        v = [ident[(ds, sc)] for sc in SCORERS]
        idt[ds] = ("not run" if None in v else
                   f"{'identified' if all(max(x) <= 0.02 for x in v) else 'not identified'} (per-fold median differences, "
                   + "; ".join(f"{sc}: scorer {x[0]:.4f}, model {x[1]:.4f}" for sc, x in zip(SCORERS, v)) + "; threshold 0.02)")

    L = ["# R3 item 8 / P2-b: scorer and model channels on the medbench (B) arm", "", f"Commit: {commit}", "",
         f"Verdict: Kermany p1 (scorer channel > 0) {p1.split(' ')[0]}, p2 (model channel > 0) {p2.split(' ')[0]}; "
         f"identification {idt['kermany'].split(' (')[0]}.", "",
         "Precommit: results/r3/8/PRECOMMIT.json, P2_b_retrain_second_medical_dataset. Cached medbench (B) features, CPU, "
         "crossfit_ood Track A scorers; jackknife over patients / lesion groups.", ""]
    if missing:
        L += [f"Missing cells (not run): {len(missing)}: " + ", ".join("_".join(map(str, k)) for k in missing), ""]
    L += ["## Predictions", "", f"- p1 (Kermany scorer channel > 0, both scorers): {p1}",
          f"- p2 (Kermany model channel > 0, both scorers): {p2}", f"- p3 (ISIC 2019 same direction): {p3}",
          f"  - ISIC scorer channel: {i1}", f"  - ISIC model channel: {i2}", "",
          "## Identification check", "", f"- Kermany: {idt['kermany']}", f"- ISIC 2019: {idt['isic2019']}", "",
          "## Secondary: fraction of the gap closed", ""]
    for (ds, sc), (m, nex, n) in sec.items():
        L.append(f"- {NAME[ds]}, {sc}: median {f4(m)} over {n - nex} cells with |leaky - truth| >= 0.01 ({nex} of {n} counted cells excluded)")
    L += ["", "## Cells", "", "No earlier CI exists for these channels (new quantities); the jackknife CI is the only CI. "
          "n_groups_fit = training patients / groups of the fold; K = 2 (within-S folds).", "",
          "| dataset | arch | seed | fold | scorer | leaky | truth | F2 | scorer channel [CI] | model channel [CI] | fraction closed | "
          "n_groups_fit | K | d | ID acc seen / unseen | quality | flags | orphan seen dropped |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in C.items():
        if c is None:
            continue
        for sc in SCORERS:
            x = c["scorers"][sc]
            fl = ", ".join(n for n, v in x["flags"].items() if v) or "-"
            L.append(f"| {NAME[k[0]]} | {k[1]} | {k[2]} | {k[3]} | {sc} | {x['leaky']['est']:.4f} | {x['truth']['est']:.4f} | "
                     f"{x['F2']['est']:.4f} | {x['scorer_channel']['est']:+.4f} [{x['scorer_channel']['ci95'][0]:+.4f}, "
                     f"{x['scorer_channel']['ci95'][1]:+.4f}] | {x['model_channel']['est']:+.4f} [{x['model_channel']['ci95'][0]:+.4f}, "
                     f"{x['model_channel']['ci95'][1]:+.4f}] | {f4(x['fraction_closed'])} | {c['n_groups_train']} | {c['K_within']} | "
                     f"{c['d']} | {c['acc_seen']:.4f} / {c['acc_unseen']:.4f} | {'pass' if c['passes_quality'] else 'fail'} | {fl} | "
                     f"{c['n_orphan_seen_dropped']} ({c['n_orphan_seen_groups']} groups) |")

    L += ["", "## Post hoc sensitivity: OOD restricted to patients absent from every ID set (Kermany)", "",
          "Requested after the overlap audit (results/r3/8/ood_overlap/audit.json); not in the precommit. ISIC 2019 shows no "
          "lesion or pixel overlap, so it has no sensitivity run.", "",
          "| arm | OOD images | from patients in an ID set | clean OOD images (patients) | seen-OOD pixel-identical pairs | train-OOD pairs |",
          "|---|---|---|---|---|---|"]
    for f in ("b0", "b1"):
        a = audit[f"kermany_{f}"]
        L.append(f"| Kermany {f} | {a['n_ood_images']} | {a['ood_images_from_any_id_group']} | {a['clean_ood_images']} "
                 f"({a['clean_ood_groups']}) | {a['by_set']['seen']['pixel_identical_pairs']} | {a['by_set']['train']['pixel_identical_pairs']} |")
    L += ["", "| scorer | statistic | median, all OOD (n cells) | median, clean OOD (n cells) | median lower bound, all | median lower bound, clean |",
          "|---|---|---|---|---|---|"]
    for sc in SCORERS:
        for stat in ("leaky", "F2", "truth", "scorer_channel", "model_channel"):
            a, b = med(C, "kermany", sc, stat), med(CC, "kermany", sc, stat)
            fa = lambda x, key: "n/a" if x is None else f"{x[key]:+.4f}"  # noqa: E731
            L.append(f"| {sc} | {stat} | {fa(a, 'median')} ({a['n'] if a else 0}) | {fa(b, 'median')} ({b['n'] if b else 0}) | "
                     f"{fa(a, 'median_lo')} | {fa(b, 'median_lo')} |")
    L += ["", "## Deviations and caveats", "",
          "1. Technical deviation: the package check (leaky and F2 vs crossfit_auroc) allows 1e-9 plus one pair per near-tied "
          "seen / OOD score pair. Kermany b1 kNN failed the plain 1e-9 check by exactly one pair (7.5647508e-09) because "
          "pixel-identical seen / OOD images give tied scores that the two code paths break differently. Estimates unchanged; "
          "package values and tie counts are stored per cell. Cells computed before the fix passed the plain check.",
          f"2. Dataset finding: Kermany OCT has pixel-identical images across classes of the same patient; seen-OOD identical pairs: "
          f"{audit['kermany_b0']['by_set']['seen']['pixel_identical_pairs']} (b0), {audit['kermany_b1']['by_set']['seen']['pixel_identical_pairs']} (b1).",
          "3. Kermany OOD (DRUSEN) shares patients with the ID sets (table above); the precommitted estimand is kept, the "
          "restricted-OOD rerun is post hoc.",
          "4. Orphan seen images (group with no training image after the medbench val carve-out) are dropped, as in item 1; counts per cell.",
          "5. ISIC 2019 has no patient ID; groups are lesions (or single images), so patient-level overlap cannot be checked."]
    (P2B / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
