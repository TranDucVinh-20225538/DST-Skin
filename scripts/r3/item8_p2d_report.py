#!/usr/bin/env python3
"""R3 item 8 / P2-d report: results/r3/8/p2d/REPORT.md from pilot_ood.json, train/, cells/ and the Phase 0 JSONs.
Usage: item8_p2d_report.py [<commit>]

Backbone-level aggregation (fixed before any cell was computed, as P2-a): CNN backbone = mean of the two seeds' Delta_fit;
robust (> 0) = every seed's jackknife CI > 0; a backbone is excluded from the bars if any of its cells is near-chance,
at ceiling or below chance; for the ordering's robust versions a CNN backbone's Kvasir lower bound = min over seeds of the
jackknife lower bound and its brain upper bound = max over seeds of the upper bound.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
P2D, P2A = REPO / "results/r3/8/p2d", REPO / "results/r3/8/p2a"
BACKBONES = {"resnet50": ["resnet50_s42", "resnet50_s43"], "convnext_tiny": ["convnext_tiny_s42", "convnext_tiny_s43"],
             "dinov2_vitb14": ["dinov2_vitb14"], "biomedclip": ["biomedclip"]}
FMS = ("dinov2_vitb14", "biomedclip")
SCORERS = ("mahalanobis_l2", "knn_mean_cosine")
DV = (("kvasir", "lit"), ("kvasir", "seg"), ("brain", "rec"))
NAME = {("kvasir", "lit"): "Kvasir-Capsule K_lit", ("kvasir", "seg"): "Kvasir-Capsule K_seg", ("brain", "rec"): "brain MRI"}


def j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.4f}"


def ci(c):
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def gate_rows(ds, v):
    rows = {}
    for bb, runs in BACKBONES.items():
        for r in runs:
            if bb in FMS:
                t = j(P2D / "train" / f"{ds}_{v}_{r}_probe.json")
                rows[r] = None if t is None else {
                    "kind": "probe", "auroc_seen": t["macro_auroc"]["seen"], "auroc_unseen": t["macro_auroc"]["unseen"],
                    "acc_seen": t["acc"]["seen"], "loss_cond": None, "pre": t["macro_auroc"]["seen"] >= 0.70,
                    "post": t["macro_auroc"]["seen"] >= 0.70}
            else:
                t = j(P2D / "train" / f"{ds}_{v}_{r}.json")
                if t is None:
                    rows[r] = None
                    continue
                lc = t["final_loss"] <= 0.5 * t["epoch1_loss"]
                a = t["id_macro_auroc"]["seen"]
                rows[r] = {"kind": "cnn", "auroc_seen": a, "auroc_unseen": t["id_macro_auroc"]["unseen"],
                           "acc_seen": t["id_acc"]["seen"], "loss_cond": lc, "loss": (t["epoch1_loss"], t["final_loss"]),
                           "pre": bool(lc and a >= 0.75), "post": a >= 0.75, "epochs": t["epochs"], "best_epoch": t["best_epoch"]}
    bb = {g: {b: all(rows.get(r) and rows[r][g] for r in runs) for b, runs in BACKBONES.items()} for g in ("pre", "post")}
    return rows, bb


def cell(ds, v, r, sc):
    return j(P2D / "cells" / f"{ds}_{v}_{r}_{sc}.json")


def bb_stat(ds, v, bb, sc, stat):
    cs = [cell(ds, v, r, sc) for r in BACKBONES[bb]]
    if any(c is None for c in cs):
        return None
    s = [c["stats"][stat] for c in cs]
    return {"est": float(np.mean([x["est"] for x in s])), "lo": float(min(x["jk_ci95"][0] for x in s)),
            "hi": float(max(x["jk_ci95"][1] for x in s)), "robust": all(x["jk_ci95"][0] > 0 for x in s),
            "excluded": any(c["flags"]["near_chance"] or c["flags"]["ceiling"] or c["flags"]["below_chance"] for c in cs)}


def counted(ds, v, sc, stat, passing):
    d = {bb: bb_stat(ds, v, bb, sc, stat) for bb in passing}
    if any(x is None for x in d.values()):
        return None
    return {bb: x for bb, x in d.items() if not x["excluded"]}


def bar(ds, v, sc, stat, passing, thr=0.05):
    if len(passing) < 3:
        return f"INCONCLUSIVE ({len(passing)} gate-passing backbone(s) < 3)"
    c = counted(ds, v, sc, stat, passing)
    if c is None:
        return "not run (cells missing)"
    hits = [b for b, x in c.items() if x["est"] > thr]
    rob = [b for b in hits if c[b]["robust"]]
    need = len(passing) / 2
    return (f"{'holds' if len(hits) >= need else 'fails'} ({len(hits)}/{len(passing)} gate-passing backbones with "
            f"Delta_fit > {thr}; robust: {'holds' if len(rob) >= need else 'fails'}, {len(rob)} with jackknife CI > 0)")


def order(sc, pk, pb, kstat="delta_fit_gap"):
    if len(pk) < 3 or len(pb) < 3:
        return "INCONCLUSIVE (< 3 gate-passing backbones in a dataset)"
    k, b = counted("kvasir", "seg", sc, kstat, pk), counted("brain", "rec", sc, "delta_fit", pb)
    if k is None or b is None:
        return "not run (cells missing)"
    if not k or not b:
        return "n/a (no counted backbone after exclusions)"
    mk, mb = np.median([x["est"] for x in k.values()]), np.median([x["est"] for x in b.values()])
    lk, ub = np.median([x["lo"] for x in k.values()]), np.median([x["hi"] for x in b.values()])
    return (f"point: {'holds' if mk > mb else 'fails'} (Kvasir K_seg gap median {mk:+.4f} vs brain median {mb:+.4f}); "
            f"robust: {'holds' if lk > ub else 'fails'} (median Kvasir lower bound {lk:+.4f} vs median brain upper bound {ub:+.4f})")


def main() -> int:
    commit = sys.argv[1] if len(sys.argv) > 1 else "uncommitted"
    pilot = j(P2D / "pilot_ood.json")
    kp, bp = j(P2D / "kvasir_phase0.json"), j(P2D / "brain_phase0.json")
    G = {dv: gate_rows(*dv) for dv in DV}
    P = {dv: [b for b, ok in G[dv][1]["pre"].items() if ok] for dv in DV}
    Ppost = {dv: [b for b, ok in G[dv][1]["post"].items() if ok] for dv in DV}
    kc, bc = pilot["kvasir_capsule"]["primary"], pilot["brain"]["primary"]
    pr = {}
    for sc, tag in (("mahalanobis_l2", ""), ("knn_mean_cosine", "_knn")):
        pr[f"p_order{tag}"] = order(sc, P[("kvasir", "seg")], P[("brain", "rec")])
        pr[f"p1_kvasir{tag}"] = bar("kvasir", "seg", sc, "delta_fit_gap", P[("kvasir", "seg")])
        pr[f"p2_brain{tag}"] = bar("brain", "rec", sc, "delta_fit", P[("brain", "rec")])
    pk = P[("kvasir", "seg")]
    if len(pk) < 3:
        pr["p_gap_knn"] = "INCONCLUSIVE (< 3 gate-passing backbones)"
    else:
        a, b, e = (counted("kvasir", "seg", "knn_mean_cosine", s, pk) for s in ("delta_fit", "delta_fit_gap", "gap_effect"))
        if a is None:
            pr["p_gap_knn"] = "not run (cells missing)"
        elif not a:
            pr["p_gap_knn"] = "n/a (no counted backbone)"
        else:
            ma, mb, le = (np.median([x["est"] for x in a.values()]), np.median([x["est"] for x in b.values()]),
                          np.median([x["lo"] for x in e.values()]))
            pr["p_gap_knn"] = (f"point: {'holds' if ma > mb else 'fails'} (no-gap median {ma:+.4f} vs gap median {mb:+.4f}); "
                               f"robust: {'holds' if le > 0 else 'fails'} (median lower bound of the paired difference {le:+.4f})")

    L = ["# R3 item 8 / P2-d: Delta_fit on Kvasir-Capsule and brain MRI", "", f"Commit: {commit}", "",
         f"Verdict (PRIMARY p_order, mahalanobis_l2): {pr['p_order']}.", "",
         "Precommit: results/r3/8/p2d/PRECOMMIT.json (FINAL). Pilot plan: results/r3/8/p2d/pilot_plan.md.", "",
         "## Pilot (ceiling, frozen DINOv2-B, no Delta)", "",
         "| dataset | candidate | Maha AUROC seen | Maha AUROC unseen | kNN seen | kNN unseen | skipped (> 0.97) |", "|---|---|---|---|---|---|---|"]
    import csv
    rows = list(csv.DictReader(open(P2D / "pilot_ood.csv")))
    for ds, info in (("kvasir_capsule", pilot["kvasir_capsule"]), ("brain", pilot["brain"])):
        for c, ci_ in info["candidates"].items():
            v = {(r["scorer"], r["id"]): float(r["auroc"]) for r in rows if r["dataset"] == ds and r["candidate"] == c}
            L.append(f"| {ds} | {c} | {v[('mahalanobis_l2', 'seen')]:.4f} | {v[('mahalanobis_l2', 'unseen')]:.4f} | "
                     f"{v[('knn_mean_cosine', 'seen')]:.4f} | {v[('knn_mean_cosine', 'unseen')]:.4f} | {'yes' if ci_['skipped'] else 'no'} |")
    L += ["", f"Primary: Kvasir-Capsule {kc} ({pilot['kvasir_capsule']['rule_applied']}); brain {bc} "
          f"({pilot['brain']['rule_applied']}). Only the primary candidate is trained and scored (the PRECOMMIT GPU estimate "
          "covers one candidate per dataset); secondary candidates are not run. The pilot's seen - unseen differences are "
          "disclosed in the table and not interpreted.", ""]
    L += ["## Gates", "",
          f"- G_audit_sanity: Kvasir-Capsule three-way parse agrees ({kp['three_way_parse']['archive_rsplit_equals_regex']}, "
          f"split vs archive videos {kp['three_way_parse']['video_equal_split_vs_archive']}); brain counts match the README: "
          f"{bp['G_audit_sanity']}.",
          f"- G_leak: Kvasir-Capsule K_lit {kp['candidates'][kc]['G_leak_K_lit'] if kc else 'n/a'}; brain "
          f"{bp['candidates'][bc]['G_leak'] if bc else 'n/a'} (share of seen candidates with a fit image in the group).",
          "- G_competence (preregistered: CNN final train loss <= 50% of epoch-1 loss and seen macro one-vs-rest AUROC >= 0.75; "
          "FM probe seen macro AUROC >= 0.70). Post hoc variant (AUROC only), as in P2-a, shown beside it.", "",
          "| dataset / variant | run | kind | ID macro AUROC seen | unseen | ID accuracy seen | loss epoch 1 -> final | preregistered | post hoc |",
          "|---|---|---|---|---|---|---|---|---|"]
    for dv in DV:
        for r, x in G[dv][0].items():
            if x is None:
                L.append(f"| {NAME[dv]} | {r} | not run | | | | | | |")
                continue
            ls = f"{x['loss'][0]:.4f} -> {x['loss'][1]:.4f}" if x["kind"] == "cnn" else "n/a"
            L.append(f"| {NAME[dv]} | {r} | {x['kind']} | {x['auroc_seen']:.4f} | {x['auroc_unseen']:.4f} | {x['acc_seen']:.4f} | "
                     f"{ls} | {'pass' if x['pre'] else 'fail'} | {'pass' if x['post'] else 'fail'} |")
    L += ["", "Gate-passing backbones (preregistered): " + "; ".join(f"{NAME[dv]}: {', '.join(P[dv]) or 'none'}" for dv in DV) + ".",
          "Post hoc (AUROC only): " + "; ".join(f"{NAME[dv]}: {', '.join(Ppost[dv]) or 'none'}" for dv in DV) + ".", ""]

    L += ["## Predictions (preregistered gate)", ""] + [f"- {k}: {v}" for k, v in pr.items()] + [""]
    L += ["## Verdict rule (per dataset)", "",
          f"- Kvasir-Capsule (K_seg with gap), mahalanobis_l2: {bar('kvasir', 'seg', 'mahalanobis_l2', 'delta_fit_gap', P[('kvasir', 'seg')])}",
          f"- brain MRI, mahalanobis_l2: {bar('brain', 'rec', 'mahalanobis_l2', 'delta_fit', P[('brain', 'rec')])}", ""]

    L += ["## Cells", "", "Jackknife (new, primary) and fixed-score cluster bootstrap (old, reference only) 95% CIs side by side. "
          "n_groups_fit = fit groups per F2 fold; K = 2.", "",
          "| dataset / variant | run | scorer | statistic | estimate | jackknife CI | bootstrap CI (ref) | n_groups_fit | K | d | ID acc seen | flags |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for dv in DV:
        for bb, runs in BACKBONES.items():
            for r in runs:
                for sc in SCORERS:
                    c = cell(*dv, r, sc)
                    if c is None:
                        L.append(f"| {NAME[dv]} | {r} | {sc} | not run | | | | | | | | |")
                        continue
                    g = G[dv][0].get(r)
                    acc = f"{g['acc_seen']:.4f}" if g else "n/a"
                    fl = ", ".join(k for k, v in c["flags"].items() if v) or "-"
                    for s in [x for x in ("leaky", "F2", "delta_fit", "leaky_gap", "delta_fit_gap", "gap_effect", "truth",
                                          "a_gap", "a_gap_cm") if x in c["stats"]]:
                        t = c["stats"][s]
                        L.append(f"| {NAME[dv]} | {r} | {sc} | {s} | {f4(t['est'])} | {ci(t['jk_ci95'])} | "
                                 f"{ci(t['boot_ci95_ref'])} | {c['n_groups_fit_per_fold']} | {c['K']} | {c['d']} | {acc} | {fl} |")
    L += ["", "## CXR (P2-a, descriptive)", "", "| backbone | scorer | Delta_fit (Shenzhen) |", "|---|---|---|"]
    for r in ("resnet50_s42", "resnet50_s43", "convnext_tiny_s42", "convnext_tiny_s43", "dinov2_vitb14", "rad_dino"):
        for sc in SCORERS:
            c = j(P2A / "cells" / f"{r}_{sc}.json")
            if c:
                L.append(f"| {r} | {sc} | {f4(c['ood']['shenzhen']['delta_fit']['est'])} |")
    nseen = {dv: next((cell(*dv, r, "mahalanobis_l2")["n"] for r in BACKBONES["resnet50"] if cell(*dv, r, "mahalanobis_l2")), None)
             for dv in DV}
    L += ["", "## Caveats", "",
          "1. Brain MRI has no slice index, so it has no adjacent-slice gap variant; adjacent-slice similarity stays in the brain "
          "estimate and can only raise it (residual asymmetry against the ordering prediction).",
          "2. K_seg is a mechanistic control, not a protocol from the literature; K_lit is the literature-protocol effect size "
          "(frame-level random split, Li et al. 2023; El-Ghany et al. 2024), reported without a gap.",
          "3. A_gap is descriptive; Kvasir-Capsule has 7 unseen videos (A_gap underpowered by construction).",
          "4. Seen images whose group has no fit image are dropped and counted: " + "; ".join(
              f"{NAME[dv]} {n['seen_dropped_no_fit_group']}" for dv, n in nseen.items() if n) + ".",
          "5. Backbone aggregation over seeds (mean; robust = every seed's CI > 0; ordering bounds = min / max over seeds) is "
          "not specified in the precommit and was fixed before the cells were computed.",
          "6. Package check: leaky and F2 match crossfit_auroc within 1e-9 plus one pair per near-tied seen / OOD score pair "
          "(the P2-b technical fix, applied from the start here)."]
    (P2D / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
