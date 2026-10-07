#!/usr/bin/env python3
"""Gate report for decisions/precommit_foundation_leakage_gate_2026-10-07.md (L4).

Reads outputs/reports/rigor_pack/foundation_gate/cells/camelyon_{fm}.json (non-smoke), the load records
(outputs/rigor_pack/foundation_gate/load/{fm}.json), the CNN anchors (mechanism cells + m4logit) and the
DermaMNIST medbench score files; writes access_status.json, tables.json, summary.csv, summary.md,
gate_decision.md in outputs/reports/rigor_pack/foundation_gate/. Empty cells are NA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

REPO = C.REPO
REP = REPO / "outputs/reports/rigor_pack/foundation_gate"
MECH = REPO / "outputs/reports/rigor_pack/mechanism_fix/cells"
MED = REPO / "outputs/reports/rigor_pack/medbench/dermamnist"
FMS = ("dinov2_vitb14", "uni", "conch_v1_5", "dinov2_vitl14")
ANCHORS = ("resnet50", "convnext_tiny")
FEAT = ("Mahalanobis", "kNN", "ViM")
LOGIT = ("MSP", "Energy", "ELogitNorm")
ORDER = list(C.METHODS_ORDER)
THR, NOISE, REACT_FLAT = 0.02, 0.01, 0.02
EXTRA_STOP = {
    "conch_v1": {"status": "STOP", "repo": "MahmoodLab/CONCH",
                 "reason": "GatedRepoError 403: configured HF token is not in the authorized list for "
                           "MahmoodLab/CONCH (checked 2026-10-07); precommitted fallback CONCH v1.5 used"},
    "uni2_h": {"status": "not run", "repo": "MahmoodLab/UNI2-h",
               "reason": "precommit L2: one UNI only (UNI v1); UNI2-h weights not cached"},
}


def fmt(x, d=3):
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else ("%+.*f" % (d, x))


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fm_verdict(c: dict) -> dict:
    counted = [m for m in FEAT if not c["near_chance"][m]]
    d = {m: c["delta"][m] for m in FEAT}
    n_inf = sum(d[m] > THR for m in counted)
    excl = sum((c["delta_ci"][m][0] > 0 or c["delta_ci"][m][1] < 0) for m in counted)
    med = float(np.median([d[m] for m in counted])) if counted else None
    return {"counted": counted, "n_feat_gt_0.02": int(n_inf), "inflation": bool(n_inf >= 2),
            "median_feat_delta": med, "n_ci_excl_0": int(excl), "ci_ok": bool(excl >= 2),
            "react_delta": c["delta"]["ReAct"], "react_near_chance": bool(c["near_chance"]["ReAct"])}


def main() -> int:
    REP.mkdir(parents=True, exist_ok=True)
    # access status
    access = {}
    for fm in FMS:
        r = load_json(REPO / "outputs/rigor_pack/foundation_gate/load" / ("%s.json" % fm))
        access[fm] = r if r else {"status": "not attempted"}
    access.update(EXTRA_STOP)
    (REP / "access_status.json").write_text(json.dumps(access, indent=2) + "\n")

    cells = {fm: load_json(REP / "cells" / ("camelyon_%s.json" % fm)) for fm in FMS}
    cells = {k: v for k, v in cells.items() if v}
    anchors = {}
    for a in ANCHORS:
        m, lg = load_json(MECH / ("camelyon_%s.json" % a)), load_json(MECH / ("m4logit_camelyon_%s.json" % a))
        anchors[a] = {"standard": m["standard"], "same_2fold": m["same_2fold"], "disjoint_2fold": m["disjoint_2fold"],
                      "delta": m["delta"],
                      "logit_gap": {k: lg[k]["unseen_inst_seen_group"] - lg[k]["unseen_group"] for k in ("MSP", "Energy")}}
    anchor_med = float(np.median([np.median([anchors[a]["delta"][m] for m in FEAT]) for a in ANCHORS]))

    gate = compute_gate({fm: fm_verdict(c) for fm, c in cells.items()}, anchor_med)
    v, n = gate["per_fm"], gate["n_fm"]
    vc = load_json(REP / "cells" / "camelyon_virchow2.json")
    gate_v2 = compute_gate({**v, "virchow2": fm_verdict(vc)}, anchor_med) if vc else None
    return report(cells, anchors, anchor_med, gate, gate_v2)


def compute_gate(v: dict, anchor_med: float) -> dict:
    n = len(v)
    gate = {"n_fm": n, "fms": list(v), "anchor_median_feature_delta": anchor_med, "per_fm": v}
    if n == 0:
        gate.update(decision="INCONCLUSIVE", reason="no FM loaded / scored")
    else:
        n_inf = sum(x["inflation"] for x in v.values())
        meds = [x["median_feat_delta"] for x in v.values() if x["median_feat_delta"] is not None]
        med_of_med = float(np.median(meds)) if meds else None
        n_ci = sum(x["ci_ok"] for x in v.values())
        c1a = 2 * n_inf >= n
        c1b = (med_of_med is not None and np.sign(med_of_med) == np.sign(anchor_med) and med_of_med > NOISE
               and 2 * n_ci >= n)
        react = [abs(x["react_delta"]) for x in v.values() if not x["react_near_chance"]]
        react_med = float(np.median(react)) if react else None
        c2 = react_med is not None and react_med <= REACT_FLAT
        gate.update({"n_fm_inflation": int(n_inf), "cond1a_half_inflate": bool(c1a),
                     "median_of_fm_median_feature_delta": med_of_med, "n_fm_ci_ok": int(n_ci),
                     "cond1b_sign_noise": bool(c1b), "cond1": bool(c1a or c1b),
                     "median_abs_react_delta": react_med, "n_react_counted": len(react), "cond2_react_flat": bool(c2),
                     "decision": "PASS" if (c1a or c1b) and c2 else "FAIL"})
    return gate


def gate_lines(gate: dict, anchor_med: float) -> list:
    v, n = gate["per_fm"], gate["n_fm"]
    if not n:
        return ["No FM loaded and scored."]
    G = ["- n loaded and scored FMs = %d (%s)" % (n, ", ".join(v)),
         "- (1a) FMs with >= 2 of 3 counted feature Δ_fit > 0.02: %d / %d → %s" % (
             gate["n_fm_inflation"], n, gate["cond1a_half_inflate"]),
         "- (1b) median of per-FM median feature Δ_fit = %s (CNN anchor median %s), FMs with >= 2 CIs excluding 0: "
         "%d / %d → %s" % (fmt(gate["median_of_fm_median_feature_delta"]), fmt(anchor_med), gate["n_fm_ci_ok"], n,
                          gate["cond1b_sign_noise"]),
         "- (1) = %s" % gate["cond1"],
         "- (2) median |Δ_fit ReAct| over %d FMs = %s (<= 0.02) → %s" % (
             gate["n_react_counted"], fmt(gate["median_abs_react_delta"]), gate["cond2_react_flat"]), "",
         "| FM | counted feature scores | # Δ_fit > 0.02 | inflation | median feature Δ_fit | # CI excl. 0 | ReAct Δ_fit |",
         "|---|---|---|---|---|---|---|"]
    for fm, x in v.items():
        G.append("| %s | %s | %d | %s | %s | %d | %s |" % (fm, ", ".join(x["counted"]), x["n_feat_gt_0.02"], x["inflation"],
                                                         fmt(x["median_feat_delta"]), x["n_ci_excl_0"], fmt(x["react_delta"])))
    return G


def report(cells, anchors, anchor_med, gate, gate_v2) -> int:
    # DermaMNIST (secondary, not part of the gate)
    derma = {}
    for name, tag in [(fm, "fm_%s" % fm) for fm in FMS] + [(a, a) for a in ANCHORS]:
        row = {}
        for armn in ("std", "b0", "b1"):
            p = (REP / "dermamnist" if name in FMS else MED) / ("scores_%s_s42_%s.json" % (tag, armn))
            s = load_json(p)
            if s is None:
                continue
            if armn == "std" and "afit" in s:
                row["delta_fit"] = s["afit"]["delta_fit"]
                row["delta_fit_ci"] = s["afit"]["delta_fit_ci"]
                row["A_gap_std"] = s["gap_ood"]
                row["acc_test_unseen"] = s.get("acc_test_unseen")
            else:
                row["B_gap_%s" % armn] = s["gap_ood"]
                row["acc_unseen_%s" % armn] = s.get("acc_unseen")
            pr = load_json(REP / "dermamnist" / ("probe_%s_%s.json" % (name, armn)))
            if pr:
                row["probe_train_acc_%s" % armn] = pr["probe_train_acc"]
        if row:
            derma[name] = row

    tables = {"camelyon": {"fm": cells, "anchors": anchors}, "dermamnist": derma, "gate": gate,
              "gate_with_virchow2": gate_v2}
    (REP / "tables.json").write_text(json.dumps(tables, indent=2) + "\n")

    rows = []
    for name, c in list(cells.items()) + list(anchors.items()):
        r = {"model": name, "kind": "FM" if name in FMS else "CNN anchor", "dataset": "camelyon17"}
        for m in ORDER:
            r["std_%s" % m] = c["standard"].get(m)
        for m in FEAT + ("ReAct",):
            r["same_%s" % m], r["disjoint_%s" % m] = c["same_2fold"][m], c["disjoint_2fold"][m]
            r["dfit_%s" % m] = c["delta"][m]
            if "delta_ci" in c:
                r["dfit_%s_lo" % m], r["dfit_%s_hi" % m] = c["delta_ci"][m]
                r["near_chance_%s" % m] = c["near_chance"][m]
        for m in LOGIT:
            g = c["logit_gap"].get(m)
            r["logit_gap_%s" % m] = g["gap"] if isinstance(g, dict) else g
            if "logit_gap_ci" in c:
                r["logit_gap_%s_lo" % m], r["logit_gap_%s_hi" % m] = c["logit_gap_ci"][m]
        if name in FMS:
            r["probe_train_acc"], r["probe_id_acc"] = c["probe"]["train_acc"], c["probe"]["id_acc"]
            r["fold_probe_acc_seen"], r["fold_probe_acc_unseen"] = c["fold_probe_acc"]["seen"], c["fold_probe_acc"]["unseen"]
            r["kendall_tau_b_same_vs_disjoint"] = c["kendall_tau_b_same_vs_disjoint"]
        rows.append(r)
    for name, d in derma.items():
        r = {"model": name, "kind": "FM" if name in FMS else "CNN anchor", "dataset": "dermamnist"}
        for m in ("Mahalanobis", "kNN", "ViM", "ReAct"):
            if "delta_fit" in d:
                r["dfit_%s" % m] = d["delta_fit"][m]
                r["dfit_%s_lo" % m], r["dfit_%s_hi" % m] = d["delta_fit_ci"][m]
        rows.append(r)
    pd.DataFrame(rows).to_csv(REP / "summary.csv", index=False)

    L = ["# Foundation-embedding group-leakage gate: summary", "",
         "Precommit `decisions/precommit_foundation_leakage_gate_2026-10-07.md`. Seed 42. Δ_fit = same-slide-fit "
         "AUROC − disjoint-slide-fit AUROC (H11c 2-fold, fold mean); [95% cluster-bootstrap CI]. NA = not run.", "",
         "## Camelyon17", "",
         "| model | Maha Δ_fit | kNN Δ_fit | ViM Δ_fit | ReAct Δ_fit | MSP gap | Energy gap | probe ID acc | fold-probe acc seen / unseen |",
         "|---|---|---|---|---|---|---|---|---|"]
    for fm, c in cells.items():
        ci = lambda m: "%s [%s, %s]%s" % (fmt(c["delta"][m]), fmt(c["delta_ci"][m][0]), fmt(c["delta_ci"][m][1]),  # noqa: E731
                                          " (near-chance)" if c["near_chance"][m] else "")
        L.append("| %s | %s | %s | %s | %s | %s | %s | %.3f | %.3f / %.3f |" % (
            fm, ci("Mahalanobis"), ci("kNN"), ci("ViM"), ci("ReAct"), fmt(c["logit_gap"]["MSP"]["gap"]),
            fmt(c["logit_gap"]["Energy"]["gap"]), c["probe"]["id_acc"], c["fold_probe_acc"]["seen"],
            c["fold_probe_acc"]["unseen"]))
    for a, c in anchors.items():
        L.append("| %s (CNN anchor) | %s | %s | %s | %s | %s | %s | — | — |" % (
            a, fmt(c["delta"]["Mahalanobis"]), fmt(c["delta"]["kNN"]), fmt(c["delta"]["ViM"]), fmt(c["delta"]["ReAct"]),
            fmt(c["logit_gap"]["MSP"]), fmt(c["logit_gap"]["Energy"])))
    L += ["", "Standard AUROC (full fit):", "", "| model | " + " | ".join(ORDER) + " |", "|---" * (len(ORDER) + 1) + "|"]
    for name, c in list(cells.items()) + list(anchors.items()):
        L.append("| %s | " % name + " | ".join("%.3f" % c["standard"][m] for m in ORDER) + " |")
    L += ["", "## DermaMNIST (secondary, not part of the gate)", "",
          "| model | Maha Δ_fit | kNN Δ_fit | ViM Δ_fit | ReAct Δ_fit | acc test_unseen |", "|---|---|---|---|---|---|"]
    for name, d in derma.items():
        if "delta_fit" not in d:
            continue
        cc = lambda m: "%s [%s, %s]" % (fmt(d["delta_fit"][m]), fmt(d["delta_fit_ci"][m][0]), fmt(d["delta_fit_ci"][m][1]))  # noqa: E731
        L.append("| %s | %s | %s | %s | %s | %s |" % (name, cc("Mahalanobis"), cc("kNN"), cc("ViM"), cc("ReAct"),
                                                      "NA" if d.get("acc_test_unseen") is None else "%.3f" % d["acc_test_unseen"]))
    (REP / "summary.md").write_text("\n".join(L) + "\n")

    G = ["# Gate decision (L4)", "", "**%s**" % gate["decision"], ""] + gate_lines(gate, anchor_med)
    if gate_v2:
        G += ["", "## With Virchow2 (campaign addition, deviation 2; reported separately)", "",
              "**%s**" % gate_v2["decision"], ""] + gate_lines(gate_v2, anchor_med)
    (REP / "gate_decision.md").write_text("\n".join(G) + "\n")
    print("\n".join(G))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
