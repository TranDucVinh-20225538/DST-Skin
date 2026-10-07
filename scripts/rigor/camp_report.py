#!/usr/bin/env python3
"""MICCAI campaign report (decisions/precommit_miccai_full_campaign_2026-10-07.md): Tracks A / C / D / E tables,
L7 decision. Reads only existing result files; empty cells are NA.

Writes outputs/reports/rigor_pack/miccai_campaign/{summary.md, summary.csv, tables.json,
decision_miccai_vs_midl.md, access_status.json, trackA_foundation/*, trackC_clinical_tau/*,
trackD_backbone_leak/*, trackE_f2_medical/*}.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import medbench_report as MR  # noqa: E402

REPO = C.REPO
OUT = REPO / "outputs/reports/rigor_pack/miccai_campaign"
FG = REPO / "outputs/reports/rigor_pack/foundation_gate"
MECH = REPO / "outputs/reports/rigor_pack/mechanism_fix/cells"
MED = REPO / "outputs/reports/rigor_pack/medbench"
CACHE = REPO / "outputs/rigor_pack/miccai_campaign/scores"
CPU = OUT / "cpu_cells"
V2 = REPO / "outputs/reports/rigor_pack/leakage/logit_retrain_slide_disjoint_v2"
FMS = ("dinov2_vitb14", "uni", "conch_v1_5", "virchow2", "dinov2_vitl14")
CNN8 = ("resnet18", "resnet50", "densenet121", "convnext_tiny", "mobilenet_v3_large", "regnet_y_3_2gf", "effb3",
        "efficientnet_v2_s")
ANCH = ("resnet50", "convnext_tiny")
SEEDS = (42, 43, 44)
MEDDS = ("breakhis", "isic2019", "dermamnist", "kermany")
FEAT = ("Mahalanobis", "kNN", "ViM")
FIT = ("Mahalanobis", "kNN", "ViM", "ReAct")
ORDER = list(C.METHODS_ORDER)
NEAR = (0.45, 0.55)


def lj(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def near(same, dis, m):
    return NEAR[0] <= same[m] <= NEAR[1] and NEAR[0] <= dis[m] <= NEAR[1]


def med_feat(delta: dict, same: dict, dis: dict):
    c = [m for m in FEAT if not near(same, dis, m)]
    return float(np.median([delta[m] for m in c])) if c else None


# ---------------------------------------------------------------- Camelyon cells
def cam_cell(model: str, seed: int):
    if model in FMS:
        return lj(FG / "cells" / ("camelyon_%s%s.json" % (model, "" if seed == 42 else "_s%d" % seed)))
    j = lj(FG / "cells" / ("camelyon_%s%s.json" % (model, "" if seed == 42 else "_s%d" % seed)))
    if j is None and seed == 42:
        m = lj(MECH / ("camelyon_%s.json" % model))
        if m:
            j = {"model": model, "kind": "cnn", "seed": 42, "standard": m["standard"], "same_2fold": m["same_2fold"],
                 "disjoint_2fold": m["disjoint_2fold"], "delta": m["delta"], "F2": m["F2"], "source": "mechanism cell"}
    return j


def cam_rank(j):
    sc = [m for m in ORDER if not near(j["same_2fold"], j["disjoint_2fold"], m)]
    a = [j["same_2fold"][m] for m in sc]
    b = [j["disjoint_2fold"][m] for m in sc]
    wl, wd = sc[int(np.argmax(a))], sc[int(np.argmax(b))]
    return {"winner_leaky": wl, "winner_disjoint": wd, "winner_changed": wl != wd,
            "kendall_tau_b": float(kendalltau(a, b).statistic)}


def cam_cache_metrics(name: str, j: dict):
    """Track C (τ@95% TPR, fold fits) and Track E F2 from the per-image score cache."""
    p = CACHE / ("%s.npz" % name)
    if not p.exists():
        return None, None
    z = np.load(p)
    fi = z["fi"]
    tc = {}
    for m in ORDER:
        r = []
        for f in (0, 1):
            s_leak, s_new, s_ood = z["f%d_id_%s" % (f, m)][fi == f], z["f%d_id_%s" % (f, m)][fi == 1 - f], z["f%d_ood_%s" % (f, m)]
            t = np.percentile(s_leak, 5)
            r.append([np.mean(s_leak >= t), np.mean(s_new >= t), np.mean(s_new < t), np.mean(s_ood >= t)])
        r = np.mean(r, 0)
        tc[m] = {"tpr_leaky": float(r[0]), "tpr_new": float(r[1]), "fa_new": float(r[2]), "ood_pass": float(r[3])}
    from src.utils.benchmark_metrics import calc_auroc
    keep = fi >= 0
    F2 = {m: float(calc_auroc(z["std_id_%s" % m][keep], z["std_ood_%s" % m])) for m in ORDER}
    for m in FIT:
        idc = np.full(len(fi), np.nan)
        for g in (0, 1):
            idc[fi == g] = z["f%d_id_%s" % (1 - g, m)][fi == g]
        F2[m] = float(calc_auroc(idc[keep], (z["f0_ood_%s" % m] + z["f1_ood_%s" % m]) / 2))
    return tc, {"F1": j["disjoint_2fold"], "F2": F2}


# ---------------------------------------------------------------- medbench rows (CNN + FM)
def fm_rows() -> pd.DataFrame:
    """medbench_report.run_rows over the FM score files; G1 does not apply to a convex probe (precommit L5):
    a placeholder train record makes G1 True, G2 / G3 as medbench."""
    tmp = Path(tempfile.mkdtemp())
    try:
        for ds in MEDDS:
            (tmp / ds).mkdir()
            for p in (FG / ds).glob("scores_fm_*.json") if (FG / ds).exists() else []:
                shutil.copy(p, tmp / ds / p.name)
                (tmp / ds / p.name.replace("scores_", "train_")).write_text(json.dumps(
                    {"final_train_acc": 1.0, "final_loss": 0.0, "epoch1_loss": 1.0, "train_seconds": 0.0}))
        for ds in set(MR.ALL_DS) - set(MEDDS):
            (tmp / ds).mkdir()
        R = MR.run_rows(tmp)
    finally:
        shutil.rmtree(tmp)
    if len(R):
        R["G1"] = "n/a"
    return R


def med_backbone_delta(R: pd.DataFrame, ds: str):
    """Per backbone (seed 42, std-type arms, passing): mean Δ_fit / same / disjoint over runs, median feature Δ."""
    P = R[(R.dataset == ds) & (R.seed == 42) & (R.family == "A") & R.passes.astype(bool)]
    out = {}
    for arch, g in P.groupby("arch"):
        if "dfit_Mahalanobis" not in g or g["dfit_Mahalanobis"].isna().all():
            continue
        d = {m: float(g["dfit_%s" % m].mean()) for m in FIT}
        s = {m: float(g["same_%s" % m].mean()) for m in FIT}
        u = {m: float(g["disjoint_%s" % m].mean()) for m in FIT}
        out[arch] = {"delta": d, "same": s, "disjoint": u, "median_feature_delta": med_feat(d, s, u), "n_runs": len(g)}
    return out


def main() -> int:
    for d in ("trackA_foundation", "trackC_clinical_tau", "trackD_backbone_leak", "trackE_f2_medical"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    tables = {}

    # ---------------- Track A: Camelyon
    cam = {}
    for model in CNN8 + FMS:
        for s in SEEDS:
            j = cam_cell(model, s)
            if j:
                cam[(model, s)] = j
    rowsA = []
    for (model, s), j in cam.items():
        r = {"dataset": "camelyon17", "backbone": model, "kind": "FM" if model in FMS else "CNN", "seed": s,
             "source": j.get("source", "campaign scorer")}
        for m in ORDER:
            r["std_%s" % m] = j["standard"][m]
            r["leaky_%s" % m], r["disjoint_%s" % m] = j["same_2fold"][m], j["disjoint_2fold"][m]
        for m in FIT:
            r["dfit_%s" % m] = j["delta"][m]
            if "delta_ci" in j:
                r["dfit_lo_%s" % m], r["dfit_hi_%s" % m] = j["delta_ci"][m]
        r["median_feature_delta"] = med_feat(j["delta"], j["same_2fold"], j["disjoint_2fold"])
        if "logit_gap" in j:
            for m in ("MSP", "Energy", "ELogitNorm"):
                r["logit_gap_%s" % m] = j["logit_gap"][m]["gap"]
        if "probe" in j:
            r["id_acc"] = j["probe"]["id_acc"]
        if "fold_probe_acc" in j:
            r["acc_seen"], r["acc_unseen"] = j["fold_probe_acc"]["seen"], j["fold_probe_acc"]["unseen"]
        r.update({"rank_" + k: v for k, v in cam_rank(j).items()})
        rowsA.append(r)

    # ---------------- Track A: medical (FM) + CNN medbench rows
    Rc = MR.run_rows(MED)
    Rf = fm_rows()
    R = pd.concat([Rc, Rf], ignore_index=True) if len(Rf) else Rc
    for ds in ("breakhis", "isic2019", "dermamnist"):
        for arch, v in med_backbone_delta(Rf, ds).items() if len(Rf) else []:
            r = {"dataset": ds, "backbone": arch.replace("fm_", ""), "kind": "FM", "seed": 42}
            for m in FIT:
                r["dfit_%s" % m] = v["delta"][m]
                r["leaky_%s" % m], r["disjoint_%s" % m] = v["same"][m], v["disjoint"][m]
            r["median_feature_delta"] = v["median_feature_delta"]
            g = Rf[(Rf.dataset == ds) & (Rf.arch == arch) & (Rf.family == "A")]
            for col in ("acc_test_seen", "acc_test_unseen", "acc_seen", "acc_unseen"):
                if col in g and g[col].notna().any():
                    r[col] = float(g[col].mean())
            rowsA.append(r)
    A = pd.DataFrame(rowsA)
    A.to_csv(OUT / "trackA_foundation/summary.csv", index=False)
    if len(Rf):
        Rf.drop(columns=[c for c in Rf.columns if c.startswith("pair_ci")]).to_csv(
            OUT / "trackA_foundation/medical_fm_runs.csv", index=False)

    # ---------------- L7 (a): Δ per dataset
    l7 = {}
    camb = {m: j for (m, s), j in cam.items() if s == 42}
    per = {m: med_feat(j["delta"], j["same_2fold"], j["disjoint_2fold"]) for m, j in camb.items()}
    l7["camelyon17"] = {"per_backbone": per}
    for ds in MEDDS:
        l7[ds] = {"per_backbone": {a: v["median_feature_delta"] for a, v in med_backbone_delta(R, ds).items()}}
    for ds, v in l7.items():
        vals = {k: x for k, x in v["per_backbone"].items() if x is not None}
        fm = [x for k, x in vals.items() if k.replace("fm_", "") in FMS]
        cn = [x for k, x in vals.items() if k.replace("fm_", "") not in FMS]
        v["median"] = float(np.median(list(vals.values()))) if vals else None
        v["median_fm"] = float(np.median(fm)) if fm else None
        v["median_cnn"] = float(np.median(cn)) if cn else None
        v["n_backbones"], v["n_fm"] = len(vals), len(fm)
        v["ge_0.05"] = bool(v["median"] is not None and v["median"] >= 0.05)
    three = [med_feat(j["delta"], j["same_2fold"], j["disjoint_2fold"]) for (m, s), j in cam.items() if m in ANCH + FMS]
    three = [x for x in three if x is not None]
    l7["camelyon17"]["three_seed_median"] = float(np.median(three)) if three else None
    l7["camelyon17"]["three_seed_n"] = len(three)
    unstable = (l7["camelyon17"]["ge_0.05"] and l7["camelyon17"]["three_seed_median"] is not None
                and l7["camelyon17"]["three_seed_median"] < 0.02)
    l7["camelyon17"]["unstable"] = bool(unstable)
    count_ds = [ds for ds, v in l7.items() if v["ge_0.05"] and not (ds == "camelyon17" and unstable)]

    # ---------------- L7 (b): ranking flip
    flip = {}
    cr = [cam_rank(j) for j in camb.values()]
    flip["camelyon17"] = {"n": len(cr), "n_changed": int(sum(x["winner_changed"] for x in cr)),
                          "median_tau": float(np.median([x["kendall_tau_b"] for x in cr])) if cr else None}
    RK = MR.ranking(R)
    RK = RK[RK.seed == 42] if len(RK) else RK
    if len(RK):
        RK.to_csv(OUT / "trackA_foundation/ranking_medical_seed42.csv", index=False)
    for ds in MEDDS:
        g = RK[RK.dataset == ds] if len(RK) else RK
        flip[ds] = {"n": int(len(g)), "n_changed": int(g.winner_changed.sum()) if len(g) else 0,
                    "median_tau": float(g.kendall_tau_b.median()) if len(g) else None,
                    "n_fm_rows": int(g.arch.str.startswith("fm_").sum()) if len(g) else 0}
    for ds, v in flip.items():
        v["flip"] = bool(v["n"] and (2 * v["n_changed"] >= v["n"] or (v["median_tau"] is not None and v["median_tau"] < 0.5)))
    flip_ds = [ds for ds, v in flip.items() if v["flip"]]
    venue = "MICCAI" if len(count_ds) >= 2 and len(flip_ds) >= 1 else "MIDL"

    # ---------------- Track C
    rowsC = []
    for (model, s), j in cam.items():
        name = "camelyon_%s%s" % (model, "" if s == 42 else "_s%d" % s)
        tc, _ = cam_cache_metrics(name, j)
        if tc:
            for m in ORDER:
                rowsC.append({"dataset": "camelyon17", "backbone": model, "seed": s, "score": m, **tc[m]})
    for p in sorted(CPU.glob("*.json")):
        j = lj(p)
        for m in ORDER:
            rowsC.append({"dataset": j["ds"], "backbone": j["model"].replace("fm_", ""), "seed": 42, "arm": j["arm"],
                          "score": m, **{k: v for k, v in j["trackC"][m].items() if k != "tau"}})
    TC = pd.DataFrame(rowsC)
    if len(TC):
        TC = TC.groupby(["dataset", "backbone", "seed", "score"], as_index=False)[["tpr_leaky", "tpr_new", "fa_new", "ood_pass"]].mean()
        TC.to_csv(OUT / "trackC_clinical_tau/tau95.csv", index=False)
        tcs = TC[TC.seed == 42].groupby(["dataset", "score"]).tpr_new.median().unstack()
        tcs.to_csv(OUT / "trackC_clinical_tau/median_tpr_new_by_dataset.csv")

    # ---------------- Track D
    rowsD = []
    pb = pd.read_csv(REPO / "outputs/reports/rigor_pack/isbi_patch/phaseB_bar.csv").set_index("arch")
    for arch in ("resnet50", "convnext_tiny", "densenet121"):
        for s in SEEDS:
            js = [lj(V2 / ("%s_s%d_f%d.json" % (arch, s, f))) for f in (0, 1)]
            j = cam.get((arch, s))
            r = {"arch": arch, "seed": s,
                 "a_feature_dfit_median": med_feat(j["delta"], j["same_2fold"], j["disjoint_2fold"]) if j else None,
                 "a_dfit_ReAct": j["delta"]["ReAct"] if j else None}
            for m in FEAT:
                r["a_dfit_%s" % m] = j["delta"][m] if j else None
            if all(js):
                r["b_within_gap_MSP"] = float(np.mean([x["gap_msp"] for x in js]))
                r["b_within_gap_Energy"] = float(np.mean([x["gap_energy"] for x in js]))
                r["b_acc_seen"] = float(np.mean([x["acc_id_seen"] for x in js]))
                r["b_acc_unseen"] = float(np.mean([x["acc_id_unseen"] for x in js]))
            if s == 42 and arch in pb.index:
                r["b_published_minus_disjoint_MSP"] = float(pb.loc[arch, "delta_msp"])
                r["b_published_minus_disjoint_Energy"] = float(pb.loc[arch, "delta_energy"])
            rowsD.append(r)
    D = pd.DataFrame(rowsD)
    D.to_csv(OUT / "trackD_backbone_leak/side_by_side.csv", index=False)

    # ---------------- Track E
    rowsE = []
    for (model, s), j in cam.items():
        if s != 42:
            continue
        if "F2" in j:
            e = {"F1": j["disjoint_2fold"], "F2": j["F2"]}
        else:
            _, e = cam_cache_metrics("camelyon_%s" % model, j)
        if e:
            rowsE.append({"dataset": "camelyon17", "backbone": model, **{"F1_%s" % m: e["F1"][m] for m in ORDER},
                          **{"F2_%s" % m: e["F2"][m] for m in ORDER}})
    cpuE = {}
    for p in sorted(CPU.glob("*.json")):
        j = lj(p)
        cpuE.setdefault((j["ds"], j["model"]), []).append(j["trackE"])
    for (ds, model), lst in cpuE.items():
        rowsE.append({"dataset": ds, "backbone": model.replace("fm_", ""),
                      **{"F1_%s" % m: float(np.mean([x["F1"][m] for x in lst])) for m in ORDER},
                      **{"F2_%s" % m: float(np.mean([x["F2"][m] for x in lst])) for m in ORDER}})
    E = pd.DataFrame(rowsE)
    resE = {}
    if len(E):
        E["abs_diff_med"] = E.apply(lambda r: float(np.median([abs(r["F2_%s" % m] - r["F1_%s" % m]) for m in FIT])), axis=1)
        E["tau"] = E.apply(lambda r: float(kendalltau([r["F2_%s" % m] for m in ORDER], [r["F1_%s" % m] for m in ORDER]).statistic), axis=1)
        E.to_csv(OUT / "trackE_f2_medical/f2_vs_f1.csv", index=False)
        for ds, g in E.groupby("dataset"):
            diffs = [abs(r["F2_%s" % m] - r["F1_%s" % m]) for _, r in g.iterrows() for m in FIT]
            resE[ds] = {"median_abs_diff": float(np.median(diffs)), "median_tau": float(g.tau.median()), "n_backbones": len(g)}
            resE[ds]["works"] = resE[ds]["median_abs_diff"] <= 0.02 and resE[ds]["median_tau"] >= 0.8
        n_ok = sum(v["works"] for v in resE.values())
        resE["_verdict"] = {"n_work": n_ok, "n": len(resE), "F2_works": bool(n_ok >= 2 / 3 * len(resE))}

    # ---------------- outputs
    tables.update({"l7_delta": l7, "l7_flip": flip, "l7_count_ds": count_ds, "l7_flip_ds": flip_ds, "venue": venue,
                   "trackE": resE,
                   "react_camelyon_fm_median_abs": float(np.median([abs(j["delta"]["ReAct"]) for m, j in camb.items() if m in FMS]))
                   if any(m in FMS for m in camb) else None})
    (OUT / "tables.json").write_text(json.dumps(tables, indent=2, default=str) + "\n")
    keep = ["dataset", "backbone", "kind", "seed"] + [c for c in A.columns if c.startswith(("leaky_", "disjoint_", "dfit_", "std_"))] \
        + [c for c in ("median_feature_delta", "id_acc", "acc_seen", "acc_unseen", "acc_test_seen", "acc_test_unseen") if c in A]
    S = A[[c for c in keep if c in A]].copy()
    if len(TC):
        t = TC.pivot_table(index=["dataset", "backbone", "seed"], columns="score", values="tpr_new").add_prefix("tau95_tpr_new_").reset_index()
        S = S.merge(t, on=["dataset", "backbone", "seed"], how="left")
    S.to_csv(OUT / "summary.csv", index=False)

    f = lambda x: "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else "%+.3f" % x  # noqa: E731
    L = ["# Decision: MICCAI vs MIDL (L7)", "", "**Venue: %s**" % venue, "",
         "Rule (precommit L7, locked before results): Δ_fit (feature) >= 0.05 on >= 2 medical datasets AND ranking flip "
         "on >= 1 medical → MICCAI; else MIDL.", "",
         "| dataset | median feature Δ_fit (all backbones) | CNN median | FM median | n backbones (FM) | >= 0.05 | ranking rows changed / n | median τ-b | flip |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ds in ("camelyon17",) + MEDDS:
        v, fl = l7[ds], flip[ds]
        L.append("| %s | %s | %s | %s | %d (%d) | %s | %d / %d | %s | %s |" % (
            ds, f(v["median"]), f(v["median_cnn"]), f(v["median_fm"]), v["n_backbones"], v["n_fm"], v["ge_0.05"],
            fl["n_changed"], fl["n"], f(fl["median_tau"]), fl["flip"]))
    L += ["", "- Datasets counted toward Δ >= 0.05: %s (%d)" % (", ".join(count_ds) or "none", len(count_ds)),
          "- Camelyon 3-seed median feature Δ_fit (anchors + FMs, seeds 42-44, n = %d): %s → unstable = %s" % (
              l7["camelyon17"]["three_seed_n"], f(l7["camelyon17"]["three_seed_median"]), unstable),
          "- Datasets with a ranking flip: %s" % (", ".join(flip_ds) or "none"),
          "- Secondary: ReAct median |Δ_fit| over Camelyon FM cells = %s" % f(tables["react_camelyon_fm_median_abs"])]
    (OUT / "decision_miccai_vs_midl.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
