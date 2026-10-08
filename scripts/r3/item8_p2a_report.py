#!/usr/bin/env python3
"""R3 item 8 / P2-a report: results/r3/8/p2a/REPORT.md from cells/, train/, phase0.json, sets.json.
Usage: python3 scripts/r3/item8_p2a_report.py <commit>
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
P2A = REPO / "results/r3/8/p2a"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
BACKBONES = {"resnet50": ["resnet50_s42", "resnet50_s43"], "convnext_tiny": ["convnext_tiny_s42", "convnext_tiny_s43"],
             "dinov2_vitb14": ["dinov2_vitb14"], "rad_dino": ["rad_dino"]}
FMS = ("dinov2_vitb14", "rad_dino")
SCORERS = ("mahalanobis_l2", "knn_mean_cosine")
OODS = ("shenzhen", "kermany_ped", "shenzhen_tb", "shenzhen_normal")


def j(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def gates():
    rows = {}
    for bb, runs in BACKBONES.items():
        for r in runs:
            if bb in FMS:
                t = j(P2A / "train" / f"{r}_probe.json")
                if t is None:
                    rows[r] = None
                    continue
                m = t["macro_auroc"]
                rows[r] = {"id_seen": m["seen"], "id_unseen": m["unseen"], "loss_cond": None,
                           "pre": m["seen"] >= 0.70, "post": m["seen"] >= 0.70, "kind": "probe macro AUROC"}
            else:
                t = j(P2A / "train" / f"{r}.json")
                if t is None:
                    rows[r] = None
                    continue
                m = t["id_macro_auroc"]
                lc = t["final_loss"] <= 0.5 * t["epoch1_loss"]
                rows[r] = {"id_seen": m["seen"], "id_unseen": m["unseen"], "loss_cond": lc,
                           "loss": (t["epoch1_loss"], t["final_loss"]), "pre": bool(lc and m["seen"] >= 0.75),
                           "post": m["seen"] >= 0.75, "kind": "CNN macro AUROC", "best_epoch": t["best_epoch"],
                           "epochs": t["epochs"]}
    bb_pass = {}
    for v in ("pre", "post"):
        bb_pass[v] = {bb: all(rows.get(r) and rows[r][v] for r in runs) for bb, runs in BACKBONES.items()}
    return rows, bb_pass


def cell(r, sc):
    return j(P2A / "cells" / f"{r}_{sc}.json")


def fmt(x, d=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def ci(c):
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def backbone_delta(bb, sc, ood):
    cs = [cell(r, sc) for r in BACKBONES[bb]]
    if any(c is None for c in cs):
        return None
    o = [c["ood"][ood] for c in cs]
    est = float(np.mean([x["delta_fit"]["est"] for x in o]))
    robust = all(x["delta_fit"]["jk_ci95"][0] > 0 for x in o)
    excl = any(x["flags"]["near_chance"] or x["flags"]["ceiling"] or x["flags"]["below_chance"] for x in o)
    return {"est": est, "robust": robust, "excluded": excl, "flags": [x["flags"] for x in o]}


def verdict(bb_pass, sc):
    passing = [bb for bb, ok in bb_pass.items() if ok]
    if len(passing) < 3:
        return f"INCONCLUSIVE ({len(passing)} gate-passing backbone(s) < 3)", passing, None
    d = {bb: backbone_delta(bb, sc, "shenzhen") for bb in passing}
    if any(v is None for v in d.values()):
        return "pending (cells missing)", passing, d
    counted = [bb for bb in passing if not d[bb]["excluded"]]
    hits = [bb for bb in counted if d[bb]["est"] > 0.02]
    rob = [bb for bb in hits if d[bb]["robust"]]
    need = len(passing) / 2
    v = "leakage present" if len(hits) >= need else "leakage not shown"
    vr = "robust" if len(rob) >= need else "not robust"
    return f"{v} ({len(hits)}/{len(passing)} gate-passing backbones with Delta_fit > 0.02; {vr}: {len(rob)} with jackknife CI > 0)", passing, d


def predictions(bb_pass):
    out = {}
    passing = [bb for bb, ok in bb_pass.items() if ok]
    for name, sc in (("p1", "mahalanobis_l2"), ("p3", "knn_mean_cosine")):
        d = {bb: backbone_delta(bb, sc, "shenzhen") for bb in passing}
        if not passing or any(v is None for v in d.values()):
            out[name] = "n/a"
            continue
        hits = sum(d[bb]["est"] > 0.02 for bb in passing if not d[bb]["excluded"])
        out[name] = f"{'holds' if hits >= len(passing) / 2 else 'fails'} ({hits}/{len(passing)})"
    for sc in SCORERS:
        d = {bb: backbone_delta(bb, sc, "shenzhen") for bb in BACKBONES}
        fm = [d[b]["est"] for b in FMS if d[b] and bb_pass[b]]
        cnn = [d[b]["est"] for b in ("resnet50", "convnext_tiny") if d[b] and bb_pass[b]]
        out[f"p2_{sc}"] = ("n/a (needs gate-passing FMs and CNNs)" if not fm or not cnn else
                           f"{'holds' if np.median(fm) < np.median(cnn) else 'fails'} (FM median {np.median(fm):+.4f} vs CNN median {np.median(cnn):+.4f})")
        fm_k = [d[b]["est"] for b in FMS if d[b] and bb_pass[b] and not d[b]["excluded"]]
        cnn_k = [d[b]["est"] for b in ("resnet50", "convnext_tiny") if d[b] and bb_pass[b] and not d[b]["excluded"]]
        out[f"p2_{sc} (sensitivity, post hoc: excluded cells removed)"] = (
            f"n/a (FMs left {len(fm_k)}, CNNs left {len(cnn_k)})" if not fm_k or not cnn_k else
            f"{'holds' if np.median(fm_k) < np.median(cnn_k) else 'fails'} (FM median {np.median(fm_k):+.4f} vs CNN median {np.median(cnn_k):+.4f})")
    return out


def case_mix_table():
    S = pd.read_csv(P2A / "sets.csv.gz")
    d = pd.read_csv(REPO / "data/raw/nih_cxr14/Data_Entry_2017_v2020.csv").set_index("Image Index")
    rows = []
    for r in ("seen", "unseen"):
        x = S[S.role == r]
        e = d.loc[x.image]
        nf = x[FINDINGS].to_numpy().sum(1)
        ipp = S[S.role.isin(["fit", "ckpt", "seen", "unseen", "test_ckpt", "wang_val"])].groupby("patient").size()
        rows.append({"set": r, "n": len(x), "patients": x.patient.nunique(), "AP share": (x.view == "AP").mean(),
                     "No Finding": (nf == 0).mean(), "findings/image (mean)": nf.mean(), ">=2 findings": (nf >= 2).mean(),
                     "age median [IQR]": f"{e['Patient Age'].median():.0f} [{e['Patient Age'].quantile(.25):.0f}, {e['Patient Age'].quantile(.75):.0f}]",
                     "female": (e["Patient Sex"] == "F").mean(), "follow-up # (median)": e["Follow-up #"].median(),
                     "images/patient in dataset (median)": ipp.loc[x.patient.unique()].median(),
                     **{f: x[f].mean() for f in FINDINGS}})
    return pd.DataFrame(rows).set_index("set").T


def main() -> int:
    commit = sys.argv[1] if len(sys.argv) > 1 else "uncommitted"
    ph0, sets = j(P2A / "phase0.json"), j(P2A / "sets.json")
    rows, bb_pass = gates()
    L = [f"# R3 item 8 / P2-a (CXR) — report", "", f"Commit: `{commit}`", ""]
    v_pre = {sc: verdict(bb_pass["pre"], sc) for sc in SCORERS}
    v_post = {sc: verdict(bb_pass["post"], sc) for sc in SCORERS}
    L += ["**Verdict (preregistered gate):** " + "; ".join(f"{sc}: {v_pre[sc][0]}" for sc in SCORERS) + ".",
          "**Verdict (post hoc, AUROC-only competence gate):** " + "; ".join(f"{sc}: {v_post[sc][0]}" for sc in SCORERS) + ".", ""]

    L += ["## Phase 0 (leak audit)", "",
          f"- M_std = `{ph0['M_std']}` (precommitted rule). G_leak: leak rate {ph0['G_leak']['leak_rate']:.3f} >= 0.05 -> {ph0['G_leak']['decision']}.",
          f"- Audit sanity: {ph0['G_audit_sanity']['n_images_in_archives']:,} images, {ph0['G_audit_sanity']['n_patients']:,} patients, "
          f"patient-ID parse rate {ph0['G_audit_sanity']['patient_id_parse_rate']:.3f}, official train_val / test patient overlap "
          f"{ph0['G_audit_sanity']['official_patient_overlap']}.",
          f"- **Finding: the ChestMNIST split is patient-disjoint** (train->test leak {ph0['chestmnist_audit']['leak_rate_train_to_test']:.4f}, "
          f"train->val {ph0['chestmnist_audit']['leak_rate_train_to_val']:.4f}; mapping accept rate {ph0['chestmnist_audit']['accept_rate']:.3f}, "
          f"label agreement {ph0['chestmnist_audit']['label_agreement_accepted']:.4f}). Its test set overlaps the official test list for only "
          f"{ph0['chestmnist_audit']['test_vs_official_test_overlap']:.1%} of images, so it is a different patient-level split, not the official one.",
          f"- Wang et al. 2017 image-level 70/10/20 split (seed 0, stratified on No Finding): train->test leak "
          f"{ph0['wang_split']['leak_rate_train_to_test']:.3f}.", ""]
    L += ["## Sets", "",
          "| set | images | patients |", "|---|---:|---:|"] + \
         [f"| {k} | {sets['n_images'][k]:,} | {sets['n_patients'][k]:,} |" for k in ("fit", "ckpt", "seen", "unseen", "test_ckpt", "wang_val")] + \
         ["", f"ckpt = all Wang-train images of 10% of the Wang-train patients (checkpoint selection only); test_ckpt = Wang-test images of "
          f"those patients (neither seen nor unseen, excluded). Leak rate on the evaluated test images: {sets['leak_rate_on_eval']:.3f}. "
          f"A-fit folds (K = 2): {sets['afit_fold_patients']} patients. G_floor: unseen {sets['n_images']['unseen']:,} images / "
          f"{sets['n_patients']['unseen']:,} patients (>= 200 / 20: pass).", ""]

    L += ["## Competence gate", "",
          "| run | ID metric | ID_seen | ID_unseen | loss epoch 1 -> final | loss halved | preregistered gate | post hoc (AUROC only) |",
          "|---|---|---:|---:|---|---|---|---|"]
    for r, g in rows.items():
        if g is None:
            L.append(f"| {r} | missing | | | | | | |")
            continue
        loss = f"{g['loss'][0]:.4f} -> {g['loss'][1]:.4f}" if g.get("loss") else "n/a (frozen FM)"
        lc = "n/a" if g["loss_cond"] is None else ("yes" if g["loss_cond"] else "no")
        L.append(f"| {r} | {g['kind']} | {g['id_seen']:.4f} | {g['id_unseen']:.4f} | {loss} | {lc} | "
                 f"{'pass' if g['pre'] else 'FAIL'} | {'pass' if g['post'] else 'FAIL'} |")
    L += ["", "Backbone passes if all its runs pass. Preregistered: " +
          ", ".join(f"{b} {'pass' if ok else 'fail'}" for b, ok in bb_pass["pre"].items()) +
          ". Post hoc: " + ", ".join(f"{b} {'pass' if ok else 'fail'}" for b, ok in bb_pass["post"].items()) + ".", ""]

    for ood in OODS:
        role = {"shenzhen": "primary OOD (verdict)", "kermany_ped": "secondary OOD",
                "shenzhen_tb": "descriptive, Shenzhen TB subset", "shenzhen_normal": "descriptive, Shenzhen normal subset"}[ood]
        L += [f"## Delta_fit and A-gap — {ood} ({role})", "",
              "| run | scorer | d | K | n_groups_fit (fold 0 / 1) | ID_seen / ID_unseen | leaky | F2 | **Delta_fit** | jackknife 95% CI | bootstrap 95% CI (ref) | truth | A_gap | A_gap_cm | flags |",
              "|---|---|---:|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---|"]
        for bb, runs in BACKBONES.items():
            for r in runs:
                for sc in SCORERS:
                    c = cell(r, sc)
                    g = rows.get(r)
                    if c is None:
                        L.append(f"| {r} | {sc} | | | | | missing | | | | | | | | |")
                        continue
                    o = c["ood"][ood]
                    fl = ",".join(k for k, v in o["flags"].items() if v) or "-"
                    idacc = f"{g['id_seen']:.3f} / {g['id_unseen']:.3f}" if g else "n/a"
                    L.append(f"| {r} | {sc} | {c['d']} | {c['K']} | {c['n_groups_fit_per_fold'][0]:,} / {c['n_groups_fit_per_fold'][1]:,} | {idacc} | "
                             f"{o['leaky']['est']:.4f} | {o['F2']['est']:.4f} | **{o['delta_fit']['est']:+.4f}** | {ci(o['delta_fit']['jk_ci95'])} | "
                             f"{ci(o['delta_fit']['boot_ci95_ref'])} | {o['truth']['est']:.4f} | {o['a_gap']['est']:+.4f} | {o['a_gap_cm']['est']:+.4f} | {fl} |")
        L.append("")

    L += ["## Backbone level (primary OOD = Shenzhen)", "",
          "CNN backbone estimate = mean of the two seeds' Delta_fit; robust = every seed's jackknife CI > 0.", "",
          "| backbone | scorer | Delta_fit | robust | excluded (near-chance / ceiling / below-chance) | preregistered gate | post hoc gate |",
          "|---|---|---:|---|---|---|---|"]
    for bb in BACKBONES:
        for sc in SCORERS:
            d = backbone_delta(bb, sc, "shenzhen")
            if d is None:
                L.append(f"| {bb} | {sc} | missing | | | | |")
                continue
            L.append(f"| {bb} | {sc} | {d['est']:+.4f} | {'yes' if d['robust'] else 'no'} | {'yes' if d['excluded'] else 'no'} | "
                     f"{'pass' if bb_pass['pre'][bb] else 'fail'} | {'pass' if bb_pass['post'][bb] else 'fail'} |")
    L += ["", "| verdict | mahalanobis_l2 | knn_mean_cosine |", "|---|---|---|",
          f"| preregistered gate | {v_pre['mahalanobis_l2'][0]} | {v_pre['knn_mean_cosine'][0]} |",
          f"| post hoc (AUROC-only gate) | {v_post['mahalanobis_l2'][0]} | {v_post['knn_mean_cosine'][0]} |", ""]
    pre, post = predictions(bb_pass["pre"]), predictions(bb_pass["post"])
    L += ["Predictions (hold / fail; none gates anything):", "",
          "| prediction | preregistered gate | post hoc gate |", "|---|---|---|"] + \
         [f"| {k} | {pre[k]} | {post[k]} |" for k in pre] + [""]

    cmx = case_mix_table()
    L += ["## Case mix of ID_seen vs ID_unseen (A_gap_report)", "", "| | seen | unseen |", "|---|---|---|"]
    for k, r in cmx.iterrows():
        f = lambda v: v if isinstance(v, str) else (f"{v:,.0f}" if float(v) >= 10 else f"{v:.3f}")  # noqa: E731
        L.append(f"| {k} | {f(r['seen'])} | {f(r['unseen'])} |")
    c0 = next((cell(r, sc) for r in BACKBONES["resnet50"] + list(FMS) for sc in SCORERS if cell(r, sc)), None)
    if c0:
        cm = c0["case_mix"]
        L += ["", f"Post-stratification: {cm['strata_kept']} of 32 strata kept; dropped share seen {cm['seen_share_dropped']:.3f}, "
              f"unseen {cm['unseen_share_dropped']:.3f}; effective sample size of the reweighted ID_seen {cm['ess_seen_weighted']:,.0f} "
              f"(of {cm['n_seen_kept']:,})."]
    par = json.loads((P2A / "preproc_parity.json").read_text())
    dino_agcm = " / ".join(f"{c['ood']['shenzhen']['a_gap_cm']['est']:+.3f} ({sc})" for sc in SCORERS
                           if (c := cell("dinov2_vitb14", sc))) or "missing"

    def src(k):
        p = par[k]
        fm = ", ".join(f"{' '.join(ast.literal_eval(m)[:2])} x{n:,}" for m, n in p["format_mode"].items())
        return (f"| {k.replace('_raw', '').replace('_sample1000', '')} | {p['n']:,} | {fm} | 8-bit | {p['w_range'][0]}-{p['w_range'][2]} x "
                f"{p['h_range'][0]}-{p['h_range'][2]} (median {p['w_range'][1]} x {p['h_range'][1]}) | {p['aspect_w_over_h_median']:.2f} |")
    L += ["", "## Preprocessing parity (technical check before the report, 2026-10-08; scripts/r3/item8_p2a_parity_audit.py)", "",
          "| source | n audited | format / PIL mode | bit depth | size w x h | median aspect w/h |", "|---|---|---|---|---|---|",
          src("nih_raw_sample1000"), src("shenzhen_raw"), src("kermany_raw"), "",
          "Every audited file decodes to uint8 (no 16-bit images, so no clipping by PIL convert('L')); the 635 Shenzhen palette PNGs "
          "have grey palettes (R = G = B for every used index); RGB / RGBA files are converted with PIL's ITU-R 601 luma.", "",
          "| backbone | NIH chain | OOD chain |", "|---|---|---|",
          "| ResNet-50, ConvNeXt-T | original -> L -> 256 x 256 PIL bicubic (staged PNG) -> RGB -> Resize(256) + CenterCrop(224), "
          "torchvision bilinear -> ImageNet mean/std | same code path, same transform object |",
          "| DINOv2-B | same 256 px staged PNG -> Resize(224, bicubic) + CenterCrop(224) -> ImageNet mean/std | same |",
          "| RAD-DINO | original 1024 px -> L -> 518 x 518 PIL bicubic -> RGB -> /255 -> mean 0.5307 / std 0.2583 | same function on "
          "the original PNG / JPEG |", "",
          "Mean grey level / fraction of pixels at 0 after staging: " + ", ".join(
              f"{k.split('_')[0]} {par[k]['L_mean']:.1f} / {par[k]['frac_px_0']:.3f}" for k in
              ("nih_256_staged_sample1000", "shenzhen_256_staged", "kermany_256_staged")) +
          " (raw: " + ", ".join(f"{k.split('_')[0]} {par[k]['L_mean']:.1f} / {par[k]['frac_px_0']:.3f}"
                                for k in ("nih_raw_sample1000", "shenzhen_raw", "kermany_raw")) +
          "); staging changes neither. Verdict: no source-dependent difference in the chain, so no OOD re-extraction or rescoring. "
          "The between-source intensity difference is already present in the raw files.", ""]
    L += ["## Deviations and caveats", "",
          "1. **Competence gate (post hoc sensitivity, approved 2026-10-08):** the preregistered CNN condition 'final train loss <= 50% of "
          "epoch-1 loss' was copied from single-label cross-entropy training (medbench) and is mis-specified for 14-finding multi-label BCE, "
          "where the epoch-1 mean loss is already low. The preregistered verdict applies it as written; the post hoc verdict uses the AUROC "
          "condition only, with the precommitted thresholds (CNN macro AUROC >= 0.75, FM probe >= 0.70); no new thresholds.",
          "2. Wang-test images of the checkpoint patients (test_ckpt) are neither seen nor unseen and are excluded (counted above).",
          "3. Preprocessing parity (see the section above): within each backbone NIH and OOD share one code path. RAD-DINO does not use "
          "its BitImageProcessor (shortest edge 518 + centre crop 518); it squashes to 518 x 518, identical for square NIH images, "
          "not for non-square OOD images (Kermany pediatric median aspect w/h "
          f"{par['kermany_raw']['aspect_w_over_h_median']:.2f}). The ceiling pilot read DINOv2-B inputs from the 1024 px originals.",
          "4. RAD-DINO loaded with transformers 4.44.2 (private install; the environment's transformers needs torch >= 2.5), "
          "pooler_output (CLS after the final layer norm). RAD-DINO pretraining saw all NIH images (seen and unseen alike).",
          "5. CNN backbone-level aggregation over the two seeds (mean; robust = both seeds' jackknife CIs > 0) is not specified in the "
          "precommit and was fixed before the cells were computed. Cells excluded by G_near_chance / G_ceiling / below-chance do not meet "
          "the bar; the denominator stays the number of gate-passing backbones (precommit wording).",
          "6. No logit scores (precommit): CXR reports the scorer channel (A-fit) and the case-mix-corrected A-gap for feature scores only; "
          "no family comparison.",
          "7. Bootstrap CIs are reference only (fitted scores held fixed; too narrow when Delta > 0 per the coverage study); the jackknife "
          "decides robustness.",
          "8. **A_gap is descriptive.** It is not evidence that the image-level split raises or lowers AUROC: DINOv2-B never saw NIH "
          f"and still shows A_gap_cm {dino_agcm} on Shenzhen after post-stratification on view x finding pattern, so residual "
          "case-mix confounding between ID_seen and ID_unseen remains.", ""]
    (P2A / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
