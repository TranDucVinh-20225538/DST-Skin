#!/usr/bin/env python3
"""T1 item D analysis: d1_configs.csv, d1_tau.json, d2_search.csv, d2_worst10.csv, d3_real.csv, d4_null.csv,
d5_se_posthoc.csv, tests.csv, verdict.json.

D1: tau maximises Youden's J of the flag T < tau for omega = 1 (positive) vs omega = 0 over every replicate of the TRAIN
configs (d in {32, 512, 2048}), primary T on L2-normalised features; frozen and applied to the TEST configs (d in {128, 1024}):
sensitivity = P(flag | omega >= 0.25) and specificity = P(no flag | omega = 0), both over the replicates of the material TEST
configs; AUC of T per config (omega = 0 vs 1); size of the naive z test at rho = 0 (|T| > 1.96).
D2: falsifier := Delta_full >= 0.02 and AUC(T) < 0.8 among the 3,000 random and every CMA-ES evaluation.
D3: AUC(T_seen vs T_unseen) per Camelyon backbone over cells x folds x 50 subsamples (P(T_unseen > T_seen) + 1/2 ties).
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import scorer64 as S64  # noqa: E402
from item_A_analysis import boot_mean_ci, lobo, write_csv  # noqa: E402

OUT = CM.RES / "D"
RAW = OUT / "raw"
TRAIN_D, TEST_D = {32, 512, 2048}, {128, 1024}


def jl(pattern):
    rows = {}
    for p in sorted(glob.glob(str(RAW / pattern))):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows[r["unit"]] = r
    return rows


def auc(neg, pos):
    """P(pos > neg) + 1/2 ties."""
    return S64.auroc_id_pos(np.asarray(pos, float), np.asarray(neg, float))


def youden_tau(t0, t1):
    """Flag := T < tau; positives omega = 1 (t1), negatives omega = 0 (t0). Returns (tau, J, sens, spec)."""
    t0, t1 = np.sort(t0), np.sort(t1)
    cand = np.unique(np.r_[t0, t1])
    mids = np.r_[cand[0] - 1, (cand[1:] + cand[:-1]) / 2, cand[-1] + 1]
    sens = np.searchsorted(t1, mids, side="left") / len(t1)
    spec = 1 - np.searchsorted(t0, mids, side="left") / len(t0)
    j = sens + spec - 1
    i = int(np.argmax(j))
    return float(mids[i]), float(j[i]), float(sens[i]), float(spec[i])


def d1(tests):
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_D.json"))
    req = [u for u in add["work_lists"][0]["execution_order"] if u.startswith("d1|")]
    R = jl("d1_*.jsonl")
    missing = [u for u in req if u not in R]
    rows, out = [], {}
    for u in req:
        if u not in R:
            continue
        r = R[u]
        p = r["params"]
        rows.append(dict(unit=u, **{k: p[k] for k in ("d", "rho", "G", "n", "law", "block")}, N=r["N"], delta_lw=r["delta_lw_mean"],
                         delta_knn50=r["delta_knn50_mean"], delta_full=r["delta_full"], material=r["material"],
                         auc_T_norm=r["auc_T_norm"], auc_T_raw=r["auc_T_raw"], rho0_size_norm=r["rho0_size_norm"],
                         rho0_size_raw=r["rho0_size_raw"], **{f"T_norm_w{w:g}_mean": r[f"T_norm_w{w:g}_mean"] for w in (0, 0.25, 0.5, 1)}))
    write_csv(OUT / "d1_configs.csv", rows)
    for tag in ("norm", "raw"):
        tr = [R[u] for u in req if u in R and R[u]["params"]["d"] in TRAIN_D]
        te = [R[u] for u in req if u in R and R[u]["params"]["d"] in TEST_D]
        t0 = np.concatenate([r[f"T_{tag}_w0"] for r in tr]) if tr else np.array([])
        t1 = np.concatenate([r[f"T_{tag}_w1"] for r in tr]) if tr else np.array([])
        tau, J, se_tr, sp_tr = youden_tau(t0, t1) if len(t0) else (float("nan"),) * 4
        mat = [r for r in te if r["material"]]
        pos = np.concatenate([np.r_[r[f"T_{tag}_w0.25"], r[f"T_{tag}_w0.5"], r[f"T_{tag}_w1"]] for r in mat]) if mat else np.array([])
        neg = np.concatenate([r[f"T_{tag}_w0"] for r in mat]) if mat else np.array([])
        sens = float(np.mean(pos < tau)) if len(pos) else float("nan")
        spec = float(np.mean(neg >= tau)) if len(neg) else float("nan")
        out[tag] = dict(tau=tau, train_J=J, train_sens=se_tr, train_spec=sp_tr, n_train_configs=len(tr), n_test_configs=len(te),
                        n_test_material=len(mat), test_sens=sens, test_spec=spec,
                        test_auc_median=float(np.median([r[f"auc_T_{tag}"] for r in te])) if te else float("nan"),
                        test_auc_material_min=float(min([r[f"auc_T_{tag}"] for r in mat])) if mat else float("nan"),
                        size_rho0_mean=float(np.mean([r[f"rho0_size_{tag}"] for r in te])) if te else float("nan"),
                        sens_by_omega={f"{w:g}": float(np.mean(np.concatenate([r[f"T_{tag}_w{w:g}"] for r in mat]) < tau)) if mat else None
                                       for w in (0.25, 0.5, 1)})
    out["training_set"] = "every replicate (16 per config) of the TRAIN configs d in {32, 512, 2048}, omega = 0 vs omega = 1"
    out["missing_units"] = len(missing)
    CM.atomic_write_text(OUT / "d1_tau.json", json.dumps(out, indent=1))
    o = out["norm"]
    if missing:
        st = "not evaluable"
    else:
        st = "pass" if (o["test_sens"] >= 0.90 and o["test_spec"] >= 0.90) else "fail"
    tests.append(dict(test="D-i: TEST sensitivity >= 0.90 and specificity >= 0.90 on material configs (T normalised, frozen tau)",
                      status=st, value=f"sens {o['test_sens']:.3f}, spec {o['test_spec']:.3f} (tau {o['tau']:.3f}; {o['n_test_material']} material TEST configs)"
                      + (f"; {len(missing)} D1 units missing" if missing else ""), reference=">= 0.90 both", **{"pass": st == "pass"}))
    tests.append(dict(test="D1 report: naive z-score size at rho = 0 (|T| > 1.96), TEST configs, normalised", status="report-only",
                      value=f"{o['size_rho0_mean']:.3f}", reference="0.05 nominal", **{"pass": "na"}))
    tests.append(dict(test="D1 secondary: raw-feature T, TEST sens / spec", status="report-only",
                      value=f"sens {out['raw']['test_sens']:.3f}, spec {out['raw']['test_spec']:.3f}", reference=">= 0.90", **{"pass": "na"}))
    return st, out


def d2(tests):
    rnd = jl("d2r_*.jsonl")
    cma = jl("d2_cma.jsonl")
    done = (RAW / "d2_cma.done").exists() and len(rnd) >= 3000
    rows = []
    for src, R in (("random", rnd), ("cma", cma)):
        for u, r in R.items():
            p = {k: v for k, v in r["params"].items() if k != "x"}
            rows.append(dict(unit=u, source=src, **p, N=r["N"], delta_lw=r["delta_lw_mean"], delta_knn50=r["delta_knn50_mean"],
                             delta_full=r["delta_full"], material=r["material"], auc_T_norm=r["auc_T_norm"], auc_T_raw=r["auc_T_raw"]))
    write_csv(OUT / "d2_search.csv", rows)
    worst = sorted([r for r in rows if r["material"]], key=lambda r: r["auc_T_norm"])[:10]
    write_csv(OUT / "d2_worst10.csv", worst)
    fals = [r for r in rows if r["material"] and r["auc_T_norm"] < 0.8]
    st = "fail" if fals else ("pass" if done else "not evaluable")
    tests.append(dict(test="D-ii: no configuration with Delta_full >= 0.02 and AUC(T) < 0.8 (3,000 random + CMA-ES)", status=st,
                      value=f"{len(fals)} falsifiers among {len(rows)} evaluations ({len(rnd)} random, {len(cma)} CMA-ES)"
                      + ("" if done else "; search incomplete"), reference="none found", **{"pass": st == "pass"}))
    return st, fals


def real(tests):
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_D.json"))
    req = [u for u in add["work_lists"][0]["execution_order"] if u.startswith(("d3|", "d4|"))]
    R = jl("dreal_*.jsonl")
    missing = [u for u in req if u not in R]
    d3rows, d4rows, d5rows = [], [], []
    for u in req:
        if u not in R:
            continue
        r = R[u]
        for f in r["folds"]:
            base = dict(unit=u, cell=r["cell"], dataset=r["dataset"], backbone=r["backbone"], family=r["family"], fold=f["fold"],
                        status=f["status"])
            if f["status"] != "ok":
                (d3rows if u.startswith("d3|") else d4rows).append(base)
                continue
            if u.startswith("d3|"):
                row = dict(base, N_fit=f["N_fit"], n_seen=f["n_seen"], n_unseen=f["n_unseen"])
                for tag in ("norm", "raw"):
                    row[f"T_seen_{tag}_mean"] = float(np.mean(f[f"T_seen_{tag}"]))
                    row[f"T_unseen_{tag}_mean"] = float(np.mean(f[f"T_unseen_{tag}"]))
                    row[f"auc_{tag}"] = auc(f[f"T_seen_{tag}"], f[f"T_unseen_{tag}"])
                    for w in (0, 0.25, 0.5, 0.75, 1):
                        row[f"T_mix_{tag}_w{w:g}_mean"] = float(np.mean(f[f"T_mix_{tag}_w{w:g}"]))
                d3rows.append(row)
                d5rows.append(dict(base, naive_se=f["d5_naive_se"], cluster_boot_se=f["d5_cluster_boot_se"], ratio=f["d5_ratio"],
                                   status_rule8="post-hoc"))
            else:
                d4rows.append(dict(base, n_pseudo_fit=f["n_pseudo_fit"], n_pseudo_eval=f["n_pseudo_eval"],
                                   T_null_norm_mean=float(np.mean(f["T_null_norm"])), T_null_norm_sd=float(np.std(f["T_null_norm"], ddof=1)),
                                   T_null_raw_mean=float(np.mean(f["T_null_raw"])),
                                   permuted_labels_T_identical=f["permuted_labels_T_identical"]))
    write_csv(OUT / "d3_real.csv", d3rows)
    write_csv(OUT / "d4_null.csv", d4rows)
    write_csv(OUT / "d5_se_posthoc.csv", d5rows)
    # D-iii: Camelyon per backbone, pooled over cells x folds x subsamples
    cam = {}
    for u in req:
        if u.startswith("d3|camelyon") and u in R:
            for f in R[u]["folds"]:
                if f["status"] == "ok":
                    c = cam.setdefault(R[u]["backbone"], dict(s=[], u=[]))
                    c["s"] += f["T_seen_norm"]
                    c["u"] += f["T_unseen_norm"]
    per_bb = {b: auc(v["s"], v["u"]) for b, v in sorted(cam.items())}
    k = sum(a >= 0.90 for a in per_bb.values())
    cam_missing = [u for u in missing if u.startswith("d3|camelyon")]
    n_bb = len({u.split("|")[1].rsplit("_s", 1)[0] for u in req if u.startswith("d3|camelyon")})
    if cam_missing or len(per_bb) < n_bb:
        st = "fail" if n_bb - (len(per_bb) - k) < 11 else "not evaluable"
    else:
        st = "pass" if k >= 11 else "fail"
    tests.append(dict(test="D-iii: AUC(T_seen vs T_unseen) >= 0.90 in >= 11 of 13 Camelyon backbones", status=st,
                      value=f"{k}/{len(per_bb)}" + (f"; {len(cam_missing)} Camelyon D3 cells missing" if cam_missing else ""),
                      reference=">= 11/13", **{"pass": st == "pass"}))
    # report: T_unseen vs Delta_meas (LOBO over Camelyon backbones) vs ICC-only (rho_w)
    rep = {}
    try:
        a = list(csv.DictReader(open(CM.RES / "A" / "cells.csv")))
        b = list(csv.DictReader(open(CM.RES / "B" / "cells_knn.csv")))
        bbs = sorted(per_bb)
        tu = {bb: float(np.mean([r["T_unseen_norm_mean"] for r in d3rows if r["backbone"] == bb and r["dataset"] == "camelyon"
                                 and r["status"] == "ok"])) for bb in bbs}
        for lab, src, ycol in (("maha", a, "delta_meas_fold"), ("knn50", b, "delta_k50")):
            y = np.array([np.mean([float(r[ycol]) for r in src if r["dataset"] == "camelyon" and r["backbone"] == bb]) for bb in bbs])
            icc = np.array([np.mean([float(r["rho_w"]) for r in a if r["dataset"] == "camelyon" and r["backbone"] == bb]) for bb in bbs])
            pt, pi = lobo(y, [tu[bb] for bb in bbs]), lobo(y, icc)
            D = np.abs(y - pi) - np.abs(y - pt)
            lo, hi, _ = boot_mean_ci(D, 401)
            from scipy.stats import spearmanr
            rep[lab] = dict(spearman_T_unseen_vs_delta=float(spearmanr([tu[bb] for bb in bbs], y).correlation),
                            mae_T=float(np.abs(y - pt).mean()), mae_icc=float(np.abs(y - pi).mean()), mean_D=float(D.mean()), D_ci95=[lo, hi])
    except Exception as e:  # noqa: BLE001
        rep["error"] = repr(e)
    d5 = [r["ratio"] for r in d5rows]
    tests.append(dict(test="D5 (post-hoc): cluster-bootstrap SE / naive SE of the mean NN^2 difference", status="post-hoc",
                      value=f"median {np.median(d5):.2f}, IQR [{np.quantile(d5, .25):.2f}, {np.quantile(d5, .75):.2f}] over {len(d5)} cell-folds" if d5 else "",
                      reference="1 = naive calibrated", **{"pass": "na"}))
    return st, dict(per_backbone_auc=per_bb, missing=len(missing), T_unseen_vs_delta=rep)


def main():
    tests = []
    s1, o1 = d1(tests)
    s2, f2 = d2(tests)
    s3, o3 = real(tests)
    if s1 == "fail" or s3 == "fail":
        verdict = "NO-GO"
    elif "not evaluable" in (s1, s2, s3):
        verdict = "INCONCLUSIVE"
    elif s2 == "pass":
        verdict = "GO"
    else:
        verdict = "PARTIAL"
    with open(OUT / "tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["test", "status", "value", "reference", "pass"])
        w.writeheader()
        w.writerows(tests)
    gpu = sum(r.get("seconds", 0.0) for pat in ("d1_*.jsonl", "d2r_*.jsonl", "d2_cma.jsonl", "dreal_*.jsonl") for r in jl(pat).values())
    v = dict(item="D", verdict=verdict, D_i=s1, D_ii=s2, D_iii=s3, d1=o1, n_falsifiers=len(f2), d3=o3, gpu_seconds=gpu,
             rule="D-i or D-iii fail -> NO-GO; a criterion not evaluable -> INCONCLUSIVE (rule 18); all pass -> GO; D-i, D-iii pass and D-ii falsifier -> PARTIAL")
    CM.atomic_write_text(OUT / "verdict.json", json.dumps(v, indent=1, default=str))
    print(json.dumps({k: v[k] for k in ("verdict", "D_i", "D_ii", "D_iii", "n_falsifiers")}))


if __name__ == "__main__":
    main()
