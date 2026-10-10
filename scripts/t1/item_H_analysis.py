#!/usr/bin/env python3
"""T1 item H analysis (H1-H3): h1_nulls.csv, h2_placebo.csv, h2_lobo.csv, h3_coverage.csv, tests.csv, verdict.json.

S9 is read per null control (H-N1, H-N2, H-N3): triggered when more than 5 % of the evaluable cells of a control have
|Delta| > 0.02 with the 95 % CI excluding 0. H-ii: Camelyon LOBO over backbones (A1 unit table: mean over the cells of a
backbone), placebo R2_oos vs B0 <= 0 and the lower 95 % bootstrap bound (default_rng(801), 10,000) of
mean(|err_placebo| - |err_real|) > 0. H-iii: coverage of rho_w and dQ_pred >= 0.93 in every H3 design cell.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
from item_A_analysis import boot_mean_ci, lobo, r2_oos, write_csv  # noqa: E402

OUT = CM.RES / "H"
RAW = OUT / "raw"


def load():
    rows = {}
    for f in sorted(RAW.glob("h_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows[r["unit"]] = r
    return rows


def manifest_ok():
    p = CM.RES / "I" / "MANIFEST_OK"
    return p.exists() and p.read_text().strip().lower() in ("true", "1", "ok", "yes")


def main():
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_H.json"))
    req = add["work_lists"][0]["execution_order"]
    R = load()
    tests = []

    # ---------------- H1
    h1 = []
    s9 = {}
    for ctl in ("hn1", "hn2", "hn3"):
        units = [u for u in req if u.startswith(ctl + "|")]
        ev, flagged, missing = 0, 0, 0
        for u in units:
            if u not in R:
                missing += 1
                h1.append(dict(unit=u, control=ctl, status="not run: not reached"))
                continue
            r = R[u]
            row = dict(unit=u, control=ctl, cell=r["cell"], dataset=r["dataset"], backbone=r["backbone"], status=r["status"],
                       delta=r.get("delta"), se=r.get("se"), ci_lo=r.get("ci_lo"), ci_hi=r.get("ci_hi"), flag_s9=r.get("flag_s9"))
            if ctl == "hn1":
                fo = [f for f in r["folds"] if "rho_w" in f]
                for k in ("rho_w", "rho_raw", "rho_anova", "s_logo", "dq_pred", "dq_meas"):
                    row[f"mean_{k}"] = float(np.mean([f[k] for f in fo])) if fo else None
            h1.append(row)
            if r["status"] == "ok":
                ev += 1
                flagged += bool(r["flag_s9"])
        frac = flagged / ev if ev else float("nan")
        s9[ctl] = dict(evaluable=ev, flagged=flagged, missing=missing, frac=frac, triggered=bool(ev and frac > 0.05))
    write_csv(OUT / "h1_nulls.csv", h1)
    trig = [c for c in s9 if s9[c]["triggered"]]
    miss = [c for c in s9 if s9[c]["missing"] or not s9[c]["evaluable"]]
    H_i = "fail" if trig else ("not evaluable" if miss else "pass")
    for c, v in s9.items():
        tests.append(dict(test=f"S9 in H-{c[1:].upper()}: share of cells with |Delta| > 0.02 and CI excluding 0",
                          status="fail" if v["triggered"] else ("not evaluable" if v["missing"] else "pass"),
                          value=f"{v['flagged']}/{v['evaluable']} ({v['frac']:.3f}); {v['missing']} not run", reference="<= 0.05",
                          **{"pass": not v["triggered"] and not v["missing"]}))
    tests.append(dict(test="H-i: S9 not triggered in H-N1-H-N3", status=H_i, value=",".join(trig) or "none triggered",
                      reference="not triggered", **{"pass": H_i == "pass"}))
    hn1_ok = [r for r in h1 if r["control"] == "hn1" and r.get("status") == "ok"]
    null_floor = {k: (float(np.quantile([r[f"mean_{k}"] for r in hn1_ok], 0.95)) if hn1_ok else None) for k in ("rho_w",)}
    null_floor["abs_dq_pred_q95"] = float(np.quantile([abs(r["mean_dq_pred"]) for r in hn1_ok], 0.95)) if hn1_ok else None
    null_floor["abs_delta_q95"] = float(np.quantile([abs(r["delta"]) for r in hn1_ok], 0.95)) if hn1_ok else None

    # ---------------- H2
    h2u = [u for u in req if u.startswith("h2|")]
    h2 = []
    for u in h2u:
        if u not in R:
            h2.append(dict(unit=u, status="not run: not reached"))
            continue
        r = R[u]
        h2.append(dict(unit=u, cell=r["cell"], dataset=r["dataset"], backbone=r["backbone"], family=r["family"], status="ok",
                       delta_meas=r["delta_meas"], real_p_delta=r["real_p_delta"], placebo_p_delta=r["placebo_p_delta"],
                       real_rho_w=float(np.mean([f["real_rho_w"] for f in r["folds"]])),
                       placebo_rho_w=float(np.mean([f["placebo_rho_w_mean"] for f in r["folds"]])),
                       real_dq_pred=float(np.mean([f["real_dq_pred"] for f in r["folds"]])),
                       placebo_dq_pred=float(np.mean([f["placebo_dq_pred_mean"] for f in r["folds"]]))))
    write_csv(OUT / "h2_placebo.csv", h2)
    if not manifest_ok():
        H_ii, h2info = "not evaluable", dict(reason="not run: manifest mismatch (S11)")
    elif any(r["status"] != "ok" for r in h2):
        H_ii, h2info = "not evaluable", dict(reason="H2 incomplete", missing=sum(r["status"] != "ok" for r in h2))
    else:
        cam = [r for r in h2 if r["dataset"] == "camelyon"]
        bbs = sorted({r["backbone"] for r in cam})
        tab = [dict(backbone=b, **{k: float(np.mean([r[k] for r in cam if r["backbone"] == b]))
                                   for k in ("delta_meas", "real_p_delta", "placebo_p_delta")}) for b in bbs]
        y = np.array([t["delta_meas"] for t in tab])
        pr = lobo(y, [t["real_p_delta"] for t in tab])
        pp = lobo(y, [t["placebo_p_delta"] for t in tab])
        p0 = lobo(y, None)
        r2r, r2p = r2_oos(y, pr, p0), r2_oos(y, pp, p0)
        D = np.abs(y - pp) - np.abs(y - pr)
        lo, hi, pb = boot_mean_ci(D, 801)
        for t, a, b in zip(tab, pr, pp):
            t.update(pred_real=float(a), pred_placebo=float(b))
        write_csv(OUT / "h2_lobo.csv", tab)
        ok = (r2p <= 0) and (lo > 0)
        H_ii = "pass" if ok else "fail"
        h2info = dict(n_backbones=len(bbs), r2_oos_real=r2r, r2_oos_placebo=r2p, mean_D=float(D.mean()), D_ci95=[lo, hi], p_boot=pb)
        tests.append(dict(test="H-ii a: placebo LOBO R2_oos vs B0 <= 0 (Camelyon, 13 backbones)", status="pass" if r2p <= 0 else "fail",
                          value=f"{r2p:.3f} (real {r2r:.3f})", reference="<= 0", **{"pass": r2p <= 0}))
        tests.append(dict(test="H-ii b: real - placebo, LB95 of mean(|e_placebo| - |e_real|) > 0", status="pass" if lo > 0 else "fail",
                          value=f"{D.mean():.4f} [{lo:.4f}, {hi:.4f}]", reference="LB > 0", **{"pass": lo > 0}))
    tests.append(dict(test="H-ii: placebo skill <= B0 and real - placebo LB > 0", status=H_ii, value=json.dumps(h2info),
                      reference="both", **{"pass": H_ii == "pass"}))

    # ---------------- H3
    h3u = [u for u in req if u.startswith("h3|")]
    h3, below, nr = [], [], []
    for u in h3u:
        if u not in R:
            h3.append(dict(unit=u, status="not run: not reached"))
            nr.append(u)
            continue
        r = R[u]
        row = {k: v for k, v in r.items() if k not in ("params", "precommit")}
        row.setdefault("status", "ok")
        row.update({k: r["params"].get(k) for k in ("dataset", "fold", "n_g_from")})
        h3.append(row)
        if row["status"] != "ok":
            nr.append(u)
        elif not (r["coverage_rho_w"] >= 0.93 and r["coverage_dq_pred"] >= 0.93):
            below.append(u)
    write_csv(OUT / "h3_coverage.csv", h3)
    H_iii = "fail" if below else ("not evaluable" if nr else "pass")
    tests.append(dict(test="H-iii: jackknife CI coverage >= 0.93 (rho_w and dQ_pred) in every H3 design cell", status=H_iii,
                      value=f"{len(below)} cells below 0.93, {len(nr)} not run, of {len(h3u)}", reference="every cell >= 0.93",
                      **{"pass": H_iii == "pass"}))

    if H_i == "fail":
        verdict = "NO-GO"
    elif H_i == "not evaluable":
        verdict = "INCONCLUSIVE"
    elif H_ii == "pass" and H_iii == "pass":
        verdict = "GO"
    else:
        verdict = "PARTIAL"
    with open(OUT / "tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["test", "status", "value", "reference", "pass"])
        w.writeheader()
        w.writerows(tests)
    v = dict(item="H", verdict=verdict, H_i=H_i, H_ii=H_ii, H_iii=H_iii, s9=s9, h2=h2info, h3_below=below, h3_not_run=nr,
             hn1_null_floor=null_floor, manifest_ok=manifest_ok(),
             gpu_seconds=float(sum(r.get("seconds", 0.0) for r in R.values())),
             rule="H-i fail -> NO-GO; H-i not evaluable -> INCONCLUSIVE; all pass -> GO; else PARTIAL")
    CM.atomic_write_text(OUT / "verdict.json", json.dumps(v, indent=1, default=str))
    print(json.dumps({k: v[k] for k in ("verdict", "H_i", "H_ii", "H_iii")}), "below", len(below), "not run", len(nr))


if __name__ == "__main__":
    main()
