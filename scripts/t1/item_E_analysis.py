#!/usr/bin/env python3
"""T1 item E analysis: e1_cs.csv, e1_estimands.csv, e2_recovery.csv, e3_real_recovery.csv, e4_semisynth.csv, tests.csv,
verdict.json (E-i, E-ii, E-iii and the verdict with the rule-18 status words of the order)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

OUT = CM.RES / "E"
RAW = OUT / "raw"
ALPHA, REPS = 0.05, 10000
BOUND = ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / REPS)
METHODS_E1 = ("N1", "N2", "N3", "G1")
METHODS_E4 = ("N1", "N2", "G1")
EPS = (0.05, 0.1)


def jl(pattern):
    rows = {}
    for f in sorted(RAW.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows[r["unit"]] = r
    return rows


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def method_rows(uid, r, methods, base):
    out = []
    for m in methods:
        row = dict(base, unit=uid, method=m, status=r["status"], targeted=r["targeted"][m],
                   theta_group=r["theta_group"], theta_size=r["theta_size"], theta_size_se=r["theta_size_se"],
                   miscover_theta_group=r[f"{m}|miscover|theta_group"], miscover_theta_group_se=r[f"{m}|miscover|theta_group|se"],
                   miscover_theta_size=r[f"{m}|miscover|theta_size"], miscover_theta_size_se=r[f"{m}|miscover|theta_size|se"],
                   width_at_100=r[f"{m}|width_at_100"])
        for e in EPS:
            row[f"gap_over_eps{e:g}"] = r["gap_over_eps"][f"{e:g}"]
            row[f"p_cert_eps{e:g}"] = r[f"{m}|cert|eps{e:g}"]
            row[f"groups_to_cert_eps{e:g}"] = r[f"{m}|groups_to_cert|eps{e:g}"]
            for k in ("theta_group", "theta_size"):
                row[f"false_cert_eps{e:g}_{k}"] = r[f"{m}|false_cert|eps{e:g}|{k}"]
                row[f"false_cert_given_cert_eps{e:g}_{k}"] = r[f"{m}|false_cert_given_cert|eps{e:g}|{k}"]
        out.append(row)
    return out


def main():
    add = json.load(open(CM.RES / "PRECOMMIT_T1_addendum_E.json"))
    P = add["unit_params"]
    req = [u for wl in add["work_lists"] for u in wl["execution_order"]]
    g = jl("egpu_*.jsonl")
    e2 = jl("e2_*.jsonl")
    e3 = jl("e3.jsonl")
    tests, notes = [], []

    # ---------------- E1
    e1_rows, est_rows = [], []
    for u in [u for u in req if u.startswith("e1|")]:
        p = P[u]
        if u not in g:
            e1_rows.append(dict(unit=u, regime=p["regime"], size_mechanism=p["size"], outcome=p["outcome"], n=p["n"], rho=p["rho"],
                                status="not run: not reached"))
            continue
        r = g[u]
        base = dict(regime=p["regime"], size_mechanism=p["size"], outcome=p["outcome"], n=p["n"], rho=p["rho"], deff=r["deff"])
        e1_rows += method_rows(u, r, METHODS_E1, base)
        est_rows.append(dict(unit=u, **base, theta_group=r["theta_group"], theta_size=r["theta_size"], theta_size_se=r["theta_size_se"],
                             gap=abs(r["theta_size"] - r["theta_group"]), status=r["status"],
                             **{f"gap_over_eps{e:g}": r["gap_over_eps"][f"{e:g}"] for e in EPS}))
    write_csv(OUT / "e1_cs.csv", e1_rows)
    write_csv(OUT / "e1_estimands.csv", est_rows)

    # ---------------- E4
    e4_rows = []
    for u in [u for u in req if u.startswith("e4|")]:
        p = P[u]
        reg = {"a": "a", "ap": "a'", "b": "b"}[p["regime"]]
        if u not in g:
            e4_rows.append(dict(unit=u, backbone=p["backbone"], regime=reg, n=p.get("n"), status="not run: not reached"))
            continue
        r = g[u]
        base = dict(backbone=p["backbone"], regime=reg, n=p.get("n"), spearman_m_N=r.get("spearman_m_N"),
                    spearman_ci95=json.dumps(r.get("spearman_ci95")))
        e4_rows += method_rows(u, r, METHODS_E4, base)
    write_csv(OUT / "e4_semisynth.csv", e4_rows)

    # ---------------- E-i
    def g1_check(rows, kind):
        fails, nr, ok = [], [], 0
        units = {}
        for row in rows:
            units.setdefault(row["unit"], []).append(row)
        for u, rs in units.items():
            st = rs[0]["status"]
            if st != "ok":
                nr.append((u, st))
                continue
            gr = [x for x in rs if x["method"] == "G1"][0]
            if gr["miscover_theta_group"] > BOUND:
                fails.append((u, gr["miscover_theta_group"]))
            else:
                ok += 1
        return fails, nr, ok
    f1a, n1a, o1a = g1_check([r for r in e1_rows if r["regime"] == "a"], "E1a")
    f1b, n1b, o1b = g1_check([r for r in e1_rows if r["regime"] == "b"], "E1b")
    f4 = {}
    for reg in ("a", "a'", "b"):
        f4[reg] = g1_check([r for r in e4_rows if r["regime"] == reg], "E4" + reg)
    allf = f1a + f1b + sum([f4[k][0] for k in f4], [])
    alln = n1a + n1b + sum([f4[k][1] for k in f4], [])
    E_i = "fail" if allf else ("not evaluable" if alln else "pass")
    for lab, (f, n, o) in [("E1 regime (a)", (f1a, n1a, o1a)), ("E1 regime (b)", (f1b, n1b, o1b))] + \
            [(f"E4 regime ({k})", f4[k]) for k in ("a", "a'", "b")]:
        tests.append(dict(test=f"E-i component: G1 ever-miscover of theta_group <= {BOUND:.4f}, {lab}",
                          status="fail" if f else ("not evaluable" if n else "pass"),
                          value=f"{o} pass, {len(f)} fail, {len(n)} not run", reference=f"every cell <= {BOUND:.4f}",
                          pass_=(not f and not n)))
    tests.append(dict(test="E-i: G1 valid for theta_group in every E1 cell (a, b) and every E4 backbone x regime", status=E_i,
                      value=f"{len(allf)} fail, {len(alln)} not run", reference="all pass", pass_=E_i == "pass"))

    # ---------------- E-ii (regime a, DEFF >= 2, N1 against theta_size)
    def n1_frac(rows, method="N1"):
        cells = [r for r in rows if r["method"] == method and r["status"] == "ok" and r["deff"] >= 2]
        k = sum(r["miscover_theta_size"] > BOUND for r in cells)
        return k, len(cells)
    a_rows = [r for r in e1_rows if r.get("regime") == "a" and "method" in r]
    k, n = n1_frac(a_rows)
    n_req = sum(1 for u in req if u.startswith("e1|a|") and 1 + (P[u]["n"] - 1) * P[u]["rho"] >= 2)
    if n < n_req:
        E_ii = "not evaluable"
    else:
        E_ii = "pass" if k >= 0.9 * n else "fail"
    tests.append(dict(test="E-ii: N1 ever-miscover > bound in >= 90 % of regime-(a) E1 cells with DEFF >= 2 (wording only)",
                      status=E_ii, value=f"{k}/{n}" + (f" (required {n_req})" if n < n_req else ""), reference=">= 90 %",
                      pass_=E_ii == "pass"))
    ke, ne = n1_frac([r for r in a_rows if r["size_mechanism"] == "constant"])
    tests.append(dict(test="E-ii report-only: equal-size subset of regime (a)", status="report-only", value=f"{ke}/{ne}",
                      reference=">= 90 % (same reading)", pass_="na"))
    k2, n2 = n1_frac(a_rows, "N2")
    tests.append(dict(test="E-ii report-only: N2 ever-miscover > bound, regime (a), DEFF >= 2", status="report-only",
                      value=f"{k2}/{n2}", reference="(N2 valid = baseline)", pass_="na"))

    # ---------------- E2
    e2_rows = []
    for u in [u for u in req if u.startswith("e2|")]:
        p = P[u]
        if u not in e2:
            e2_rows.append(dict(unit=u, **p, status="not run: not reached"))
            continue
        r = e2[u]
        row = dict(unit=u, **p, status="ok", lemma_stat=r["lemma_stat"], lemma_pred=r["lemma_pred"], reps=r["reps"])
        for k_ in [k for k in r if k.startswith(("M1_", "M2_", "cov_"))]:
            row[k_] = r[k_]
        for m in ("M1", "M2"):
            row[f"{m}_silent_failure"] = bool(r[f"{m}_khat_over_k"] >= 2 and r[f"{m}_cov"] < 0.90)
        e2_rows.append(row)
    write_csv(OUT / "e2_recovery.csv", e2_rows)
    e2_done = all(r["status"] == "ok" for r in e2_rows)
    viol = [r for r in e2_rows if r["status"] == "ok" and r["lemma_pred"] and r["M1_cov"] < 0.90]

    # ---------------- E3
    e3_rows = []
    for u in [u for u in req if u.startswith("e3|")]:
        if u not in e3:
            e3_rows.append(dict(unit=u, cell=P[u]["cell"], status="not run: not reached"))
            continue
        r = e3[u]
        for f in r["folds"]:
            e3_rows.append(dict(unit=u, cell=r["cell"], dataset=r["dataset"], backbone=r["backbone"], status="ok",
                                match_both_folds=r["match_both_folds"], **f))
    write_csv(OUT / "e3_real_recovery.csv", e3_rows)
    e3_cells = [e3[u] for u in req if u.startswith("e3|") and u in e3]
    e3_done = len(e3_cells) == sum(1 for u in req if u.startswith("e3|"))
    match = float(np.mean([c["match_both_folds"] for c in e3_cells])) if e3_cells else float("nan")
    if viol or (e3_done and match < 0.8):
        E_iii = "fail"
    elif not e2_done or not e3_done:
        E_iii = "not evaluable"
    else:
        E_iii = "pass"
    tests.append(dict(test="E-iii (E2): no config with rho_f sqrt(d/2) >= 3 and G >= 60 has M1 CR1 coverage < 0.90",
                      status="fail" if viol else ("pass" if e2_done else "not evaluable"),
                      value=f"{len(viol)} violations among {sum(1 for r in e2_rows if r.get('lemma_pred'))} predicted-recoverable configs"
                            + ("" if e2_done else " (incomplete)"), reference="0 violations", pass_=(not viol and e2_done)))
    tests.append(dict(test="E-iii (E3): lemma-predicted recoverability matches observed (both folds) in >= 80 % of cells",
                      status=("fail" if match < 0.8 else "pass") if e3_done else "not evaluable",
                      value=f"{match:.3f} ({sum(c['match_both_folds'] for c in e3_cells)}/{len(e3_cells)})", reference=">= 0.80",
                      pass_=bool(e3_done and match >= 0.8)))
    tests.append(dict(test="E-iii", status=E_iii, value="", reference="E2 and E3 parts pass", pass_=E_iii == "pass"))

    # ---------------- verdict (rule 18 wording of the order)
    if E_i == "fail":
        verdict = "NO-GO"
    elif E_i == "not evaluable":
        verdict = "INCONCLUSIVE"
    elif E_iii == "pass":
        verdict = "GO"
    else:
        verdict = "PARTIAL"
    with open(OUT / "tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["test", "status", "value", "reference", "pass"])
        w.writeheader()
        for t in tests:
            w.writerow(dict(test=t["test"], status=t["status"], value=t["value"], reference=t["reference"], **{"pass": t["pass_"]}))
    v = dict(item="E", verdict=verdict, E_i=E_i, E_ii=E_ii, E_iii=E_iii, bound=BOUND,
             E_i_fails=[list(x) for x in allf], E_i_not_run=[list(x) for x in alln],
             E2_violations=[r["unit"] for r in viol], E3_match=match,
             gpu_seconds=sum(r.get("seconds", 0.0) for r in g.values()),
             rule="E-i fail -> NO-GO; E-i not evaluable -> INCONCLUSIVE; E-i pass and E-iii pass -> GO; else PARTIAL")
    CM.atomic_write_text(OUT / "verdict.json", json.dumps(v, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: v[k] for k in ("verdict", "E_i", "E_ii", "E_iii", "E3_match")}))
    print(f"E-i fails {len(allf)}, not run {len(alln)}; E2 violations {len(viol)}")


if __name__ == "__main__":
    main()
