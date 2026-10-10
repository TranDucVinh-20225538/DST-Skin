#!/usr/bin/env python3
"""T1 H4: results/t1/H/falsification_table.csv (claims F-01 ... F-21) and falsification_reportonly.csv (F-02c, F-16i,
F-19i, F-19r), filled mechanically from each item's verdict.json / tests.csv status words.

Status map: pass -> supported-in-tested-region; fail -> not-supported; not evaluable -> not-evaluable; not run (or item
missing) -> not-run. Special rows (order H4): F-05 A-iii not evaluable -> not-evaluable; F-12 needs D-i and D-iii;
F-19 is set by G-ii alone; F-20 as written in the order.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

MAP = {"pass": "supported-in-tested-region", "fail": "not-supported", "not evaluable": "not-evaluable",
       "not run": "not-run"}


def vj(item):
    p = CM.RES / item / "verdict.json"
    return json.load(open(p)) if p.exists() else None


def word(v, key):
    if v is None:
        return "not run"
    s = v.get(key, v.get("status", {}).get(key) if isinstance(v.get("status"), dict) else None)
    if s is None:
        return "not run"
    if s is True:
        return "pass"
    if s is False:
        return "fail"
    s = str(s)
    for w in ("not evaluable", "not run", "pass", "fail"):
        if s.startswith(w):
            return w
    return s


def main():
    A, B, C, D, E, F, G, H = (vj(x) for x in "ABCDEFGH")
    a0 = (A or {}).get("a0", {})
    rows = []

    def add(fid, claim, item, thr, observed, status):
        rows.append(dict(id=fid, claim=claim, item=item, threshold=thr, observed=observed, status=status))

    w = "pass" if a0.get("a0a") else ("fail" if A else "not run")
    add("F-01", "K2 reproduces in the toy", "A0a", "within 3 SE; theory fn 1 %", f"A0a {w}", MAP[w])
    w = "pass" if a0.get("a0b") else ("fail" if A else "not run")
    add("F-02", "Track A Delta reproduced", "A0b", "1e-6", f"A0b {w}", MAP[w])
    for fid, claim, key, thr in (("F-03", "P beats ICC-only across backbones", "A_i", "LB(mean D_b) > 0 vs B1 and B1w"),
                                 ("F-04", "P beats d/N, rho^2, (bound if discriminable)", "A_ii", "LB > 0"),
                                 ("F-05", "P predicts dependence on G, n, N, lambda", "A_iii", "LB > 0 vs B6-scaling and B5"),
                                 ("F-06", "Delta Q accurate", "A_iv", "median rel. error <= 0.35; R2_oos > 0")):
        w = word(A, key)
        add(fid, claim, key.replace("_", "-"), thr, f"{key.replace('_', '-')} {w}", MAP[w])
    for fid, claim, key, thr in (("F-07", "kNN leak collapses on rho sqrt(PR)", "B_i", "R2_oos > 0; LB > 0 vs ICC"),
                                 ("F-08", "matched toy simulator predicts kNN", "B_ii", "LB > 0 vs ICC")):
        w = word(B, key)
        add(fid, claim, key.replace("_", "-"), thr, f"{key.replace('_', '-')} {w}", MAP[w])
    for fid, claim, key, thr in (("F-09", "closed form valid in R0", "C_i", "median <= 0.10, p90 <= 0.25"),
                                 ("F-10", "cap holds", "C_ii", "no S8"),
                                 ("F-11", "kNN collapse in the toy", "C_iii", "score >= 0.90")):
        w = word(C, key)
        add(fid, claim, key.replace("_", "-"), thr, f"{key.replace('_', '-')} {w}", MAP[w])
    d1, d3 = word(D, "D_i"), word(D, "D_iii")
    w = "fail" if "fail" in (d1, d3) else ("pass" if (d1, d3) == ("pass", "pass") else
                                          ("not run" if D is None else "not evaluable"))
    add("F-12", "T separates overlap", "D-i/D-iii", "sens, spec >= 0.90; AUC >= 0.90 on >= 11/13",
        f"D-i {d1}; D-iii {d3}", MAP[w])
    w = word(D, "D_ii")
    add("F-13", "no regime with material leak and blind T", "D-ii", "none found",
        f"D-ii {w}; falsifiers {(D or {}).get('n_falsifiers')}", MAP[w])
    w = word(E, "E_i")
    add("F-14", "group-level CS valid for theta_group (regimes (a) and (b), E1 and E4)", "E-i", "<= alpha + 3 SE everywhere",
        f"E-i {w}", MAP[w])
    w = word(E, "E_iii")
    add("F-15", "label-free recovery above threshold", "E-iii", "no failure at rho_f sqrt(d/2) >= 3", f"E-iii {w}", MAP[w])
    for fid, claim, key, thr in (("F-16", "first-order sign law on real cells", "F_ii", "sens >= .80 (LB > .65), spec >= .90"),
                                 ("F-17", "Edgeworth sign on balanced shifts", "F_iii", "acc >= .70 (LB > .5), beats skew baseline"),
                                 ("F-18", "blind-but-detectable on real features", "F_iv", ">= 80 % of backbones")):
        w = word(F, key)
        add(fid, claim, key.replace("_", "-"), thr, f"{key.replace('_', '-')} {w}", MAP[w])
    w = word(G, "G_ii")
    add("F-19", "scorer-channel prediction under fine-tuning (primary patch-level Delta_G; predictive accuracy under joint "
        "change of fit set and eval set across k in {1, 2, 4}, not a pure fit-effect measurement)", "G-ii",
        "rank >= 5/6; rel. error <= 0.5", f"G-ii {w} (G2 not run: A = NO-GO); 0 model instances", MAP[w])
    hi, hii = word(H, "H_i"), word(H, "H_ii")
    if H is None:
        w20 = "not-run"
    elif hi == "fail":
        w20 = "not-supported"
    elif hii == "not run" or hii == "not evaluable" or (H or {}).get("verdict") == "INCONCLUSIVE":
        w20 = "not-evaluable"
    elif hi == "pass" and hii == "pass":
        w20 = "supported-in-tested-region"
    elif hii == "fail":
        w20 = "not-supported"
    else:
        w20 = "not-evaluable"
    add("F-20", "null controls and placebo", "H1/H2", "S9 not triggered; real - placebo LB > 0", f"H-i {hi}; H-ii {hii}", w20)
    w = word(H, "H_iii")
    add("F-21", "estimator CIs cover", "H3", ">= 0.93 in every cell", f"H-iii {w}", MAP[w])
    out = CM.RES / "H"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "falsification_table.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    gi = word(G, "G_i")
    sens = []
    p = CM.RES / "G" / "verdict.json"
    if p.exists():
        g = json.load(open(p))
        sens = [f"sign flags {g.get('n_sign_flags')}", f"magnitude flags {g.get('n_mag_flags')}"]
    rep = [dict(id="F-02c", item="A0c", content="unit conversion of the scorer precision", threshold="1e-9 relative (as A0c)",
                observed="pass" if a0.get("a0c") else ("fail" if A else "not run")),
           dict(id="F-16i", item="F-i", content="F1 reproduction of the Gaussian reference", threshold="within tolerance (as F-i)",
                observed=word(F, "F_i")),
           dict(id="F-19i", item="G-i", content="frozen-feature checks c1-c3", threshold="as G-i",
                observed=f"{gi}; median e_rel {(G or {}).get('median_e_rel')}; entering {(G or {}).get('entering')}"),
           dict(id="F-19r", item="G", content="slide-equal Delta_G^sw vs patch-level Delta_G; slide- vs patch-dose", threshold="none",
                observed="; ".join(sens) or "not run")]
    with open(out / "falsification_reportonly.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rep[0]))
        wr.writeheader()
        wr.writerows(rep)
    cnt = {s: sum(r["status"] == s for r in rows) for s in MAP.values()}
    print(json.dumps(cnt), "claims", len(rows))
    assert sum(cnt.values()) == len(rows) == 21


if __name__ == "__main__":
    main()
