#!/usr/bin/env python3
"""R3 item 1 stop-check table: package point estimate (paper_2fold, Track A folds) vs Track A Delta_fit.

Reads ~/r3work/item1/match/*.json and cells_*.csv (trackA_delta_*). Camelyon seed-42 CNN cells without a Track A
ViM value in cells_camelyon.csv take dfit_ViM from trackA_foundation/slide_identity.csv. Writes
results/r3/1/match.csv and results/r3/1/match_table.md (tolerance 0.001).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-3
W = Path.home() / "r3work/item1"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/r3/1"
SI = REPO / "outputs/reports/rigor_pack/miccai_campaign/trackA_foundation/slide_identity.csv"
SC = ("mahalanobis", "knn", "vim")


def main():
    cells = pd.concat([pd.read_csv(W / f"cells_{p}.csv") for p in ("med", "medfm", "camelyon")]).set_index("cell")
    si = pd.read_csv(SI).set_index("model")
    rows = []
    for f in sorted((W / "match").glob("*.json")):
        m = json.loads(f.read_text())
        c = cells.loc[m["cell"]]
        for s in SC:
            ref, src = c[f"trackA_delta_{s}"], "cells"
            if s == "vim" and np.isnan(ref) and c.dataset == "camelyon" and c.seed == 42 and c.model in si.index:
                ref, src = si.loc[c.model, "dfit_ViM"], "slide_identity"
            r = {"cell": m["cell"], "dataset": c.dataset, "model": c.model, "seed": c.seed, "kind": c.kind,
                 "scorer": s, "trackA_delta": ref, "trackA_source": src, "vim_dim": m["vim_dim"], "head": m["head"]}
            for k in ("tracka", "tracka_seen", "package"):
                if k in m:
                    r[f"delta_{k}"] = m[k][s]["delta"]
                    r[f"absdiff_{k}"] = abs(m[k][s]["delta"] - ref) if not np.isnan(ref) else np.nan
            for k in ("tracka", "tracka_seen"):
                v = r.get(f"absdiff_{k}", np.nan)
                r[f"match_{k}"] = bool(v <= TOL) if not np.isnan(v) else "n/a"
            rows.append(r)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "match.csv", index=False)

    lab = {"tracka_seen": "Track A scorers, Track A seen set", "tracka": "Track A scorers, strict seen set",
           "package": "package scorers, strict seen set"}
    ks = [k for k in lab if f"absdiff_{k}" in d]
    lines = ["| dataset | scorer | cells | " + " | ".join(f"match: {lab[k]} | max abs diff: {lab[k]}" for k in ks) + " |",
             "|---|---|---|" + "---|---|" * len(ks)]
    for (ds, s), g in d.groupby(["dataset", "scorer"]):
        v = g[g.trackA_delta.notna()]
        cols = []
        for k in ks:
            x = v[f"absdiff_{k}"].dropna()
            cols.append(f"{int((x <= TOL).sum())}/{len(x)} | {x.max():.1e}" if len(x) else "n/a | n/a")
        lines.append(f"| {ds} | {s} | {len(v)} | " + " | ".join(cols) + " |")
    best = d.absdiff_tracka_seen.fillna(d.absdiff_tracka) if "absdiff_tracka_seen" in d else d.absdiff_tracka
    miss = d[(best > TOL) & (d.scorer != "vim")]
    lines += ["", f"Mahalanobis / kNN cells failing 0.001 with Track A scorers and Track A seen set (strict where no "
              f"fold_id_eval): {len(miss)}"]
    if len(miss):
        lines += ["", "| cell | scorer | Track A Δ | package Δ | abs diff |", "|---|---|---|---|---|"]
        pk = d.delta_tracka_seen.fillna(d.delta_tracka) if "delta_tracka_seen" in d else d.delta_tracka
        lines += [f"| {r.cell} | {r.scorer} | {r.trackA_delta:.5f} | {p:.5f} | {b:.2e} |"
                  for r, p, b in zip(miss.itertuples(), pk[miss.index], best[miss.index])]
    (OUT / "match_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
