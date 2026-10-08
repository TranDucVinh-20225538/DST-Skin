#!/usr/bin/env python3
"""R3 item 8 (post hoc): results/r3/8/ood_overlap/REPORT.md and tracka_kermany_clean_ood.csv.

Overlap counts from audit.json; Track A Kermany std sensitivity = item-1 v2 rows (full OOD, ~/r3work/item1/out_v2)
beside the same cells with OOD restricted to patients absent from every ID set (~/r3work/item8_oodsens/out_v2).
P2-b sensitivity is in results/r3/8/p2b/REPORT.md. Usage: item8_ood_overlap_report.py [<commit>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/r3/8/ood_overlap"
FULL = Path.home() / "r3work/item1/out_v2"
CLEAN = Path.home() / "r3work/item8_oodsens/out_v2"
ARMS = ("kermany_b0", "kermany_b1", "kermany_std", "isic2019_b0", "isic2019_b1", "isic2019_std")


def main() -> int:
    commit = sys.argv[1] if len(sys.argv) > 1 else "uncommitted"
    a = json.loads((OUT / "audit.json").read_text())
    cells = [x.strip() for x in (Path.home() / "r3work/item8_oodsens/list_kermany_std.txt").read_text().split() if x.strip()]
    meta = pd.read_csv(Path.home() / "r3work/item1/cells_med.csv").set_index("cell")
    rows, missing = [], []
    for c in cells:
        pf, pc = FULL / c / "paper_ci.csv", CLEAN / c / "paper_ci.csv"
        if not (pf.exists() and pc.exists()):
            missing.append(c)
            continue
        f, k = pd.read_csv(pf).set_index("scorer"), pd.read_csv(pc).set_index("scorer")
        for sc in k.index:
            r = dict(cell=c, scorer=sc, K=int(k.loc[sc, "K"]), n_groups_train=int(k.loc[sc, "n_groups_train"]),
                     d=int(meta.loc[c, "d"]), id_acc=float(meta.loc[c, "id_acc"]))
            assert int(f.loc[sc, "n_groups_train"]) == r["n_groups_train"]
            for tag, t in (("full", f), ("clean", k)):
                for col in ("n_ood", "auroc_seen", "auroc_unseen", "delta", "old_bootstrap_lo", "old_bootstrap_hi",
                            "new_jackknife_lo", "new_jackknife_hi", "status_thr", "status_gt0"):
                    r[f"{col}_{tag}"] = t.loc[sc, col]
            rows.append(r)
    T = pd.DataFrame(rows)
    if len(T):
        T.to_csv(OUT / "tracka_kermany_clean_ood.csv", index=False)
    L = ["# R3 item 8 (post hoc): OOD / ID patient overlap and OOD-restricted sensitivity", "", f"Commit: {commit}", ""]
    if len(T):
        ch = int((T.status_thr_full != T.status_thr_clean).sum())
        L += [f"Verdict: Track A Kermany, OOD restricted to patients absent from every ID set: median Δ change "
              f"{np.median(T.delta_clean - T.delta_full):+.4f} (range {(T.delta_clean - T.delta_full).min():+.4f} to "
              f"{(T.delta_clean - T.delta_full).max():+.4f}); status vs 0.02 changes in {ch} of {len(T)} rows.", ""]
    else:
        L += ["Verdict: Track A sensitivity not run (no completed cell pairs).", ""]
    if missing:
        L += [f"Not run / no output ({len(missing)}): " + ", ".join(missing), ""]
    L += ["Post hoc: requested after the P2-b Kermany tie finding; not in any precommit. Precommitted estimands unchanged.", "",
          "## Overlap counts (ID sets vs OOD)", "",
          "| arm | group unit | OOD images (groups) | OOD groups in each ID set | OOD groups in any ID set | "
          "OOD images from those groups | clean OOD images (groups) | pixel-identical ID-OOD pairs per ID set | "
          "cross-group identical pairs |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ARMS:
        x = a[k]
        b = x["by_set"]
        sets = list(b)
        L.append(f"| {k} | {x['group_unit']} | {x['n_ood_images']} ({x['n_ood_groups']}) | "
                 + ", ".join(f"{s} {b[s]['ood_groups_in_set']}" for s in sets) + f" | {x['ood_groups_in_any_id_set']} | "
                 f"{x['ood_images_from_any_id_group']} | {x['clean_ood_images']} ({x['clean_ood_groups']}) | "
                 + ", ".join(f"{s} {b[s]['pixel_identical_pairs']}" for s in sets) + " | "
                 + str(sum(b[s]["pixel_identical_pairs_cross_group"] for s in sets)) + " |")
    L.append("")
    if len(T):
        L += ["## Track A Kermany (std arm): full OOD vs clean OOD", "",
              "Same cells, folds, scorers and settings as item 1 v2 (seen = strict); only OOD rows are removed. "
              "n_groups_fit = n_groups_train / K per fold.", "",
              "| cell | scorer | n_groups_fit | K | d | ID acc | n OOD full / clean | AUROC seen full / clean | AUROC unseen full / clean | "
              "Δ full | Δ clean | old CI full | old CI clean | new CI full | new CI clean | status 0.02 full / clean | status 0 full / clean |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in T.itertuples():
            L.append(f"| {r.cell} | {r.scorer} | {r.n_groups_train}/{r.K} | {r.K} | {r.d} | {r.id_acc:.4f} | {r.n_ood_full} / {r.n_ood_clean} | "
                     f"{r.auroc_seen_full:.4f} / {r.auroc_seen_clean:.4f} | {r.auroc_unseen_full:.4f} / {r.auroc_unseen_clean:.4f} | "
                     f"{r.delta_full:.4f} | {r.delta_clean:.4f} | [{r.old_bootstrap_lo_full:.4f}, {r.old_bootstrap_hi_full:.4f}] | "
                     f"[{r.old_bootstrap_lo_clean:.4f}, {r.old_bootstrap_hi_clean:.4f}] | [{r.new_jackknife_lo_full:.4f}, "
                     f"{r.new_jackknife_hi_full:.4f}] | [{r.new_jackknife_lo_clean:.4f}, {r.new_jackknife_hi_clean:.4f}] | "
                     f"{r.status_thr_full} / {r.status_thr_clean} | {r.status_gt0_full} / {r.status_gt0_clean} |")
    L += ["", "## Caveats", "",
          "1. ISIC 2019 has no patient ID; overlap is checked at lesion and pixel level only. No overlap was found, so no "
          "ISIC sensitivity run."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
