#!/usr/bin/env python3
"""Precommit addendum for one item's work list(s) (rule 16), written before the item launches.

    make_addendum.py A   -> results/t1/PRECOMMIT_T1_addendum_A.json, decisions/precommit_t1_2026-10-11_addendum_A.md
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

CAM_BACKBONES = ["conch_v1_5", "convnext_tiny", "densenet121", "dinov2_vitb14", "dinov2_vitl14", "effb3",
                 "efficientnet_v2_s", "mobilenet_v3_large", "regnet_y_3_2gf", "resnet18", "resnet50", "uni", "virchow2"]
A2_G = [3, 6, 9, 12, 15]
A2_N = ["10", "30", "100", "300", "1000", "all"]


def registry_cells():
    return [r["cell_id"] for r in csv.DictReader(open(CM.RES / "I" / "registry.csv"))]


def units_A():
    cells = registry_cells()
    u = [f"a1|{c}" for c in cells] + [f"a3|{c}" for c in cells]
    u += [f"a2|{b}|f{f}|G{G:02d}|n{n}|r{r}" for b in CAM_BACKBONES for f in (0, 1) for G in A2_G for n in A2_N for r in range(10)]
    return {0: u}


def units_B():
    return {0: [f"b|{c}" for c in registry_cells()]}


def units_C():
    import c_design
    return {0: sorted(c_design.all_units())}


def params_C():
    import c_design
    return c_design.all_units()


def units_D():
    import d_design
    return {0: sorted(d_design.all_units(CM.RES / "I" / "registry.csv"))}


def params_D():
    import d_design
    return d_design.all_units(CM.RES / "I" / "registry.csv")


def units_E():
    import e_design
    a, b = e_design.lists(CM.RES / "I" / "registry.csv")
    return {0: sorted(a), 1: sorted(b)}


def params_E():
    import e_design
    a, b = e_design.lists(CM.RES / "I" / "registry.csv")
    return {**a, **b}


def params_H():
    reg = list(csv.DictReader(open(CM.RES / "I" / "registry.csv")))
    import item_H1
    p = {u: dict(kind=u.split("|")[0], cell=u.split("|", 1)[1]) for u in item_H1.units()}
    for r in reg:
        p[f"h2|{r['cell_id']}"] = dict(kind="h2", cell=r["cell_id"], n_perm=20)
    rhos = (0.02, 0.05, 0.1, 0.2)
    for f in (0, 1):
        for d in (768, 1024, 2560):
            for rho in rhos:
                p[f"h3|camelyon|f{f}|d{d}|rho{rho:g}"] = dict(kind="h3", design="camelyon", n_g_from="camelyon_resnet50_s42", fold=f, d=d, rho=rho)
    first = {}
    for r in sorted(reg, key=lambda r: r["cell_id"]):
        if r["dataset"] != "camelyon":
            first.setdefault((r["dataset"], int(r["d"])), r["cell_id"])
    for (ds, d), cell in sorted(first.items()):
        for rho in rhos:
            p[f"h3|{ds}|d{d}|rho{rho:g}"] = dict(kind="h3", design=ds, n_g_from=cell, fold=0, d=d, rho=rho)
    return p


def units_H():
    return {0: sorted(params_H())}


UNITS = {"A": units_A, "B": units_B, "C": units_C, "D": units_D, "E": units_E, "H": units_H}
PARAMS = {"C": params_C, "D": params_D, "E": params_E, "H": params_H}


def work_list(item, j, n_lists, ids):
    sorted_ids = sorted(ids)
    child = np.random.SeedSequence([CM.MASTER_SEED, CM.ITEM_CODE[item], 16, 0]).spawn(n_lists)[j]
    perm = np.random.default_rng(child).permutation(len(sorted_ids))
    return dict(item=item, list_index_j=j, n_lists=n_lists,
                seed=f"SeedSequence([{CM.MASTER_SEED}, {CM.ITEM_CODE[item]}, 16, 0]).spawn({n_lists})[{j}]",
                n_units=len(sorted_ids),
                sha256_sorted_ids_newline_joined=hashlib.sha256("\n".join(sorted_ids).encode()).hexdigest(),
                sorted_ids=sorted_ids, execution_order=[sorted_ids[p] for p in perm])


def main(item):
    lists = UNITS[item]()
    n_lists = 2 if item in ("E", "G") else 1
    wl = [work_list(item, j, n_lists, ids) for j, ids in sorted(lists.items())]
    date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat(timespec="seconds")
    rec = dict(title=f"PRECOMMIT T1 addendum: work list(s) of item {item} (rule 16)", written_asia_saigon=date,
               precommit_commit=CM.precommit_hash(), registry_sha256=CM.sha256_file(CM.RES / "I" / "registry.csv"),
               status=f"Written before any item-{item} computation (no criterion, threshold, seed or grid is changed).",
               unit_id_formats={"A": "a1|<cell_id> (A1 per registry cell), a3|<cell_id> (A3 audit per registry cell), "
                                     "a2|<backbone>|f<fold>|G<G'>|n<n'>|r<repeat> (A2, 13 Camelyon backbones, seed 42)",
                               "B": "b|<cell_id> (one unit = both paper_2fold folds of one registry cell: kNN k in {1,5,10,20,50,100,200}, descriptors, MTS)",
                               "C": "see the docstring of scripts/t1/c_design.py (c0|..., c1|<i>, c3|..., c4tv|..., c4sc|..., c5|<variant>|<i>); parameters of every unit in unit_params",
                               "D": "see the docstring of scripts/t1/d_design.py (d1|<i>, d2r|<i>, d3|<cell>, d4|<cell>); parameters in unit_params; D2 CMA-ES evaluations are adaptive and not in the list",
                               "E": "see the docstring of scripts/t1/e_design.py; list j = 0 = regime (a) (E1 regime-(a) cells, E2, E3, E4 (a)/(a')), "
                                    "list j = 1 = regime (b) (E1 regime-(b) cells, E4 (b)), processed after list 0; parameters in unit_params",
                               "H": "hn1|<cell> (H-N1 Gaussian twin), hn3|<cell> (H-N3 identical split), hn2|<cell> (H-N2 one image per group; datasets with >= 500 groups), "
                                    "h2|<cell> (H2 placebo, 20 within-hospital / within-cell permutations), h3|<design>|...|d<d>|rho<rho> (H3 coverage; Camelyon n_g = real slide counts of each fold, "
                                    "medbench design = n_g of fold 0 of the first registry cell (sorted) of each dataset x d); parameters in unit_params"}.get(item),
               work_lists=wl)
    if item in PARAMS:
        rec["unit_params"] = PARAMS[item]()
    out = CM.RES / f"PRECOMMIT_T1_addendum_{item}.json"
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    CM.atomic_write_text(out, json.dumps(rec, indent=1) + "\n")
    md = CM.REPO / "decisions" / f"precommit_t1_2026-10-11_addendum_{item}.md"
    lines = [f"# Precommit T1 addendum — item {item} work list(s) (rule 16)", "",
             f"Written {date} (Asia/Saigon), before item {item} launches. Precommit commit `{rec['precommit_commit']}`.",
             f"Registry `results/t1/I/registry.csv` sha256 `{rec['registry_sha256']}`.", "",
             f"Unit IDs: {rec['unit_id_formats']}", "", "| list j | seed | units | sha256 of newline-joined sorted IDs |", "|---|---|---|---|"]
    lines += [f"| {w['list_index_j']} | `{w['seed']}` | {w['n_units']} | `{w['sha256_sorted_ids_newline_joined']}` |" for w in wl]
    lines += ["", f"Full sorted list and execution order: `results/t1/PRECOMMIT_T1_addendum_{item}.json`.", ""]
    CM.atomic_write_text(md, "\n".join(lines))
    print(out, md, [w["n_units"] for w in wl], [w["sha256_sorted_ids_newline_joined"] for w in wl])


if __name__ == "__main__":
    main(sys.argv[1])
