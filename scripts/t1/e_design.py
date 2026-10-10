"""T1 item E: work lists (rule 16: E has two lists, regime (a) j = 0 and regime (b) j = 1).

Unit IDs:
  e1|a|<outcome>|<size>|n<n>|rho<rho>          E1 regime (a): 3 outcomes x 3 size laws x 4 n x 6 rho = 216 cells
  e1|b|<outcome>|<mech>|n<n>|rho<rho>          E1 regime (b): 3 outcomes x 3 mechanisms x 3 n x 2 rho = 54 cells
  e2|n<n>|rhof<rho_f>|d<d>                     E2 label-free recovery: 3 n x 5 rho_f x 5 d = 75 configs (500 replicates each)
  e3|<cell_id>                                 E3 real features (Camelyon 13 backbones seed 42 + medbench cells)
  e4|<a|ap>|<backbone>[|n<n>]                  E4 semi-synthetic, regimes (a) n in {20, 100} and (a')   [list j = 0]
  e4|b|<backbone>                              E4 regime (b)                                         [list j = 1]
outcome in {flag0.3, flag0.05, beta}; size in {const, poisson, lognormal}; mech in {b1, b2, b3}.
"""
from __future__ import annotations

import csv

OUTCOMES = ["flag0.3", "flag0.05", "beta"]
SIZES_A = ["const", "poisson", "lognormal"]
MECH_B = ["b1", "b2", "b3"]
CAM_BACKBONES = ["conch_v1_5", "convnext_tiny", "densenet121", "dinov2_vitb14", "dinov2_vitl14", "effb3",
                 "efficientnet_v2_s", "mobilenet_v3_large", "regnet_y_3_2gf", "resnet18", "resnet50", "uni", "virchow2"]


def lists(registry_csv):
    a, b = {}, {}
    for o in OUTCOMES:
        for s in SIZES_A:
            for n in (5, 20, 50, 200):
                for rho in (0, 0.02, 0.05, 0.1, 0.2, 0.4):
                    a[f"e1|a|{o}|{s}|n{n}|rho{rho:g}"] = dict(regime="a", outcome=o, size=s, n=n, rho=rho)
        for mch in MECH_B:
            for n in (5, 20, 50):
                for rho in (0.05, 0.2):
                    b[f"e1|b|{o}|{mch}|n{n}|rho{rho:g}"] = dict(regime="b", outcome=o, size=mch, n=n, rho=rho)
    for n in (5, 15, 50):
        for rf in (0.05, 0.1, 0.2, 0.3, 0.6):
            for d in (4, 16, 64, 256, 1024):
                a[f"e2|n{n}|rhof{rf:g}|d{d}"] = dict(n=n, rho_f=rf, d=d, G=60)
    reg = list(csv.DictReader(open(registry_csv)))
    for r in reg:
        if r["dataset"] != "camelyon" or r["cell_id"] in {f"camelyon_{b}_s42" for b in CAM_BACKBONES}:
            a[f"e3|{r['cell_id']}"] = dict(cell=r["cell_id"])
    for bb in CAM_BACKBONES:
        for n in (20, 100):
            a[f"e4|a|{bb}|n{n}"] = dict(backbone=bb, regime="a", n=n)
        a[f"e4|ap|{bb}"] = dict(backbone=bb, regime="ap")
        b[f"e4|b|{bb}"] = dict(backbone=bb, regime="b")
    return a, b
