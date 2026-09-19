#!/usr/bin/env python3
"""Direction #2: ID-only binary Maha vs MSP gate. CPU, no retrain.

Precommit (locked before numbers): split ID-val 50/50 seed=42.
Spearman(MSP, Maha) on calib; if ρ < 0.5 → Maha else MSP.
No OOD labels at fit. No convex mix. Bar = Maha and ViM, not MSP.
If the gate picks Maha on almost every Camelyon backbone, write
'ID-only selector reconfirms Maha' — not a new detector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr

from src.utils.benchmark_metrics import calc_auroc
from src.utils.scoring import OODScorer

SEED = 42
RHO_THRESH = 0.5  # precommit; do not tune on OOD
SKIP_STEMS = ("_supcon",)

CAMELYON = Path("outputs/features/camelyon17/frac1/seed42")
SKIN_DIR = Path("outputs/features/skin")
SKIN_ROOT = Path("outputs/features")
OUT = Path("outputs/reports/id_only_maha_msp_gate.csv")
TXT = Path("outputs/reports/id_only_maha_msp_gate.txt")


def load_pt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def l2_normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def fit_maha(train_feats):
    xn = l2_normalize(train_feats)
    lw = LedoitWolf().fit(xn)
    return lw.location_, lw.precision_


def score_maha(feats, mu, precision):
    x = l2_normalize(feats)
    diff = x - mu
    d2 = np.sum(diff * (diff @ precision), axis=1)
    d2 = np.maximum(d2, 0.0)
    return (-np.sqrt(d2)).astype(np.float64)


def coverage_at_risk10(scores, logits, labels) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    pred = np.asarray(logits).argmax(1)
    labels = np.asarray(labels)
    correct = (pred == labels).astype(np.float64)
    order = np.argsort(-scores)
    csort = correct[order]
    n = len(csort)
    best = None
    for k in range(1, n + 1):
        risk = 1.0 - float(csort[:k].mean())
        if risk <= 0.10:
            best = k / n
    return best


def cells() -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for path in sorted(CAMELYON.glob("*_features.pt")):
        stem = path.name.replace("_features.pt", "")
        if any(s in stem for s in SKIP_STEMS) or stem.endswith("_224"):
            continue
        out.append(("camelyon17", stem, path))
    skin_map = {
        "resnet18": SKIN_ROOT / "resnet18_isic_pad_features.pt",
        "resnet50": SKIN_ROOT / "resnet50_isic_pad_features.pt",
        "effb3": SKIN_ROOT / "effb3_isic_pad_features.pt",
        "densenet121": SKIN_DIR / "densenet121_isic_pad_features.pt",
        "convnext_tiny": SKIN_DIR / "convnext_tiny_isic_pad_features.pt",
        "mobilenet_v3_large": SKIN_DIR / "mobilenet_v3_large_isic_pad_features.pt",
        "regnet_y_3_2gf": SKIN_DIR / "regnet_y_3_2gf_isic_pad_features.pt",
        "efficientnet_v2_s": SKIN_DIR / "efficientnet_v2_s_isic_pad_features.pt",
    }
    for stem, path in skin_map.items():
        if path.exists():
            out.append(("skin_isic_pad", stem, path))
    return out


def run_cell(domain: str, backbone: str, path: Path) -> dict:
    print(f"=== {domain} {backbone} {path} ===", flush=True)
    data = load_pt(path)
    train_f = to_np(data["train_feats"])
    val_logits = to_np(data["val_logits"])
    val_feats = to_np(data["val_feats"])
    ood_logits = to_np(data["ood_logits"])
    ood_feats = to_np(data["ood_feats"])
    ood_labels = to_np(data["ood_labels"])

    mu, prec = fit_maha(train_f)
    msp_val = OODScorer.score_msp(val_logits)
    maha_val = score_maha(val_feats, mu, prec)
    msp_ood = OODScorer.score_msp(ood_logits)
    maha_ood = score_maha(ood_feats, mu, prec)

    rng = np.random.default_rng(SEED)
    n = len(msp_val)
    perm = rng.permutation(n)
    mid = n // 2
    calib, ev = perm[:mid], perm[mid:]

    rho, _ = spearmanr(msp_val[calib], maha_val[calib])
    rho = float(rho)
    chosen = "mahalanobis" if rho < RHO_THRESH else "msp"

    vim_auroc = float("nan")
    if "fc_weight" in data and "fc_bias" in data and "train_logits" in data:
        try:
            from src.utils.ood_vim_react import fit_vim, vim_score

            vp = fit_vim(
                train_feats=train_f,
                train_logits=to_np(data["train_logits"]),
                fc_weight=to_np(data["fc_weight"]),
                fc_bias=to_np(data["fc_bias"]),
                d=None,
            )
            vim_ev = -vim_score(val_feats[ev], vp)
            vim_ood = -vim_score(ood_feats, vp)
            vim_auroc = calc_auroc(vim_ev, vim_ood)
        except Exception as exc:
            print(f"  ViM skip: {exc}", flush=True)

    def auroc_pair(id_s, ood_s):
        return calc_auroc(id_s[ev], ood_s)

    a_msp = auroc_pair(msp_val, msp_ood)
    a_maha = auroc_pair(maha_val, maha_ood)
    if chosen == "mahalanobis":
        gate_id, gate_ood = maha_val, maha_ood
    else:
        gate_id, gate_ood = msp_val, msp_ood
    a_gate = auroc_pair(gate_id, gate_ood)

    cov_msp = coverage_at_risk10(msp_ood, ood_logits, ood_labels)
    cov_maha = coverage_at_risk10(maha_ood, ood_logits, ood_labels)
    cov_gate = coverage_at_risk10(gate_ood, ood_logits, ood_labels)

    rec = {
        "domain": domain,
        "backbone": backbone,
        "n_id_val": n,
        "n_calib": int(len(calib)),
        "n_eval": int(len(ev)),
        "spearman_msp_maha_calib": rho,
        "rho_thresh": RHO_THRESH,
        "gate_choice": chosen,
        "auroc_msp_eval": a_msp,
        "auroc_maha_eval": a_maha,
        "auroc_vim_eval": vim_auroc,
        "auroc_gate_eval": a_gate,
        "cov10_msp_ood": cov_msp,
        "cov10_maha_ood": cov_maha,
        "cov10_gate_ood": cov_gate,
    }
    print(
        f"  ρ={rho:.3f} → {chosen}; "
        f"AUROC MSP={a_msp:.3f} Maha={a_maha:.3f} gate={a_gate:.3f} ViM={vim_auroc:.3f}",
        flush=True,
    )
    del data, train_f
    return rec


def main() -> None:
    rows = [run_cell(d, b, p) for d, b, p in cells()]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    lines = [
        "ID-only Maha/MSP gate. Precommit ρ<0.5 → Maha else MSP.",
        "Bar = Maha and ViM. Beating MSP does not count.",
        "",
        df.to_string(index=False),
        "",
    ]
    for domain, g in df.groupby("domain"):
        n_m = int((g.gate_choice == "mahalanobis").sum())
        n = len(g)
        spread_gate = float(g.auroc_gate_eval.std(ddof=1)) if n > 1 else float("nan")
        spread_maha = float(g.auroc_maha_eval.std(ddof=1)) if n > 1 else float("nan")
        spread_msp = float(g.auroc_msp_eval.std(ddof=1)) if n > 1 else float("nan")
        mean_gate = float(g.auroc_gate_eval.mean())
        mean_maha = float(g.auroc_maha_eval.mean())
        mean_vim = float(g.auroc_vim_eval.mean())
        note = (
            f"{domain}: gate chose Maha {n_m}/{n}. "
            f"mean AUROC gate={mean_gate:.3f} Maha={mean_maha:.3f} ViM={mean_vim:.3f}. "
            f"spread gate={spread_gate:.3f} Maha={spread_maha:.3f} MSP={spread_msp:.3f}."
        )
        if n_m >= n - 1:
            note += " Selector reconfirms Maha — not a new detector."
        lines.append(note)
        print(note, flush=True)
    TXT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT} {TXT}", flush=True)


if __name__ == "__main__":
    main()
