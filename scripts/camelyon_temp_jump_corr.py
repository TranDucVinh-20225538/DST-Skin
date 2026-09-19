#!/usr/bin/env python3
"""Job A — logit scale vs Camelyon MSP-jump. No training.

On the 8 zoo backbones with saved features.pt:
  T* = argmin_T NLL(softmax(z/T), y) on ID-val.
  mean logit-norm on ID-val (same scale family; not a second hunt).
  jump = MSP AUROC − mean(R18, R50).

PRIMARY kill test: Pearson r(T*, jump), n=8, |r|>0.6 = notable.
Do not compute feature-norm / entropy if |r| is low.
T-scaled MSP AUROC is stored for the |r|>0.6 follow-up only (same T*, free).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr

from src.utils.benchmark_metrics import calc_auroc, per_sample_logit_norm
from src.utils.scoring import OODScorer

ZOO = (
    "resnet18",
    "resnet50",
    "effb3",
    "densenet121",
    "convnext_tiny",
    "mobilenet_v3_large",
    "regnet_y_3_2gf",
    "efficientnet_v2_s",
)
FEAT_DIR = Path("outputs/features/camelyon17/frac1/seed42")
SCORE_DIR = Path("outputs/reports/camelyon17/frac1/seed42")
OUT_DIR = Path("outputs/reports")
R_THRESH = 0.6
BASELINE = ("resnet18", "resnet50")


def _load_pt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def nll_temperature(t: float, logits: np.ndarray, labels: np.ndarray) -> float:
    t = float(max(t, 1e-3))
    z = torch.as_tensor(logits, dtype=torch.float64) / t
    logp = F.log_softmax(z, dim=1)
    y = torch.as_tensor(labels, dtype=torch.int64)
    return float(-logp[torch.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    res = minimize_scalar(
        lambda t: nll_temperature(t, logits, labels),
        bounds=(0.05, 10.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(res.x)


def msp_t(logits: np.ndarray, t: float) -> np.ndarray:
    z = torch.as_tensor(np.asarray(logits, dtype=np.float64)) / float(t)
    return F.softmax(z, dim=1).numpy().max(axis=1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for bb in ZOO:
        feat_path = FEAT_DIR / f"{bb}_features.pt"
        score_path = SCORE_DIR / f"{bb}_score_comparison.csv"
        if not feat_path.exists() or not score_path.exists():
            raise FileNotFoundError(f"missing {feat_path} or {score_path}")
        print(f"=== {bb} ===", flush=True)
        data = _load_pt(feat_path)
        val_logits = _to_np(data["val_logits"]).astype(np.float64)
        val_labels = _to_np(data["val_labels"]).astype(np.int64)
        ood_logits = _to_np(data["ood_logits"]).astype(np.float64)
        scores = pd.read_csv(score_path)
        msp = float(scores.loc[scores["Method"] == "msp", "AUROC"].iloc[0])
        t_star = fit_temperature(val_logits, val_labels)
        logit_n = float(np.mean(per_sample_logit_norm(val_logits)))
        msp_scaled = calc_auroc(msp_t(val_logits, t_star), msp_t(ood_logits, t_star))
        msp_raw_check = calc_auroc(
            OODScorer.score_msp(val_logits), OODScorer.score_msp(ood_logits)
        )
        rec = {
            "backbone": bb,
            "msp_auroc": msp,
            "msp_auroc_recomputed": msp_raw_check,
            "t_star": t_star,
            "mean_logit_norm_id_val": logit_n,
            "nll_at_1": nll_temperature(1.0, val_logits, val_labels),
            "nll_at_tstar": nll_temperature(t_star, val_logits, val_labels),
            "msp_auroc_tstar": msp_scaled,
        }
        print(
            f"  MSP={msp:.4f} T*={t_star:.4f} ||z||={logit_n:.3f} "
            f"MSP@T*={msp_scaled:.4f}",
            flush=True,
        )
        rows.append(rec)
        del data

    df = pd.DataFrame(rows)
    base = float(df.loc[df["backbone"].isin(BASELINE), "msp_auroc"].mean())
    df["msp_jump_vs_resnet"] = df["msp_auroc"] - base
    df["msp_jump_tstar_vs_resnet"] = df["msp_auroc_tstar"] - float(
        df.loc[df["backbone"].isin(BASELINE), "msp_auroc_tstar"].mean()
    )

    t = df["t_star"].to_numpy()
    z = df["mean_logit_norm_id_val"].to_numpy()
    j = df["msp_jump_vs_resnet"].to_numpy()
    r_t, p_t = pearsonr(t, j)
    rho_t, p_rho_t = spearmanr(t, j)
    r_z, p_z = pearsonr(z, j)

    notable = abs(float(r_t)) > R_THRESH
    if notable:
        verdict = (
            f"NOTABLE |r(T*,jump)|={abs(r_t):.3f}>{R_THRESH}. "
            "Jump may be partly a scale/calibration artifact. "
            "Add temperature-scaled MSP before AUROC; inspect gap shrink. "
            "Recipe B should include T-scaling, not SupCon-only."
        )
    else:
        verdict = (
            f"LOW |r(T*,jump)|={abs(r_t):.3f}<={R_THRESH}. "
            "Scale/calibration confound dismissed. "
            "Do not hunt feature-norm or entropy. "
            "Pilot B = CE+SupCon only (no extra T-scale in the training recipe)."
        )

    summary = {
        "n": int(len(df)),
        "resnet_mean_msp": base,
        "primary_test": "pearson_r(T_star, msp_jump_vs_resnet)",
        "r_tstar_jump": float(r_t),
        "p_tstar_jump": float(p_t),
        "spearman_tstar_jump": float(rho_t),
        "p_spearman_tstar_jump": float(p_rho_t),
        "r_logitnorm_jump_not_kill_test": float(r_z),
        "p_logitnorm_jump_not_kill_test": float(p_z),
        "threshold": R_THRESH,
        "notable": notable,
        "verdict": verdict,
        "dense_minus_r18_msp": float(
            df.loc[df.backbone == "densenet121", "msp_auroc"].iloc[0]
            - df.loc[df.backbone == "resnet18", "msp_auroc"].iloc[0]
        ),
        "dense_minus_r18_msp_tstar": float(
            df.loc[df.backbone == "densenet121", "msp_auroc_tstar"].iloc[0]
            - df.loc[df.backbone == "resnet18", "msp_auroc_tstar"].iloc[0]
        ),
    }
    csv_path = OUT_DIR / "camelyon_temp_jump_corr.csv"
    json_path = OUT_DIR / "camelyon_temp_jump_corr.json"
    txt_path = OUT_DIR / "camelyon_temp_jump_corr.txt"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))
    txt_path.write_text(
        "PRIMARY: Pearson r(T*, MSP-jump vs ResNet mean), n=8, |r|>0.6 notable.\n"
        "Logit-norm r is reported as the listed alternative, not a second hunt.\n"
        f"r(T*,jump)={r_t:.4f} p={p_t:.4f}\n"
        f"Spearman(T*,jump)={rho_t:.4f} p={p_rho_t:.4f}\n"
        f"r(||z||,jump)={r_z:.4f} p={p_z:.4f} (not kill test)\n"
        f"DenseNet-R18 MSP gap raw={summary['dense_minus_r18_msp']:.4f} "
        f"T*={summary['dense_minus_r18_msp_tstar']:.4f}\n\n"
        f"{verdict}\n"
    )
    print(f"\nWrote {csv_path}")
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(verdict)


if __name__ == "__main__":
    main()
