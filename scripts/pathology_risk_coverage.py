#!/usr/bin/env python3
"""Risk-coverage on saved Camelyon/MIDOG features (CPU, no new GPU).

Two protocols, both higher-score-first:

  id_selective: ID-val only (analyze_benchmark_* style).
  ood_triage:   rank OOD samples by score, measure classification error
                among the accepted prefix (plot_triage_final_all.py / MIDL paradox).

Phao B: does AUROC ranking disagree with coverage@10% risk / AURC ranking?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

from src.utils.benchmark_metrics import calc_auroc
from src.utils.scoring import OODScorer

CAMELYON = Path("outputs/features/camelyon17/frac1/seed42")
MIDOG = Path("outputs/features/midog/seed42")
OUT = Path("outputs/reports/pathology_risk_coverage")
METHODS_LOGIT = ("msp", "energy")
COVERAGES = np.linspace(1.0, 0.10, 91)


def load_pt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_np(x):
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


def scores_from_logits(logits):
    z = np.asarray(logits, dtype=np.float32)
    return {
        "msp": OODScorer.score_msp(z),
        "energy": OODScorer.score_energy(z),
    }


def risk_at_coverages(scores, correct, coverages=COVERAGES):
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-scores)
    csort = correct[order]
    n = len(csort)
    risks = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        risks.append(1.0 - float(csort[:k].mean()))
    cov = np.asarray(coverages, dtype=np.float64)
    risk = np.asarray(risks, dtype=np.float64)
    aurc = float(np.trapz(risk, cov))  # lower better; cov decreases so abs
    aurc = float(np.trapz(risk[::-1], cov[::-1]))
    return cov, risk, aurc


def coverage_at_risk(scores, correct, target=0.10):
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-scores)
    csort = correct[order]
    n = len(csort)
    best = None
    for k in range(1, n + 1):
        risk = 1.0 - float(csort[:k].mean())
        if risk <= target:
            best = {"target_risk": target, "coverage": k / n, "accepted": k, "risk": risk}
    return best


def row_for(domain, model, protocol, method, auroc, scores, correct):
    _, _, aurc = risk_at_coverages(scores, correct)
    c10 = coverage_at_risk(scores, correct, 0.10)
    c15 = coverage_at_risk(scores, correct, 0.15)
    return {
        "domain": domain,
        "model": model,
        "protocol": protocol,
        "method": method,
        "ood_auroc": auroc,
        "aurc": aurc,
        "coverage_at_risk10": None if c10 is None else c10["coverage"],
        "coverage_at_risk15": None if c15 is None else c15["coverage"],
        "full_risk": float(1.0 - np.mean(correct)),
        "n": int(len(correct)),
    }


def process_cnn(domain: str, model: str, path: Path) -> list[dict]:
    if not path.exists():
        print(f"skip {path}", flush=True)
        return []
    print(f"=== {domain}/{model} {path} ===", flush=True)
    data = load_pt(path)
    val_z = to_np(data["val_logits"])
    val_y = to_np(data["val_labels"])
    val_f = to_np(data["val_feats"])
    ood_z = to_np(data["ood_logits"])
    ood_y = to_np(data["ood_labels"])
    ood_f = to_np(data["ood_feats"])
    train_f = to_np(data["train_feats"])
    logit_s_val = scores_from_logits(val_z)
    logit_s_ood = scores_from_logits(ood_z)
    print(f"  fit Maha n_train={len(train_f)} d={train_f.shape[1]}", flush=True)
    mu, prec = fit_maha(train_f)
    maha_val = score_maha(val_f, mu, prec)
    maha_ood = score_maha(ood_f, mu, prec)
    val_ok = (val_z.argmax(1) == val_y).astype(np.float64)
    ood_ok = (ood_z.argmax(1) == ood_y).astype(np.float64)
    out = []
    bundle = {**{k: (logit_s_val[k], logit_s_ood[k]) for k in METHODS_LOGIT}, "mahalanobis": (maha_val, maha_ood)}
    for method, (id_s, ood_s) in bundle.items():
        a = calc_auroc(id_s, ood_s)
        out.append(row_for(domain, model, "id_selective", method, a, id_s, val_ok))
        out.append(row_for(domain, model, "ood_triage", method, a, ood_s, ood_ok))
        print(
            f"  {method:12s} AUROC={a:.3f}  ID_AURC={out[-2]['aurc']:.4f} "
            f"OOD_triage_AURC={out[-1]['aurc']:.4f} "
            f"cov@10% ID={out[-2]['coverage_at_risk10']} OOD={out[-1]['coverage_at_risk10']}",
            flush=True,
        )
    del data, train_f, val_f, ood_f
    return out


def rank_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, protocol, method), g in df.groupby(["domain", "protocol", "method"]):
        if len(g) < 2:
            continue
        by_auroc = list(g.sort_values("ood_auroc", ascending=False)["model"])
        by_cov = list(g.sort_values("coverage_at_risk10", ascending=False)["model"])
        by_aurc = list(g.sort_values("aurc", ascending=True)["model"])
        rows.append({
            "domain": domain,
            "protocol": protocol,
            "method": method,
            "rank_by_ood_auroc": ">".join(by_auroc),
            "rank_by_coverage@10": ">".join(str(x) for x in by_cov),
            "rank_by_aurc_lower_better": ">".join(by_aurc),
            "auroc_vs_coverage10_disagree": by_auroc != by_cov,
            "auroc_vs_aurc_disagree": by_auroc != by_aurc,
        })
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=("all", "camelyon17", "midog"),
        default="all",
    )
    parser.add_argument(
        "--official-zoo",
        action="store_true",
        help="Official 8 CNN stems only. Skips supcon/vit/native-384.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    official = {
        "resnet18",
        "resnet50",
        "densenet121",
        "convnext_tiny",
        "mobilenet_v3_large",
        "regnet_y_3_2gf",
        "effb3",
        "efficientnet_v2_s_224",
    }
    jobs = []
    if args.domain in ("all", "camelyon17"):
        for path in sorted(CAMELYON.glob("*_features.pt")):
            name = path.name.replace("_features.pt", "")
            if args.official_zoo and name not in official:
                continue
            jobs.append(("camelyon17", name, path))
    if args.domain in ("all", "midog"):
        for path in sorted(MIDOG.glob("*_features.pt")):
            name = path.name.replace("_features.pt", "")
            if "openmibood" in name or "native50" in name:
                continue
            if args.official_zoo and name not in official:
                continue
            jobs.append(("midog", name, path))
    rows = []
    for domain, model, path in jobs:
        rows.extend(process_cnn(domain, model, path))
    df = pd.DataFrame(rows)
    tag = args.domain
    if args.official_zoo:
        tag = f"{args.domain}_n8"
    out_csv = OUT / ("rc_by_model.csv" if tag == "all" else f"rc_by_model_{tag}.csv")
    df.to_csv(out_csv, index=False)
    msp = df[(df["protocol"] == "ood_triage") & (df["method"] == "msp")]
    msp_out = OUT / f"msp_ood_triage_{tag}.csv"
    msp[["domain", "model", "ood_auroc", "coverage_at_risk10", "aurc", "full_risk", "n"]].to_csv(
        msp_out, index=False
    )
    disagreement = rank_disagreement(df)
    disagreement.to_csv(OUT / f"auroc_vs_rc_rank_{tag}.csv", index=False)
    if args.official_zoo and args.domain == "camelyon17":
        note = OUT.parent / "camelyon_phao_b_n8.txt"
        cam = msp[msp["domain"] == "camelyon17"].sort_values("ood_auroc", ascending=False)
        by_auroc = list(cam["model"])
        by_cov = list(cam.sort_values("coverage_at_risk10", ascending=False)["model"])
        lines = [
            "Camelyon OOD-triage coverage@risk10% MSP, official n=8. Not a new official W.",
            "Locked Phao B n=4 table is unchanged. This is the leftover family span.",
            f"AUROC order: {' > '.join(by_auroc)}",
            f"coverage@risk10% order: {' > '.join(by_cov)}",
            f"disagree={by_auroc != by_cov}",
            "Do not write AUROC-winner=utility-loser as a law.",
            "",
            cam.to_csv(index=False),
        ]
        note.write_text("\n".join(lines))
        print(f"Wrote {note}")
    print("\n=== MSP OOD-triage ===")
    print(msp[["domain", "model", "ood_auroc", "coverage_at_risk10"]].to_string(index=False))
    print(f"\nWrote {out_csv}")
    print(f"Wrote {msp_out}")


if __name__ == "__main__":
    main()
