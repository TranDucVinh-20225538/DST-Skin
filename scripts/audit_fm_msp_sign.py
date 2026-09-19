#!/usr/bin/env python3
"""CPU audit: is FM MSP<0.5 a sign-convention bug or real inversion?

Convention (src/utils/benchmark_metrics.calc_auroc):
  ID label=1, OOD label=0, higher score = more ID.
MSP/Energy in OODScorer are documented higher=ID.
If AUROC<0.5 under that convention, OOD is *more confident* than ID.
Flipping scores would just give 1-AUROC — we report both so a sign bug is visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.utils.scoring import OODScorer

ROOT = Path("outputs/features/fm")
OUT = Path("outputs/reports/fm")
CELLS = [
    ("midog", "uni"),
    ("midog", "gigapath"),
    ("midog", "phikon"),
    ("camelyon17", "uni"),
    ("camelyon17", "gigapath"),
    ("camelyon17", "phikon"),
]


def load_pt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def fit_logits(train_feats, train_labels):
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(train_feats, train_labels)
    W = np.asarray(clf.coef_, dtype=np.float32)
    b = np.asarray(clf.intercept_, dtype=np.float32)

    def logits(X):
        z = np.asarray(X, dtype=np.float32) @ W.T + b
        if z.ndim == 1 or (z.ndim == 2 and z.shape[1] == 1):
            z = np.asarray(z, dtype=np.float32).reshape(-1)
            return np.stack([-z, z], axis=1)
        return np.asarray(z, dtype=np.float32)

    return logits, float(clf.score(train_feats, train_labels)), int(clf.classes_.size)


def auroc(id_s, ood_s) -> float:
    y = np.concatenate([np.ones(len(id_s)), np.zeros(len(ood_s))])
    s = np.concatenate([id_s, ood_s])
    return float(roc_auc_score(y, s))


def summarize_scores(name, id_s, ood_s) -> dict:
    id_s = np.asarray(id_s, dtype=np.float64)
    ood_s = np.asarray(ood_s, dtype=np.float64)
    a = auroc(id_s, ood_s)
    a_flip = auroc(-id_s, -ood_s)
    return {
        "method": name,
        "id_mean": float(id_s.mean()),
        "ood_mean": float(ood_s.mean()),
        "id_median": float(np.median(id_s)),
        "ood_median": float(np.median(ood_s)),
        "ood_minus_id_mean": float(ood_s.mean() - id_s.mean()),
        "auroc_higher_is_id": a,
        "auroc_if_flipped": a_flip,
        "sum_to_one": abs(a + a_flip - 1.0) < 1e-6,
        "below_chance": a < 0.5,
        "ood_more_confident": float(ood_s.mean()) > float(id_s.mean()),
    }


def audit_one(domain: str, fm: str) -> dict | None:
    path = ROOT / domain / fm / "features.pt"
    if not path.exists():
        print(f"skip missing {path}", flush=True)
        return None
    data = load_pt(path)
    train_f = np.asarray(data["train_feats"], dtype=np.float32)
    train_y = np.asarray(data["train_labels"])
    val_f = np.asarray(data["val_feats"], dtype=np.float32)
    val_y = np.asarray(data["val_labels"])
    ood_f = np.asarray(data["ood_feats"], dtype=np.float32)
    ood_y = np.asarray(data["ood_labels"])
    logits_fn, train_acc, n_cls = fit_logits(train_f, train_y)
    val_z = logits_fn(val_f)
    ood_z = logits_fn(ood_f)
    msp = OODScorer.score_msp
    energy = OODScorer.score_energy
    val_pred = val_z.argmax(1)
    ood_pred = ood_z.argmax(1)
    mismatch = data.get("mismatch", {})
    rec = {
        "domain": domain,
        "fm": fm,
        "n_train": int(len(train_f)),
        "n_val": int(len(val_f)),
        "n_ood": int(len(ood_f)),
        "n_cls_probe": n_cls,
        "linear_train_acc": train_acc,
        "linear_val_acc": float((val_pred == val_y).mean()),
        "linear_ood_acc": float((ood_pred == ood_y).mean()),
        "mismatch_severity": mismatch.get("mismatch_severity"),
        "upsample_factor": mismatch.get("upsample_factor"),
    }
    rows = [
        summarize_scores("msp", msp(val_z), msp(ood_z)),
        summarize_scores("energy", energy(val_z), energy(ood_z)),
        summarize_scores("logit_norm", np.linalg.norm(val_z, axis=1), np.linalg.norm(ood_z, axis=1)),
    ]
    for row in rows:
        rec[f"{row['method']}_auroc"] = row["auroc_higher_is_id"]
        rec[f"{row['method']}_id_mean"] = row["id_mean"]
        rec[f"{row['method']}_ood_mean"] = row["ood_mean"]
        rec[f"{row['method']}_below_chance"] = row["below_chance"]
        rec[f"{row['method']}_ood_more_confident"] = row["ood_more_confident"]
        rec[f"{row['method']}_sum_to_one"] = row["sum_to_one"]
    rec["logit_family_all_below_chance"] = all(rec[f"{m}_below_chance"] for m in ("msp", "energy", "logit_norm"))
    rec["sign_bug_suspected"] = not all(rec[f"{m}_sum_to_one"] for m in ("msp", "energy", "logit_norm"))
    print(
        f"{domain}/{fm}: MSP {rec['msp_auroc']:.3f} "
        f"ID_mean={rec['msp_id_mean']:.4f} OOD_mean={rec['msp_ood_mean']:.4f} "
        f"val_acc={rec['linear_val_acc']:.3f} ood_acc={rec['linear_ood_acc']:.3f} "
        f"mismatch={rec['mismatch_severity']} "
        f"family_below={rec['logit_family_all_below_chance']} "
        f"sign_bug={rec['sign_bug_suspected']}",
        flush=True,
    )
    (OUT / domain / fm).mkdir(parents=True, exist_ok=True)
    (OUT / domain / fm / "msp_sign_audit.json").write_text(json.dumps({"cell": rec, "methods": rows}, indent=2))
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in (audit_one(d, f) for d, f in CELLS) if r is not None]
    if not rows:
        raise SystemExit("No FM feature files yet.")
    df = pd.DataFrame(rows)
    out = OUT / "msp_sign_audit.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(df[["domain", "fm", "msp_auroc", "energy_auroc", "logit_norm_auroc",
              "msp_below_chance", "logit_family_all_below_chance",
              "sign_bug_suspected", "mismatch_severity"]].to_string(index=False))


if __name__ == "__main__":
    main()
