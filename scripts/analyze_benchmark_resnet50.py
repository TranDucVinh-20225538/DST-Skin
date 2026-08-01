import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.serialization import add_safe_globals

from src.utils.scoring import OODScorer


DEVICE = torch.device("cpu")
FEATURE_FILE = "outputs/features/resnet50_isic_pad_features.pt"
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def calc_auroc(id_scores, ood_scores):
    """
    Quy ước:
      - score cao = ID
      - positive class = ID
    """
    id_scores = to_numpy(id_scores).astype(np.float32)
    ood_scores = to_numpy(ood_scores).astype(np.float32)

    y_true = np.concatenate([
        np.ones(len(id_scores), dtype=np.int32),
        np.zeros(len(ood_scores), dtype=np.int32),
    ])
    y_score = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(y_true, y_score)


def calc_fpr95(id_scores, ood_scores):
    """
    FPR@95TPR với quy ước score cao = ID.
    threshold = percentile 5% của ID score
    """
    id_scores = to_numpy(id_scores).astype(np.float32)
    ood_scores = to_numpy(ood_scores).astype(np.float32)

    thr = np.percentile(id_scores, 5)
    fpr = np.mean(ood_scores > thr)
    return float(fpr), float(thr)


def risk_coverage(id_scores, id_correct):
    """
    id_scores: score cao = đáng tin hơn
    id_correct: 1 nếu dự đoán đúng trên ID val
    """
    id_scores = to_numpy(id_scores).astype(np.float32)
    id_correct = to_numpy(id_correct).astype(np.float32)

    order = np.argsort(-id_scores)  # score cao -> auto trước
    scores_sorted = id_scores[order]
    correct_sorted = id_correct[order]

    N = len(scores_sorted)
    coverages = []
    risks = []
    accepted = []

    for k in range(1, N + 1):
        cov = k / N
        subset_correct = correct_sorted[:k]
        risk = 1.0 - subset_correct.mean()

        coverages.append(cov)
        risks.append(risk)
        accepted.append(k)

    return pd.DataFrame({
        "accepted": accepted,
        "coverage": coverages,
        "risk": risks,
    })


def extract_coverage_at_target_risk(df_rc, target_risk=0.15):
    """
    Tìm coverage lớn nhất sao cho risk <= target_risk
    """
    valid = df_rc[df_rc["risk"] <= target_risk]
    if len(valid) == 0:
        return None
    best_row = valid.iloc[-1]
    return {
        "target_risk": float(target_risk),
        "coverage": float(best_row["coverage"]),
        "accepted": int(best_row["accepted"]),
        "risk": float(best_row["risk"]),
    }


def main():
    set_seed(SEED)
    os.makedirs("outputs/reports", exist_ok=True)

    # Cho PyTorch 2.6 load được file .pt chứa numpy
    add_safe_globals([np.core.multiarray._reconstruct])

    # 1. Load features/logits đã extract sẵn
    data = torch.load(FEATURE_FILE, map_location=DEVICE, weights_only=False)

    train_logits = to_numpy(data["train_logits"])
    train_feats = to_numpy(data["train_feats"])
    train_labels = to_numpy(data["train_labels"])

    val_logits = to_numpy(data["val_logits"])
    val_feats = to_numpy(data["val_feats"])
    val_labels = to_numpy(data["val_labels"])

    ood_logits = to_numpy(data["ood_logits"])
    ood_feats = to_numpy(data["ood_feats"])

    # cần có trong file .pt
    fc_weight = to_numpy(data["fc_weight"])
    fc_bias = to_numpy(data["fc_bias"])

    print("Train feats shape:", train_feats.shape)
    print("Val feats shape  :", val_feats.shape)
    print("OOD feats shape  :", ood_feats.shape)

    # 2. Fit OODScorer một lần trên toàn bộ train_feats (bật ReAct + ViM)
    print("\n=== Fit OODScorer trên full train (ReAct + ViM) ===")
    scorer_full = OODScorer(
        k_nearest=50,
        use_react=True,
        react_percentile=90.0,
        use_vim=True,
        vim_dim=None,   # auto = D - C
    )

    scorer_full.fit(
        train_feats,
        train_labels=None,
        train_logits=train_logits,
        fc_weight=fc_weight,
        fc_bias=fc_bias,
    )

    # 3. Tính score cho ID val và OOD bằng scorer_full
    scores_id_full = scorer_full.get_all_scores(val_logits, val_feats)
    scores_ood_full = scorer_full.get_all_scores(ood_logits, ood_feats)

    # =========================
    # 5.2 SCORE COMPARISON
    # =========================
    print("\n=== Score comparison (ResNet-50, OODScorer + ViM/ReAct) ===")

    base_methods = [
        "msp",
        "energy",
        "react_energy",
        "logit_norm",
        "mahalanobis",
        "knn",
        "vim",
    ]
    methods = [m for m in base_methods if m in scores_id_full]

    rows = []

    for method in methods:
        id_s = scores_id_full[method]
        ood_s = scores_ood_full[method]

        auroc = calc_auroc(id_s, ood_s)
        fpr95, thr = calc_fpr95(id_s, ood_s)

        if method == "mahalanobis":
            tp = (id_s > thr).sum()
            fp = (ood_s > thr).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            print(f"Mahalanobis: tp={tp}, fp={fp}, precision={precision:.6f}")
            print(f"len(id_scores)={len(id_s)}, len(ood_scores)={len(ood_s)}")

        rows.append({
            "Method": method,
            "AUROC": float(auroc),
            "FPR95": float(fpr95),
            "Threshold": float(thr),
        })

        print(f"{method:18s} | AUROC={auroc:.4f} | FPR95={fpr95:.4f}")

    df_scores = pd.DataFrame(rows)
    df_scores.to_csv("outputs/reports/resnet50_score_comparison.csv", index=False)

    # =========================
    # 5.6.2 ABLATION: N TRAIN (Mahalanobis gốc)
    # =========================
    print("\n=== Ablation: N train samples for OODScorer (Mahalanobis) ===")

    n_list = [100, 500, 1000, 2500, 5000]
    results_ablation = []

    for n in n_list:
        if n > len(train_feats):
            continue

        idx = np.random.choice(len(train_feats), n, replace=False)
        feats_sub = train_feats[idx]

        # Fit một OODScorer mới trên subset (không bật ViM/ReAct để giữ đúng so sánh cũ)
        scorer_n = OODScorer(k_nearest=50)
        scorer_n.fit(feats_sub)

        scores_id_n = scorer_n.get_all_scores(val_logits, val_feats)
        scores_ood_n = scorer_n.get_all_scores(ood_logits, ood_feats)

        id_s = scores_id_n["mahalanobis"]
        ood_s = scores_ood_n["mahalanobis"]

        auroc_n = calc_auroc(id_s, ood_s)
        fpr95_n, _ = calc_fpr95(id_s, ood_s)

        results_ablation.append({
            "N_train": int(n),
            "AUROC": float(auroc_n),
            "FPR95": float(fpr95_n),
        })

        print(f"N={n:5d} | AUROC={auroc_n:.4f} | FPR95={fpr95_n:.4f}")

    df_ablation = pd.DataFrame(results_ablation)
    df_ablation.to_csv("outputs/reports/resnet50_mahalanobis_ablation_N.csv", index=False)

    # =========================
    # 5.3 RISK-COVERAGE (Mahalanobis)
    # =========================
    print("\n=== Risk-Coverage data (ResNet-18, Mahalanobis via OODScorer) ===")

    probs_val = torch.softmax(torch.from_numpy(val_logits).float(), dim=1).numpy()
    preds_val = probs_val.argmax(axis=1)
    id_correct = (preds_val == val_labels).astype(np.float32)

    id_scores_maha = scores_id_full["mahalanobis"]
    df_rc = risk_coverage(id_scores_maha, id_correct)
    df_rc.to_csv("outputs/reports/resnet50_risk_coverage_mahalanobis.csv", index=False)

    coverage_10 = extract_coverage_at_target_risk(df_rc, target_risk=0.10)
    coverage_15 = extract_coverage_at_target_risk(df_rc, target_risk=0.15)
    coverage_20 = extract_coverage_at_target_risk(df_rc, target_risk=0.20)

    summary_rc = []
    for item in [coverage_10, coverage_15, coverage_20]:
        if item is not None:
            summary_rc.append(item)

    df_rc_summary = pd.DataFrame(summary_rc)
    df_rc_summary.to_csv(
        "outputs/reports/resne50_risk_coverage_summary.csv", index=False
    )

    print("\n=== Coverage @ Target Risk (Mahalanobis) ===")
    if coverage_10 is not None:
        print(f"Risk<=10%  -> Coverage={coverage_10['coverage']:.4f}")
    if coverage_15 is not None:
        print(f"Risk<=15%  -> Coverage={coverage_15['coverage']:.4f}")
    if coverage_20 is not None:
        print(f"Risk<=20%  -> Coverage={coverage_20['coverage']:.4f}")

    print("\n✅ Saved files:")
    print(" - outputs/reports/resnet50_score_comparison.csv")
    print(" - outputs/reports/resnet50_mahalanobis_ablation_N.csv")
    print(" - outputs/reports/resnet50_risk_coverage_mahalanobis.csv")
    print(" - outputs/reports/resnet50_risk_coverage_summary.csv")


if __name__ == "__main__":
    main()