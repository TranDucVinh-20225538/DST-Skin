# scripts/run_journal_benchmark_resnet50.py

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from src.utils.feature_extractor import extract_features_and_logits
from src.utils.scoring import OODScorer
from src.datasets.isic_dataset import ISICDataset
from src.models.resnet50 import get_resnet50   


# ====================== DATASET HELPERS ======================

def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_dataloader(split: str, batch_size: int = 64):
    """
    split: 'isic_train_subset', 'isic_val', 'pad_ufes'
    """
    tfm = build_transform()

    if split == "isic_train_subset":
        csv_path = "data/processed/isic2018_binary/isic_2018_binary_train.csv"
        img_root = "data/raw/isic2018/ISIC2018_Task3_Training_Input"
        full_ds = ISICDataset(csv_path, img_root, transform=tfm)

        np.random.seed(42)
        subset_idx = np.random.choice(len(full_ds), 5000, replace=False)
        ds = torch.utils.data.Subset(full_ds, subset_idx)

    elif split == "isic_val":
        csv_path = "data/processed/isic2018_binary/isic_2018_binary_val.csv"
        img_root = "data/raw/isic2018/ISIC2018_Task3_Validation_Input"
        ds = ISICDataset(csv_path, img_root, transform=tfm)

    elif split == "pad_ufes":
        csv_path = "data/processed/pad_ufes20_binary/pad_ufes_binary.csv"
        img_root = "data/raw/pad_ufes20/images"
        ds = ISICDataset(csv_path, img_root, transform=tfm)

    else:
        raise ValueError(f"Unknown split: {split}")

    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ====================== METRIC HELPERS ======================

def calculate_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray):
    """
    FPR@95%TPR, với convention: score cao = ID.
    Ngưỡng = percentile 5% của ID (tương ứng TPR≈95%).
    """
    threshold = np.percentile(id_scores, 5)
    fpr = np.mean(ood_scores > threshold)
    return fpr, threshold


# ====================== MAIN BENCHMARK ======================

def run_benchmark():
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"🚀 Running ResNet-50 benchmark on {device}")

    # 1. DATA
    train_loader = get_dataloader("isic_train_subset")
    val_loader   = get_dataloader("isic_val")
    ood_loader   = get_dataloader("pad_ufes")

    # 2. MODEL (ResNet-50 CE-Robust)
    model = get_resnet50(num_classes=2)
    model.load_state_dict(torch.load(
        "data/models/resnet50_robust_epoch_last.pth",  # <— chỉnh đúng tên checkpoint bạn train
        map_location=device
    ))
    model.to(device)

    # 3. EXTRACT FEATURES + LOGITS
    print("--- Extracting ISIC Train (for fit) ---")
    _, train_feats = extract_features_and_logits(model, train_loader, device)

    print("--- Extracting ISIC Val (ID) ---")
    id_logits, id_feats = extract_features_and_logits(model, val_loader, device)

    print("--- Extracting PAD-UFES (OOD) ---")
    ood_logits, ood_feats = extract_features_and_logits(model, ood_loader, device)

    # 4. SCORING
    scorer = OODScorer(k_nearest=50)
    scorer.fit(train_feats)

    id_scores  = scorer.get_all_scores(id_logits,  id_feats)
    ood_scores = scorer.get_all_scores(ood_logits, ood_feats)

    # 5. SUMMARY METRICS → CSV
    rows = []
    for method in id_scores.keys():
        if method.startswith("cvid"):
            continue  # không phải vector score

        id_s  = id_scores[method]
        ood_s = ood_scores[method]

        y_true  = np.concatenate([np.ones(len(id_s)), np.zeros(len(ood_s))])
        y_score = np.concatenate([id_s, ood_s])

        auroc = roc_auc_score(y_true, y_score)
        fpr95, thr = calculate_fpr95(id_s, ood_s)

        rows.append({
            "Backbone": "resnet50",
            "TrainMode": "CE_Robust",
            "Method": method,
            "AUROC": auroc,
            "FPR95": fpr95,
            "Threshold": thr,
        })

        print(f"🔥 {method:12s} | AUROC: {auroc:.4f} | FPR95: {fpr95:.4f}")

    df = pd.DataFrame(rows)
    out_path = "outputs/results_resnet50_robust.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Đã lưu kết quả: {out_path}")


if __name__ == "__main__":
    run_benchmark()
