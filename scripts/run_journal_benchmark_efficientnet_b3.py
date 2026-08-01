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
from src.models.efficientnet_b3 import get_efficientnet_b3


# ====================== DATASET HELPERS ======================

def build_transform():
    return transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_dataloader(split: str, batch_size: int = 32):
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

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, img_root, split


# ====================== METRIC HELPERS ======================

def calculate_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray):
    """
    FPR@95%TPR, score cao = ID.
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
    print(f"🚀 Running EfficientNet-B3 benchmark on {device}")

    # 1. DATA
    train_loader, _, _ = get_dataloader("isic_train_subset")
    val_loader,   isic_img_root, _ = get_dataloader("isic_val")
    ood_loader,   pad_img_root,  _ = get_dataloader("pad_ufes")

    # 2. MODEL
    model = get_efficientnet_b3(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(
        "data/models/efficientnet_b3_robust_best.pth",
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

    id_scores_dict  = scorer.get_all_scores(id_logits,  id_feats)
    ood_scores_dict = scorer.get_all_scores(ood_logits, ood_feats)

    # 5a. SUMMARY METRICS (giữ nguyên file cũ)
    summary_rows = []
    for method in id_scores_dict.keys():
        if method.startswith("cvid"):
            continue

        id_s  = id_scores_dict[method]
        ood_s = ood_scores_dict[method]

        y_true  = np.concatenate([np.ones(len(id_s)), np.zeros(len(ood_s))])
        y_score = np.concatenate([id_s, ood_s])

        auroc = roc_auc_score(y_true, y_score)
        fpr95, thr = calculate_fpr95(id_s, ood_s)

        summary_rows.append({
            "Backbone": "efficientnet_b3",
            "TrainMode": "CE_Robust",
            "Method": method,
            "AUROC": auroc,
            "FPR95": fpr95,
            "Threshold": thr,
        })

        print(f"🔥 {method:12s} | AUROC: {auroc:.4f} | FPR95: {fpr95:.4f}")

    df_summary = pd.DataFrame(summary_rows)
    summary_path = "outputs/results_efficientnet_b3_robust.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"✅ Đã lưu summary: {summary_path}")

    # 5b. PER-SAMPLE CSV CHO GRAD-CAM (dùng 1 method chính, ví dụ 'mahalanobis_score')
    main_method = "mahalanobis"   # chỉnh theo đúng key trong scorer.get_all_scores
    if main_method not in id_scores_dict:
        raise ValueError(f"'{main_method}' không có trong score dict: {list(id_scores_dict.keys())}")

    # Lấy label & đường dẫn ảnh từ DataLoader
    def collect_image_meta(loader, img_root, dataset_name):
        img_paths = []
        labels = []
        preds = []
        for batch_imgs, batch_labels, batch_paths in loader:
            # ISICDataset nên trả về path hoặc id; nếu không, bạn sửa dataset để batch_paths là path.
            for p in batch_paths:
                img_paths.append(os.path.join(img_root, os.path.basename(p)))
            labels.extend(batch_labels.numpy().tolist())

            with torch.no_grad():
                logits = model(batch_imgs.to(device))
                probs  = torch.softmax(logits, dim=1)
                pred   = probs.argmax(dim=1).cpu().numpy().tolist()
                preds.extend(pred)

        meta_df = pd.DataFrame({
            "image_path": img_paths,
            "label": labels,
            "prediction": preds,
            "dataset": dataset_name,
        })
        return meta_df

    # Lưu ý: cần ISICDataset trả thêm đường dẫn ảnh trong __getitem__
    print("--- Collecting per-sample metadata ---")
    # Re-iterate val & ood loader với path
    val_loader_with_path, _, _ = get_dataloader("isic_val")
    ood_loader_with_path, _, _ = get_dataloader("pad_ufes")

    # Ở đây giả định DataLoader trả (img, label, path)
    id_meta  = collect_image_meta(val_loader_with_path, isic_img_root, "isic")
    ood_meta = collect_image_meta(ood_loader_with_path, pad_img_root, "pad_ufes")

    # Gán score theo thứ tự mẫu (phải cùng thứ tự như khi extract_features_and_logits)
    id_meta[main_method]  = id_scores_dict[main_method]
    ood_meta[main_method] = ood_scores_dict[main_method]

    df_per_sample = pd.concat([id_meta, ood_meta], ignore_index=True)
    per_sample_path = "outputs/efficientnet_b3_per_sample.csv"
    df_per_sample.to_csv(per_sample_path, index=False)
    print(f"✅ Đã lưu per-sample CSV cho Grad-CAM: {per_sample_path}")


if __name__ == "__main__":
    run_benchmark()