# scripts/dst_for_reader.py

import os
import numpy as np
import pandas as pd
import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.serialization import add_safe_globals

from src.utils.scoring import OODScorer
from src.models.efficientnet_b3 import get_efficientnet_b3
from src.utils.feature_extractor import extract_features_and_logits


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

FEATURE_FILE = "outputs/features/effb3_isic_pad_features.pt"
MODEL_CKPT = "data/models/efficientnet_b3_robust_best.pth"
READER_CASES_CSV = "data/reader_study/reader_cases_effb3.csv"
OUTPUT_CSV = "outputs/reports/reader_study_effb3_with_ai.csv"
BATCH_SIZE = 16


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_transform():
    return transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class ReaderStudyDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform or build_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        dummy_label = 0
        return img_tensor, dummy_label


def build_reader_reliability_flag(df):
    df = df.copy()

    df["reader_reliability_flag"] = "High"

    low_groups = {"OOD_low_maha", "ID_low_reliability_error"}
    df.loc[df["reader_group"].isin(low_groups), "reader_reliability_flag"] = "Low"

    mask_g4 = df["reader_group"] == "OOD_high_confident_wrong"
    if mask_g4.any():
        g4_scores = df.loc[mask_g4, "dstskin_maha"].values
        g4_median = np.median(g4_scores)
        df.loc[mask_g4, "reader_reliability_flag"] = np.where(
            df.loc[mask_g4, "dstskin_maha"] >= g4_median,
            "High",
            "Low"
        )

    return df


def main():
    set_seed(42)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    add_safe_globals([np._core.multiarray._reconstruct])

    print("🚀 Loading EfficientNet-B3 model...")
    model = get_efficientnet_b3(num_classes=2, pretrained=False)
    state = torch.load(MODEL_CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print("📦 Loading train features for OODScorer...")
    feat_data = torch.load(FEATURE_FILE, map_location="cpu", weights_only=False)
    train_feats = to_numpy(feat_data["train_feats"])
    train_logits = to_numpy(feat_data["train_logits"])
    fc_weight = to_numpy(feat_data["fc_weight"])
    fc_bias = to_numpy(feat_data["fc_bias"])

    print("🧮 Fitting OODScorer (Mahalanobis + k-NN)...")
    scorer = OODScorer(
        k_nearest=50,
        use_react=False,
        use_vim=False,
    )
    scorer.fit(
        train_features=train_feats,
        train_labels=None,
        train_logits=train_logits,
        fc_weight=fc_weight,
        fc_bias=fc_bias,
    )

    print("📄 Loading reader-study cases...")
    df_cases = pd.read_csv(READER_CASES_CSV)
    ds = ReaderStudyDataset(READER_CASES_CSV, transform=build_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    print("🏃 Running model on reader-study images...")
    reader_logits, reader_feats = extract_features_and_logits(
        model, loader, DEVICE, target_layer=None
    )

    logits_np = np.asarray(reader_logits)
    feats_np = np.asarray(reader_feats)

    print("  train_feats shape:", train_feats.shape)
    print("  reader_feats shape:", feats_np.shape)

    print("🧪 Computing DST-Skin (Mahalanobis) scores...")
    scores = scorer.get_all_scores(logits_np, feats_np)
    maha_scores = scores["mahalanobis"]

    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    ai_pred_idx = probs.argmax(axis=1)
    ai_pred_cls = np.where(ai_pred_idx == 1, "malignant", "benign")
    ai_prob_malignant = probs[:, 1]

    thr = np.percentile(maha_scores, 10)
    reliability_flag = np.where(maha_scores >= thr, "High", "Low")

    df_out = df_cases.copy()
    df_out["ai_pred"] = ai_pred_cls
    df_out["ai_prob_malignant"] = ai_prob_malignant
    df_out["dstskin_maha"] = maha_scores
    df_out["reliability_flag"] = reliability_flag

    df_out = build_reader_reliability_flag(df_out)

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved DST-Skin outputs for reader study to {OUTPUT_CSV}")
    print(df_out["reader_reliability_flag"].value_counts())
    print(df_out.groupby("reader_group")["reader_reliability_flag"].value_counts())


if __name__ == "__main__":
    main()