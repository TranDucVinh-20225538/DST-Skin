# scripts/select_reader_cases_v2.py

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

# =========================
# CONFIG
# =========================
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

FEATURE_FILE = "outputs/features/effb3_isic_pad_features.pt"
MODEL_CKPT = "data/models/efficientnet_b3_robust_best.pth"

CANDIDATE_CSV = "outputs/efficientnet_b3_per_sample.csv"

OUTPUT_FULL_CSV = "outputs/reports/reader_candidate_pool_with_ai.csv"
OUTPUT_SELECTED_CSV = "data/reader_study/reader_cases_selected_96.csv"

BATCH_SIZE = 16
SEED = 42

# =========================
# TARGET SPLIT (96 CASES)
# =========================
TARGET_COUNTS = {
    "SAFE_TRUSTED": 24,
    "ID_UNCERTAIN": 0,   
    "DANGEROUS_TRAP": 24,   
    "OOD_SHOULD_DEFER": 24,
    "OOD_TRAP_RESCUE": 24,  
}
TARGET_TOTAL = 96

# =========================
# THRESHOLDS
# =========================
HIGH_CONF = 0.80
MID_CONF = 0.60

OOD_POSITIVE_VALUES = {"OOD", "ood", 1, "1", True}


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
        return img_tensor, 0  # dummy label


# =========================
# HELPERS
# =========================
def infer_id_ood_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "id_ood_flag" in df.columns:
        return df

    if "dataset" in df.columns:
        df["id_ood_flag"] = np.where(
            df["dataset"].astype(str)
            .str.contains("PAD|UFES|ood", case=False, na=False),
            "OOD",
            "ID",
        )
        return df

    if "source" in df.columns:
        df["id_ood_flag"] = np.where(
            df["source"].astype(str)
            .str.contains("PAD|UFES|ood", case=False, na=False),
            "OOD",
            "ID",
        )
        return df

    df["id_ood_flag"] = "ID"
    return df


def normalize_gt_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "gt_label" in df.columns:
        col = "gt_label"
    elif "label" in df.columns:
        df["gt_label"] = df["label"]
        col = "gt_label"
    elif "target" in df.columns:
        df["gt_label"] = df["target"]
        col = "gt_label"
    else:
        raise ValueError("CSV phải có gt_label / label / target")

    def _norm(x):
        x = str(x).strip().lower()
        if x in {"1", "malignant", "melanoma", "bcc", "scc", "cancer"}:
            return "malignant"
        return "benign"

    df["gt_label"] = df[col].apply(_norm)
    return df


def compute_reliability_flag(df: pd.DataFrame, percentile: int = 15):
    """
    Mahalanobis của bạn là âm:
    - Giá trị thấp (âm nhiều) = Low reliability
    - Giá trị cao (ít âm hơn) = High reliability
    """
    df = df.copy()
    thr = np.percentile(df["dstskin_maha"], percentile)
    df["reliability_flag"] = np.where(
        df["dstskin_maha"] >= thr, "High", "Low"
    )
    return df, thr


def build_behavior_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ai_prob_pred"] = np.where(
        df["ai_pred"] == "malignant",
        df["ai_prob_malignant"],
        1.0 - df["ai_prob_malignant"],
    )

    df["ai_correct"] = (df["ai_pred"] == df["gt_label"]).astype(int)
    df["id_ood_flag"] = df["id_ood_flag"].astype(str).str.strip().str.upper()

    df["is_ood"] = df["id_ood_flag"].isin(["OOD", "PAD", "PAD-UFES", "PAD-UFES-20"])

    df["behavior_group"] = "UNASSIGNED"

    # ƯU TIÊN THỨ TỰ GÁN NHÓM:
    # 1) OOD_TRAP_RESCUE (hẹp nhất)
    mask_ood_trap_rescue = (
        df["is_ood"]
        & (df["ai_correct"] == 0)
        & (df["ai_prob_pred"] >= MID_CONF)
        & (df["reliability_flag"] == "Low")
    )
    df.loc[mask_ood_trap_rescue, "behavior_group"] = "OOD_TRAP_RESCUE"

    # 2) OOD_SHOULD_DEFER
    mask_defer = df["is_ood"] & (df["reliability_flag"] == "Low")
    df.loc[mask_defer & (df["behavior_group"] == "UNASSIGNED"),
           "behavior_group"] = "OOD_SHOULD_DEFER"

    

    # 3) ID_UNCERTAIN (ID, sai, Low R)
    mask_id_uncertain = (
    (~df["is_ood"])
    & (df["reliability_flag"] == "Low")
)
    df.loc[mask_id_uncertain & (df["behavior_group"] == "UNASSIGNED"),
           "behavior_group"] = "ID_UNCERTAIN"
    # 4) SAFE_TRUSTED (ID, đúng, high conf, high R)
    mask_safe = (
        ~df["is_ood"]
        & (df["ai_correct"] == 1)
        & (df["ai_prob_pred"] >= HIGH_CONF)
        & (df["reliability_flag"] == "High")
    )
    df.loc[mask_safe & (df["behavior_group"] == "UNASSIGNED"),
           "behavior_group"] = "SAFE_TRUSTED"
    # 5) DANGEROUS_TRAP (còn lại, ai sai + prob cao)
    mask_trap = (
        (df["ai_correct"] == 0)
        & (df["ai_prob_pred"] >= MID_CONF)
        & (df["behavior_group"] == "UNASSIGNED")
    )
    df.loc[mask_trap, "behavior_group"] = "DANGEROUS_TRAP"

    # danger_score: ưu tiên misclassified, high‑conf, maha cao
    maha_min = df["dstskin_maha"].min()
    maha_max = df["dstskin_maha"].max()
    maha_norm = (df["dstskin_maha"] - maha_min) / (
        (maha_max - maha_min) + 1e-8
    )

    df["danger_score"] = (
        0.7 * df["ai_prob_pred"]
        + 0.3 * maha_norm
        - 0.5 * df["ai_correct"]
    )

    return df


def sample_group(df, group_name, n, random_state=42, sort_by=None, ascending=False):
    sub = df[df["behavior_group"] == group_name].copy()
    if len(sub) == 0:
        print(f"⚠️ Group {group_name} has 0 cases.")
        return sub

    if sort_by is not None:
        sub = sub.sort_values(sort_by, ascending=ascending)
        return sub.head(min(n, len(sub)))

    if len(sub) <= n:
        return sub

    return sub.sample(n=n, random_state=random_state)


def dedup_concat(dfs):
    return (
        pd.concat(dfs, axis=0)
        .drop_duplicates(subset=["image_path"])
        .reset_index(drop=True)
    )


def fill_to_target(df_all, df_selected, target_total, random_state=42):
    selected_paths = set(df_selected["image_path"].tolist())
    remaining = df_all[~df_all["image_path"].isin(selected_paths)].copy()

    priority_order = [
        "DANGEROUS_TRAP",
        "OOD_TRAP_RESCUE",
        "OOD_SHOULD_DEFER",
        "SAFE_TRUSTED",
        "ID_UNCERTAIN",
        "UNASSIGNED",
    ]

    filled = [df_selected]

    for g in priority_order:
        need = target_total - sum(len(x) for x in filled)
        if need <= 0:
            break

        sub = remaining[remaining["behavior_group"] == g].copy()
        if len(sub) == 0:
            continue

        sub = sub.sort_values("danger_score", ascending=False)
        take = sub.head(min(need, len(sub)))
        filled.append(take)

        selected_paths.update(take["image_path"].tolist())
        remaining = remaining[~remaining["image_path"].isin(selected_paths)].copy()

    out = dedup_concat(filled)

    if len(out) < target_total and len(remaining) > 0:
        need = target_total - len(out)
        extra = remaining.sample(
            n=min(need, len(remaining)), random_state=random_state
        )
        out = dedup_concat([out, extra])

    return out


def main():
    set_seed(SEED)
    add_safe_globals([np._core.multiarray._reconstruct])

    os.makedirs(os.path.dirname(OUTPUT_FULL_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_SELECTED_CSV), exist_ok=True)

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

    print("🧮 Fitting OODScorer...")
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

    print("📄 Loading candidate pool...")
    df_cases = pd.read_csv(CANDIDATE_CSV)
    df_cases = normalize_gt_label(df_cases)
    df_cases = infer_id_ood_flag(df_cases)

    ds = ReaderStudyDataset(CANDIDATE_CSV, transform=build_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    print("🏃 Running model inference on candidate pool...")
    reader_logits, reader_feats = extract_features_and_logits(
        model, loader, DEVICE, target_layer=None
    )

    logits_np = np.asarray(reader_logits)
    feats_np = np.asarray(reader_feats)

    print("🧪 Computing Mahalanobis scores...")
    scores = scorer.get_all_scores(logits_np, feats_np)
    maha_scores = scores["mahalanobis"]

    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    ai_pred_idx = probs.argmax(axis=1)
    ai_pred_cls = np.where(ai_pred_idx == 1, "malignant", "benign")
    ai_prob_malignant = probs[:, 1]

    df_out = df_cases.copy()
    df_out["ai_pred"] = ai_pred_cls
    df_out["ai_prob_malignant"] = ai_prob_malignant
    df_out["dstskin_maha"] = maha_scores

    df_out, maha_thr = compute_reliability_flag(df_out, percentile=15)
    df_out = build_behavior_groups(df_out)
    print("\n=== DEBUG: ID/OOD x Reliability ===")
    print(pd.crosstab(df_out["id_ood_flag"], df_out["reliability_flag"]))

    print("\n=== DEBUG: is_ood counts ===")
    print(df_out["is_ood"].value_counts())

    print("\n=== DEBUG: ID_UNCERTAIN candidates ===")
    tmp = df_out[(~df_out["is_ood"]) & (df_out["reliability_flag"] == "Low")]
    print(f"ID_UNCERTAIN raw candidates = {len(tmp)}")
    print(f"📌 Mahalanobis reliability threshold (15th percentile): {maha_thr:.4f}")
    print("\n=== ALL CANDIDATE COUNTS ===")
    print(df_out["behavior_group"].value_counts(dropna=False))

    df_out.to_csv(OUTPUT_FULL_CSV, index=False)
    print(f"\n💾 Saved full candidate pool with AI outputs to: {OUTPUT_FULL_CSV}")

    # ============ SELECT TARGET CASES ============
    selected_parts = [
        sample_group(df_out, "SAFE_TRUSTED",
                     TARGET_COUNTS["SAFE_TRUSTED"], random_state=SEED),
        sample_group(df_out, "ID_UNCERTAIN",
                     TARGET_COUNTS["ID_UNCERTAIN"], random_state=SEED),
        sample_group(
            df_out,
            "DANGEROUS_TRAP",
            TARGET_COUNTS["DANGEROUS_TRAP"],
            random_state=SEED,
            sort_by="danger_score",
            ascending=False,
        ),
        sample_group(df_out, "OOD_SHOULD_DEFER",
                     TARGET_COUNTS["OOD_SHOULD_DEFER"], random_state=SEED),
        sample_group(
            df_out,
            "OOD_TRAP_RESCUE",
            TARGET_COUNTS["OOD_TRAP_RESCUE"],
            random_state=SEED,
            sort_by="danger_score",
            ascending=False,
        ),
    ]

    df_selected = dedup_concat(selected_parts)
    df_selected = fill_to_target(df_out, df_selected, TARGET_TOTAL, random_state=SEED)

    if len(df_selected) > TARGET_TOTAL:
        priority = {
            "OOD_TRAP_RESCUE": 5,
            "DANGEROUS_TRAP": 4,
            "OOD_SHOULD_DEFER": 3,
            "ID_UNCERTAIN": 2,
            "SAFE_TRUSTED": 1,
            "UNASSIGNED": 0,
        }
        df_selected["priority_rank"] = df_selected["behavior_group"].map(priority).fillna(0)
        df_selected = df_selected.sort_values(
            ["priority_rank", "danger_score"],
            ascending=[False, False],
        ).head(TARGET_TOTAL)

    final_cols = [
        "image_path",
        "gt_label",
        "id_ood_flag",
        "ai_pred",
        "ai_prob_malignant",
        "ai_prob_pred",
        "ai_correct",
        "dstskin_maha",
        "reliability_flag",
        "behavior_group",
        "danger_score",
    ]
    final_cols = [c for c in final_cols if c in df_selected.columns]

    df_selected = df_selected[final_cols].reset_index(drop=True)
    df_selected["case_id"] = [f"RS_{i:03d}" for i in range(1, len(df_selected) + 1)]
    ordered_cols = ["case_id"] + [c for c in df_selected.columns if c != "case_id"]
    df_selected = df_selected[ordered_cols]

    df_selected.to_csv(OUTPUT_SELECTED_CSV, index=False)

    print("\n=== FINAL SELECTED COUNTS ===")
    print(df_selected["behavior_group"].value_counts(dropna=False))
    print(f"\n✅ Saved final reader-study case set to: {OUTPUT_SELECTED_CSV}")
    print(f"📊 Final total = {len(df_selected)} cases")


if __name__ == "__main__":
    main()