import pandas as pd
import os

# ===== PATHS =====
train_gt = "data/raw/isic2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
val_gt = "data/raw/isic2018/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv"

output_dir = "data/processed/isic2018_binary"

def convert_to_binary(df):
    # Malignant = MEL + BCC + AKIEC
    df["binary_label"] = (
        df["MEL"] + df["BCC"] + df["AKIEC"]
    )

    # Nếu >0 thì malignant = 1
    df["binary_label"] = df["binary_label"].apply(lambda x: 1 if x > 0 else 0)

    return df[["image", "binary_label"]]

# ===== PROCESS TRAIN =====
train_df = pd.read_csv(train_gt)
train_binary = convert_to_binary(train_df)

train_binary.to_csv(
    os.path.join(output_dir, "isic_2018_binary_train.csv"),
    index=False
)

print("Train saved:", len(train_binary))

# ===== PROCESS VAL =====
val_df = pd.read_csv(val_gt)
val_binary = convert_to_binary(val_df)

val_binary.to_csv(
    os.path.join(output_dir, "isic_2018_binary_val.csv"),
    index=False
)

print("Val saved:", len(val_binary))
