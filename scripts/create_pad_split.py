"""Create binary PAD-UFES-20 CSV aligned with ISIC2018 binary task.

Malignant (1): MEL, BCC, SCC, ACK  (ACK matches ISIC AKIEC)
Benign   (0): NEV, SEK
"""

import os
import pandas as pd

METADATA_PATH = "data/raw/pad_ufes20/metadata.csv"
OUTPUT_PATH = "data/processed/pad_ufes20_binary/pad_ufes_binary.csv"

MALIGNANT = {"MEL", "BCC", "SCC", "ACK"}
BENIGN = {"NEV", "SEK"}


def to_binary(code: str) -> int:
    code = str(code).strip().upper()
    if code in MALIGNANT:
        return 1
    if code in BENIGN:
        return 0
    raise ValueError(f"Unknown diagnosis code: {code}")


def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Missing {METADATA_PATH}. Download PAD-UFES-20 first."
        )

    df = pd.read_csv(METADATA_PATH)
    if "img_id" not in df.columns or "diagnostic" not in df.columns:
        raise ValueError(
            f"Expected columns img_id and diagnostic, got {list(df.columns)}"
        )

    rows = []
    for _, row in df.iterrows():
        image_name = str(row["img_id"])
        if not image_name.lower().endswith(".png"):
            image_name = f"{image_name}.png"
        rows.append({
            "image": image_name,
            "binary_label": to_binary(row["diagnostic"]),
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(out)} rows to {OUTPUT_PATH}")
    print(out["binary_label"].value_counts().sort_index())
    print("By diagnostic:")
    print(df.assign(binary_label=[to_binary(x) for x in df["diagnostic"]])
              .groupby(["diagnostic", "binary_label"]).size())


if __name__ == "__main__":
    main()
