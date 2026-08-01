import os
import shutil
import pandas as pd

CSV_PATH = "data/reader_study/reader_cases_selected_96.csv"
OUTPUT_DIR = "reader_images_96"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

count = 0
for _, row in df.iterrows():
    src_path = row["image_path"]

    if not os.path.exists(src_path):
        print(f"❌ Missing: {src_path}")
        continue

    filename = os.path.basename(src_path)
    dst_path = os.path.join(OUTPUT_DIR, filename)

    shutil.copy(src_path, dst_path)
    count += 1

print(f"✅ Copied {count} images to {OUTPUT_DIR}")