import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        if "image" not in self.data.columns:
            raise ValueError("CSV must contain column 'image'")
        if "binary_label" not in self.data.columns:
            raise ValueError("CSV must contain column 'binary_label'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_id = str(row["image"])
        label = row["binary_label"]

        if not image_id.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_filename = f"{image_id}.jpg"
        else:
            image_filename = image_id

        img_path = os.path.join(self.image_dir, image_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label, img_path