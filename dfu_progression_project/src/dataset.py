import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DFUDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self.image_paths, self.labels = self.load_csv_data()

    def load_csv_data(self):
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file '{self.csv_path}' not found.")

        data = pd.read_csv(self.csv_path)

        if "image_path" not in data.columns or "label" not in data.columns:
            raise ValueError("CSV file must contain 'image_path' and 'label' columns.")

        image_paths = data["image_path"].tolist()
        labels = data["label"].tolist()

        invalid_image_paths = [
            img_path for img_path in image_paths if not os.path.isfile(img_path)
        ]
        if invalid_image_paths:
            raise FileNotFoundError(f"Invalid image paths found: {invalid_image_paths[:5]}...")

        labels = torch.LongTensor(labels)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image with PIL to ensure proper channel handling (RGB)
        image = Image.open(image_path).convert("RGB")
        
        # Keep uint8 so ToDtype(scale=True) in transforms scales to [0, 1].
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, label

