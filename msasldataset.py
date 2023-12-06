import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class ImageSequenceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_sequence_path = self.annotations.iloc[idx, 0]
        images = []
        for i in range(0, 32):
            if i < 10:
                images.append(Image.open(os.path.join(img_sequence_path, f"frame_000{i}.jpg")))
            else:
                images.append(Image.open(os.path.join(img_sequence_path, f"frame_00{i}.jpg")))
        # images = [Image.open(os.path.join(img_sequence_path, f"frame_00{i}.jpg")) for i in range(1, 33)]
        label = self.annotations.iloc[idx, 1]

        if self.transform:
            images = [self.transform(img) for img in images]

        return torch.stack(images), label