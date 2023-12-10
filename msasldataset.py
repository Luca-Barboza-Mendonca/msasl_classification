import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import json

class ImageSequenceDataset(Dataset):
    def __init__(self, csv_file, folder, transform):
        self.folder = folder
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_sequence_path = os.path.join(self.folder, self.annotations.iloc[idx, 0])
        images = []
        for i in range(0, 32):
            if i < 10:
                images.append(Image.open(os.path.join(img_sequence_path, f"frame_000{i}.jpg")))
            else:
                images.append(Image.open(os.path.join(img_sequence_path, f"frame_00{i}.jpg")))
        label = self.annotations.iloc[idx, 1]

        images = torch.stack([self.transform(img) for img in images], dim=0)

        return images.permute(1, 0, 2, 3), label

def get_class_map(file_path):
    file = open(file_path, 'r')

    content = file.read()

    class_list = json.loads(content)

    class_list = class_list[:101]

    class_map = {}

    for i in range(0, len(class_list)):
        class_map[class_list[i]] = i
    
    return class_map
