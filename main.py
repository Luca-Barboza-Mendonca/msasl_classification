import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import msasldataset
import matplotlib.pyplot as plt
import numpy as np
from model import SqueezeNet
from torchvision.utils import make_grid
from train import train_epoch
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = msasldataset.ImageSequenceDataset(csv_file=r".\labels_int.csv", folder=r".\train_frames", transform=transform)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

    model = SqueezeNet(sample_size=224, sample_duration=32, version=1.1, num_classes=101)
    model = model.cuda()

    optimizer = optim.SGD(
                model.parameters(),
                lr=1e-3,
                momentum=0.9,
                dampening=0,
                weight_decay=1e-3,
                nesterov=0.9)

    criterion = nn.CrossEntropyLoss()
    
    checkpoint = torch.load("model.pt")
    model.load_state_dict(checkpoint['mode_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    EPOCHS = 45

    for epoch in range(checkpoint['epoch'] + 1, EPOCHS):

        l = train_epoch(epoch=epoch, data_loader=dataloader, model=model, criterion=criterion, optimizer=optimizer)

        torch.save({
            'epoch': epoch,
            'mode_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': l
        }, "model.pt")