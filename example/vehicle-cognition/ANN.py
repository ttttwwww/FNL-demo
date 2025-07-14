import os
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import *
from dataset import get_uneven_fl_augment_dataset, get_uneven_fl_dataset, get_augmented_multimodal_dataset
import matplotlib.pyplot as plt


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_fc1 = nn.Linear(25088, 4096)
        self.img_fc2 = nn.Linear(4096, 100)
        self.fuse_fc = nn.Linear(100 + 80, 3)

    def forward(self, x):
        img, audio = x
        img = torch.relu(self.img_fc1(img))
        img = torch.relu(self.img_fc2(img))
        audio = torch.flatten(audio, 1)
        fuse_in = torch.cat((img, audio), 1)
        out = torch.relu(self.fuse_fc(fuse_in))
        return out


BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 20


def run_epoch(model, dataloader, optimizer, criterion, train=True):
    if train:
        model.train()
    else:
        model.eval()
    acc = 0
    sample = 0
    pbar = tqdm(dataloader)
    for data, label in pbar:
        for i, item in enumerate(data):
            data[i] = item.to(device=DEVICE, dtype=torch.float32)
        label = label.to(device=DEVICE, dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        sample += len(label)
        acc += (output.argmax(dim=1) == label).sum().item()
        if train:
            loss.backward()
            optimizer.step()
        pbar.set_postfix(acc=acc / sample, loss=loss.item())
    return acc / sample, loss


train_dataset, test_dataset = get_augmented_multimodal_dataset()


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MyNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

train_acc_list = []
train_loss_list = []

test_acc_list = []
test_loss_list = []


for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    # Train each model
    acc, loss = run_epoch(model, train_dataloader, optimizer, criterion, train=True)
    train_acc_list.append(acc)
    train_loss_list.append(loss.item())
    acc, loss = run_epoch(model, test_dataloader, optimizer, criterion, train=False)
    test_acc_list.append(acc)
    test_loss_list.append(loss.item())
    print(
        f"Model 1 - Train Acc: {train_acc_list[-1]:.4f}, Train Loss: {train_loss_list[-1]:.4f}, Test Acc: {test_acc_list[-1]:.4f}, Test Loss: {test_loss_list[-1]:.4f}")

curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_root = f"./log/ANN-{curr_time}"
save_name = os.path.join(save_root, f"model")
plot_result(save_name, train_acc_list, train_loss_list, test_acc_list, test_loss_list)
