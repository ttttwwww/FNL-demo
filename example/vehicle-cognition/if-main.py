import torch
from torch.utils.backcompat import keepdim_warning
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from snn_simulator.utils.logger import setup_logger
# examples/example1/main.py

# import os
# import sys
#
# sys.path.append(os.path.dirname(__file__))  # 添加当前路径到 sys.path

import torchvision

logger = setup_logger("snn_simulator", f"./log")
from dataset import get_default_img_dataset
from snn_simulator.dataset.single_dataset import ImageDataset, AugmentImageDataset
from snn_simulator.config.config_manager import ConfigManager
from snn_simulator.core import simulator
import torchvision.transforms as transforms
from torch import nn
from snn_simulator.models.function_layer import IFConv2D, IFTTFSLinear


class IFConvNet(nn.Module):
    def __init__(self, max_spike_time=6):
        super().__init__()
        self.max_spike_time = max_spike_time
        # [3,28,28]
        self.conv1 = IFConv2D(in_channel=3, out_channel=32, kernel_size=3, stride=1, padding=1, threshold=1,
                              max_spike_time=max_spike_time)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # [32,14,14]
        self.conv2 = IFConv2D(in_channel=32, out_channel=64, kernel_size=3, stride=1, padding=1, threshold=1,
                              max_spike_time=max_spike_time)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # [64,7,7]
        self.fc1 = IFTTFSLinear(64 * 7 * 7, 500, threshold=1, max_spike_time=max_spike_time)
        self.fc2 = IFTTFSLinear(500, 3, threshold=1, max_spike_time=max_spike_time)

    def forward(self, x):
        x = self.max_spike_time * (1 - x)
        x = torch.exp(torch.relu(x))
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    # TODO test if node for the vehicle test


train_size = 0.7
test_size = 0.3
BATCH_SIZE = 32
EPOCHS = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = IFConvNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # 假设输入128x128
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

raw_dataset = get_default_img_dataset()

train_dataset, test_dataset = random_split(raw_dataset, [train_size, test_size])
train_indices = train_dataset.indices
test_indices = test_dataset.indices

train_dataset = AugmentImageDataset(raw_dataset, train_indices, transform=train_transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    acc = 0
    samples = 0
    losses = 0
    pbar = tqdm(train_dataloader)
    model.train()
    for data, label in pbar:
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(-output, label)
        loss.backward()
        optimizer.step()
        acc += (output.argmin(1) == label).sum().item()
        samples += len(label)
        losses += loss.item()
        pbar.set_postfix({"loss": loss.item(), "acc": acc / samples})
    print(f"Epoch {epoch}/{EPOCHS}: acc={acc / samples:.4f}, loss={losses / samples:.4f}")
    model.eval()
    pbar = tqdm(test_dataloader)
    for data, label in pbar:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        acc += (output.argmax(1) == label).sum().item()
        samples += len(label)
        pbar.set_postfix({"acc": acc / samples})
    print(f"Test: acc={acc / samples:.4f}")
