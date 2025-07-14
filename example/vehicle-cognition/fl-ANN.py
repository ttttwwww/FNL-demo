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
from dataset import get_uneven_fl_augment_dataset,get_uneven_fl_dataset
import matplotlib.pyplot as plt

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_fc1 = nn.Linear(25088,4096)
        self.img_fc2 = nn.Linear(4096,100)
        self.fuse_fc = nn.Linear(100+80, 3)

    def forward(self, x):
        img,audio = x
        img = torch.relu(self.img_fc1(img))
        img = torch.relu(self.img_fc2(img))
        audio = torch.flatten(audio,1)
        fuse_in = torch.cat((img,audio),1)
        out = torch.relu(self.fuse_fc(fuse_in))
        return out

BATCH_SIZE = 32
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

EPOCHS = 20

def run_epoch(model, dataloader, optimizer, criterion, train=True):
    if train:
        model.train()
    else:
        model.eval()
    acc = 0
    sample = 0
    pbar = tqdm(dataloader)
    for data,label in pbar:
        for i,item in enumerate(data):
            data[i] = item.to(device=DEVICE,dtype=torch.float32)
        label = label.to(device=DEVICE,dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        sample += len(label)
        acc += (output.argmax(dim=1) == label).sum().item()
        if train:
            loss.backward()
            optimizer.step()
        pbar.set_postfix(acc = acc/sample,loss=loss.item())
    return acc/sample,loss

dataset_portion =[
    [0.1,0.8,0.1],
    [0.8,0.1,0.1],
    [0.1,0.1,0.8],
]
# dataset_portion=[
#     [0.5,0.1,0.1],
#     [0.1,0.5,0.1],
#     [0.1,0.1,0.5],
#     [0.3,0.3,0.3],
# ]
dataset1,dataset2,dataset3,test_dataset = get_uneven_fl_augment_dataset(dataset_portion)
# dataset1,dataset2,dataset3,test_dataset = get_uneven_fl_dataset(dataset_portion)
train_dataset_list = [dataset1, dataset2, dataset3]

train_dataloader1 = DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True)
train_dataloader2 = DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True)
train_dataloader3 = DataLoader(dataset3, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model1 = MyNet().to(DEVICE)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
model2 = MyNet().to(DEVICE)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
model3 = MyNet().to(DEVICE)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)
global_model = MyNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
weight_change_dict = defaultdict(list)
train_acc1_list = []
train_acc2_list = []
train_acc3_list = []
train_loss1_list = []
train_loss2_list = []
train_loss3_list = []
test_acc1_list = []
test_acc2_list = []
test_acc3_list = []
test_loss1_list = []
test_loss2_list = []
test_loss3_list = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")

    # Train each model
    acc1,loss1 = run_epoch(model1, train_dataloader1, optimizer1, criterion, train=True)
    train_acc1_list.append(acc1)
    train_loss1_list.append(loss1.item())
    acc1,loss1 = run_epoch(model1, test_dataloader, optimizer1, criterion, train=False)
    test_acc1_list.append(acc1)
    test_loss1_list.append(loss1.item())
    print(f"Model 1 - Train Acc: {train_acc1_list[-1]:.4f}, Train Loss: {train_loss1_list[-1]:.4f}, Test Acc: {test_acc1_list[-1]:.4f}, Test Loss: {test_loss1_list[-1]:.4f}")

    acc2,loss2 = run_epoch(model2, train_dataloader2, optimizer2, criterion, train=True)
    train_acc2_list.append(acc2)
    train_loss2_list.append(loss2.item())
    acc2,loss2 = run_epoch(model2, test_dataloader, optimizer2, criterion, train=False)
    test_acc2_list.append(acc2)
    test_loss2_list.append(loss2.item())
    print(f"Model 2 - Train Acc: {train_acc2_list[-1]:.4f}, Train Loss: {test_loss2_list[-1]:.4f}, Test Acc: {test_acc2_list[-1]:.4f}, Test Loss: {test_loss2_list[-1]:.4f}")

    acc3,loss3 = run_epoch(model3, train_dataloader3, optimizer3, criterion, train=True)
    train_acc3_list.append(acc3)
    train_loss3_list.append(loss3.item())
    acc3,loss3 = run_epoch(model3, test_dataloader, optimizer3, criterion, train=False)
    test_acc3_list.append(acc3)
    test_loss3_list.append(loss3.item())
    print(f"Model 3 - Train Acc: {train_acc3_list[-1]:.4f}, Train Loss: {test_loss3_list[-1]:.4f}, Test Acc: {test_acc3_list[-1]:.4f}, Test Loss: {test_loss3_list[-1]:.4f}")
    # Aggregate models
    # Check weight changes
    changed1, total1 = compute_num_weight_change(model1.state_dict(), global_model.state_dict())
    changed2, total2 = compute_num_weight_change(model2.state_dict(), global_model.state_dict())
    changed3, total3 = compute_num_weight_change(model3.state_dict(), global_model.state_dict())
    weight_change_dict[f"model_{1}"].append(changed1)
    weight_change_dict[f"model_{2}"].append(changed2)
    weight_change_dict[f"model_{3}"].append(changed3)
    global_params = aggregate([model1, model2, model3])
    global_model.load_state_dict(global_params)
    model1.load_state_dict(global_params)
    model2.load_state_dict(global_params)
    model3.load_state_dict(global_params)

    # Evaluate global model
    print("Evaluating global model...")
    run_epoch(global_model, test_dataloader, optimizer1, criterion, train=False)
    print(f"Global Model - Test Acc: {test_acc1_list[-1]:.4f}, Test Loss: {test_loss1_list[-1]:.4f}")
curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_root = f"./log/-fl-ANN-{curr_time}"
save_name = os.path.join(save_root, f"model_1")
plot_result(save_name,train_acc1_list,train_loss1_list,test_acc1_list,test_loss1_list)
save_name = os.path.join(save_root, f"model_2")
plot_result(save_name,train_acc2_list, train_loss2_list, test_acc2_list, test_loss2_list)
save_name = os.path.join(save_root, f"model_3")
plot_result(save_name, train_acc3_list, train_loss3_list, test_acc3_list, test_loss3_list)
save_name = os.path.join(save_root, f"data")
plot_weight_change(save_name,weight_change_dict)
