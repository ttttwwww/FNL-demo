import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from util import *
from dataset import get_augmented_multimodal_dataset, get_default_multimodal_dataset, get_uneven_fl_augment_dataset
import torch
from torch import nn


class FuseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_fc1 = layer.Linear(25088, 4096)
        self.neuron1 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
        self.img_fc2 = layer.Linear(4096, 100)
        self.neuron2 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
        self.flatten = layer.Flatten()
        self.fuse_fc = layer.Linear(100 + 80, 3)
        self.neuron3 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())

        functional.set_step_mode(self, step_mode="m")

    def forward(self, x):
        img, audio = x
        img = self.img_fc1(img)
        img = self.neuron1(img)
        img = self.img_fc2(img)
        img = self.neuron2(img)
        audio = self.flatten(audio)
        fuse_in = torch.cat((img, audio), -1)
        out = self.neuron3(self.fuse_fc(fuse_in))
        return out.mean(0)

def run_epoch(model, dataloader, optimizer, criterion, train):
    if train:
        model.train()
    else:
        model.eval()
    acc = 0
    sample = 0
    pbar = tqdm(dataloader)
    for data, label in pbar:
        optimizer.zero_grad()
        for i, item in enumerate(data):
            item = item.to(device=DEVICE, dtype=torch.float32)
            shape = [1] * item.ndim
            data[i] = encoder(item.unsqueeze(0).repeat(T, *shape))
        label = label.to(device=DEVICE, dtype=torch.long)
        out_fr = model(data)
        loss = criterion(out_fr, label)
        sample += len(label)
        acc += (out_fr.argmax(dim=1) == label).sum().item()
        if train:
            loss.backward()
            optimizer.step()
        pbar.set_postfix(acc=acc / sample, loss=loss.item())
        functional.reset_net(model)
    return acc / sample, loss


def plot_result(save_root, train_acc, train_loss, test_acc, test_loss) -> None:
    """
    Plot accuracy and loss curves after training.
    Args:
        save_root: Root dir for saving.
        train_acc: Train accuracy after training.
        train_loss: Train loss after training.
        test_acc: Test accuracy after training.
        test_loss: Test loss after training.
    """
    save_path = os.path.join(save_root, "fig/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.plot(train_acc, label="train acc")
    plt.title("Train Accuracy")
    plt.savefig(os.path.join(save_root, "fig/train_acc.png"))
    plt.show()
    plt.clf()
    plt.plot(train_loss, label="train loss")
    plt.title("Train Loss")
    plt.savefig(os.path.join(save_root, "fig/train_loss.png"))
    plt.show()
    plt.clf()

    plt.plot(test_acc, label="test acc")
    plt.title("Test Accuracy")
    plt.savefig(os.path.join(save_root, "fig/test_acc.png"))
    plt.show()
    plt.clf()
    plt.plot(test_loss, label="test loss")
    plt.title("Test Loss")
    plt.savefig(os.path.join(save_root, "fig/test_loss.png"))
    plt.show()
    plt.clf()

    save_path = os.path.join(save_root, "data/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savetxt(os.path.join(save_path, "train_acc.csv"), train_acc, delimiter=",")
    np.savetxt(os.path.join(save_path, "train_loss.csv"), train_loss, delimiter=",")
    np.savetxt(os.path.join(save_path, "test_acc.csv"), test_acc, delimiter=",")
    np.savetxt(os.path.join(save_path, "test_loss.csv"), test_loss, delimiter=",")


def plot_weight_change(save_root, weight_change_dict):
    """
    Plot the weight change of each simulator.
    """
    plt.figure(figsize=(10, 5))
    for key, value in weight_change_dict.items():
        plt.plot(value, label=key)
    total_change = np.sum(np.array(weight_change_dict.values()), axis=0)
    plt.plot(total_change, label="Total change")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Changed Weights")
    plt.title("Weight Change During Training")
    plt.legend()
    save_path = os.path.join(save_root, "fig/weight_change.png")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.show()

    save_path = os.path.join(save_root, "data/weight_change.csv")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df = pd.DataFrame(weight_change_dict)
    df.to_csv(save_path, index=False)

BATCH_SIZE = 8
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
T = 100

dataset_portion = [
    [0.1, 0.8, 0.1],
    [0.8, 0.1, 0.1],
    [0.1, 0.1, 0.8],
]
dataset1, dataset2, dataset3, test_dataset = get_uneven_fl_augment_dataset(dataset_portion)
train_dataloader1 = DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
train_dataloader2 = DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
train_dataloader3 = DataLoader(dataset3, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

model1 = FuseNet().to(DEVICE)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
model2 = FuseNet().to(DEVICE)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
model3 = FuseNet().to(DEVICE)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)
global_model = FuseNet().to(DEVICE)

encoder = encoding.PoissonEncoder()
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
save_root = f"./log/fl-rc-{curr_time}"
save_name = os.path.join(save_root, f"model_1")
plot_result(save_name,train_acc1_list,train_loss1_list,test_acc1_list,test_loss1_list)
save_name = os.path.join(save_root, f"model_2")
plot_result(save_name,train_acc2_list, train_loss2_list, test_acc2_list, test_loss2_list)
save_name = os.path.join(save_root, f"model_3")
plot_result(save_name, train_acc3_list, train_loss3_list, test_acc3_list, test_loss3_list)
save_name = os.path.join(save_root, f"data")
plot_weight_change(save_name,weight_change_dict)
