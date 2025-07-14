import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from spikingjelly.activation_based import neuron,encoding,functional,surrogate,layer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_augmented_multimodal_dataset
import torch
from torch import nn
import seaborn as sns

def plot_output(spike_train):
    spike_train = spike_train[:,0,:].detach().cpu().numpy()
    sns.heatmap(spike_train)
    plt.show()

class FuseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_fc1 = layer.Linear(25088, 4096)
        self.neuron1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.img_fc2 = layer.Linear(4096, 100)
        self.neuron2 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.flatten = layer.Flatten()
        self.fuse_fc = layer.Linear(100+80, 3)
        self.neuron3 = neuron.LIFNode(surrogate_function=surrogate.ATan())

        functional.set_step_mode(self,step_mode="m")

    def forward(self,x):
        img, audio = x
        img = self.img_fc1(img)
        img = self.neuron1(img)
        img = self.img_fc2(img)
        img = self.neuron2(img)
        audio = self.flatten(audio)
        fuse_in = torch.cat((img, audio), -1)
        out = self.neuron3(self.fuse_fc(fuse_in))
        return out.mean(0)

BATCH_SIZE = 32
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
T = 100
train_dataset,test_dataset = get_augmented_multimodal_dataset()
train_dataloader =DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


model = FuseNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
encoder = encoding.PoissonEncoder()
criterion = nn.CrossEntropyLoss()


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
            shape = [1]*item.ndim
            data[i] = encoder(item.unsqueeze(0).repeat(T,*shape))
        label = label.to(device=DEVICE, dtype=torch.long)
        out_fr = model(data)
        loss = criterion(out_fr, label)
        sample += len(label)
        acc += (out_fr.argmax(dim=1) == label).sum().item()
        if train:
            loss.backward()
            optimizer.step()
        pbar.set_postfix(acc=acc/sample, loss=loss.item())
        functional.reset_net(model)
    return acc/sample, loss



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

def plot_weight_change(save_root,weight_change_dict):
    """
    Plot the weight change of each simulator.
    """
    plt.figure(figsize=(10, 5))
    for key,value in weight_change_dict.items():
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

train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []

for epoch in range(EPOCHS):
    train_acc, train_loss = run_epoch(model, train_dataloader, optimizer, criterion, train=True)
    print(f"Epoch {epoch+1}/{EPOCHS}: Train acc={train_acc:.4f}, Train loss={train_loss:.4f}")
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss.item())

    test_acc, test_loss = run_epoch(model, test_dataloader, optimizer, criterion, train=False)
    print(f"Epoch {epoch+1}/{EPOCHS}: Test acc={test_acc:.4f}, Test loss={test_loss:.4f}")
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss.item())

plot_result("./rc_results", train_acc_list, train_loss_list, test_acc_list, test_loss_list)
