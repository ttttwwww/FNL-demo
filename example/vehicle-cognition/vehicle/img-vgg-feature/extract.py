import numpy as np
import torch
from torchvision.models import vgg11,VGG11_Weights
from torch import nn
from torch.utils.data import DataLoader
import os
from dataset import VehicleImageDataset

def make_feature_dataset(model, data_loader, device):
    save_features = None
    save_labels = None
    for img, label in data_loader:
        img = img.to(device,dtype=torch.float32)
        label = label.to(device,dtype=torch.long)
        B = img.shape[0]
        feature = model(img).view(B, -1)  # Flatten the feature map to [B, C*H*W]
        feature = feature.detach().cpu().numpy()
        if save_features is None:
            save_features = feature
        else:
            save_features = np.vstack((save_features, feature))
        if save_labels is None:
            save_labels = label.detach().cpu().numpy()
        else:
            save_labels = np.hstack((save_labels, label.detach().cpu().numpy()))
    return save_features, save_labels



if __name__ =="__main__":
    BATCH_SIZE = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = vgg11(weights=VGG11_Weights.DEFAULT).to(device)
    model.classifier[6] = nn.Linear(4096, 3)  # 有3个类别
    model = model.to(device)
    root_dir = "./../../"
    train_dataset = VehicleImageDataset(root=os.path.join(root_dir, "vehicle-aug/img/train"), width=224, height=224)
    test_dataset = VehicleImageDataset(root=os.path.join(root_dir, "vehicle-aug/img/test"), width=224, height=224)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    features, labels = make_feature_dataset(model.features, train_loader, device)
    np.savetxt("train_features.csv", features, delimiter=",")
    np.savetxt("train_labels.csv", labels, delimiter=",")

    features, labels = make_feature_dataset(model.features, test_loader, device)
    np.savetxt("test_features.csv", features, delimiter=",")
    np.savetxt("test_labels.csv", labels, delimiter=",")