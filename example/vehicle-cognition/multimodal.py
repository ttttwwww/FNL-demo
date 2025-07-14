from torch.utils.data import random_split
import torchvision.transforms as transforms
from snn_simulator.utils.logger import setup_logger
# examples/example1/main.py

# import os
# import sys
#
# sys.path.append(os.path.dirname(__file__))  # 添加当前路径到 sys.path

logger = setup_logger("snn_simulator", f"./log")
from dataset import get_default_multimodal_dataset,get_augmented_multimodal_dataset
from snn_simulator.dataset.single_dataset import ImageDataset, MultimodalDataset, AugmentImageDataset
from snn_simulator.config.config_manager import ConfigManager
from snn_simulator.core import simulator


# load_config
config_path = "./example/vehicle-cognition/multimodal-mlp.yaml"
# config_path = "./example/vehicle-cognition/multimodal-cnn.yaml"
cfm = ConfigManager(config_path)
config = cfm.load_config()

# prepare dataset
# train_size = config["dataloader"]["train_size"]
# test_size = config["dataloader"]["test_size"]
#
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # 假设输入128x128
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
# ])
#
# raw_dataset =get_default_multimodal_dataset()
# train_dataset,test_dataset = random_split(raw_dataset, [train_size, test_size])
# train_indices = train_dataset.indices
# test_indices = test_dataset.indices

# train_dataset = AugmentImageDataset(raw_dataset,train_indices,transform=train_transform)
train_dataset,test_dataset = get_augmented_multimodal_dataset()
train_dataset = MultimodalDataset(train_dataset, **config["dataset"], debug=False)
test_dataset = MultimodalDataset(test_dataset, **config["dataset"])


simulator = simulator.TimeStepBasedSimulator(config, train_dataset,test_dataset, debug=False)
simulator.run()
