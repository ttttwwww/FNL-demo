from torch.utils.data import random_split

from snn_simulator.utils.logger import setup_logger
# examples/example1/main.py

# import os
# import sys
#
# sys.path.append(os.path.dirname(__file__))  # 添加当前路径到 sys.path

import torchvision
logger = setup_logger("snn_simulator", f"./log")
from dataset import get_augmented_img_feature_dataset
from snn_simulator.dataset.single_dataset import ImageDataset, AugmentImageDataset
from snn_simulator.config.config_manager import ConfigManager
from snn_simulator.core import simulator
import torchvision.transforms as transforms


# load_config

config_path = "./example/vehicle-cognition/img-mlp.yaml"

cfm = ConfigManager(config_path)
config = cfm.load_config()

train_size = config["dataloader"]["train_size"]
test_size = config["dataloader"]["test_size"]
# prepare dataset



train_dataset,test_dataset =get_augmented_img_feature_dataset()

# train_dataset = AugmentImageDataset(raw_dataset,train_indices,transform=train_transform)
train_dataset = ImageDataset(train_dataset, **config["dataset"], debug=False)
test_dataset = ImageDataset(test_dataset, **config["dataset"])


simulator = simulator.TimeStepBasedSimulator(config, train_dataset,test_dataset, debug=False)
simulator.run()
