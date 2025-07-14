from torch.utils.data import random_split

from snn_simulator.utils.logger import setup_logger
# examples/example1/main.py

# import os
# import sys
#
# sys.path.append(os.path.dirname(__file__))  # 添加当前路径到 sys.path

logger = setup_logger("snn_simulator", f"./log")
from dataset import get_default_img_dataset,get_default_audio_dataset
from snn_simulator.dataset.single_dataset import ImageDataset
from snn_simulator.config.config_manager import ConfigManager
from snn_simulator.core import simulator


# load_config

config_path = "./example/vehicle-cognition/sound-config.yaml"

cfm = ConfigManager(config_path)
config = cfm.load_config()
train_size = config["dataloader"]["train_size"]
test_size = config["dataloader"]["test_size"]
# prepare dataset
raw_dataset = get_default_audio_dataset()
train_dataset,test_dataset = random_split( raw_dataset,[train_size,test_size])
train_dataset = ImageDataset(train_dataset, **config["dataset"], debug=False)
test_dataset = ImageDataset(test_dataset, **config["dataset"],debug=False)
simulator = simulator.TimeStepBasedSimulator(config, train_dataset,test_dataset, debug=False)
simulator.run()
