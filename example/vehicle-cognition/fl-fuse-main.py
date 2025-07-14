import math
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import random_split
import torchvision.transforms as transforms
from snn_simulator.utils.logger import setup_logger

logger = setup_logger("snn_simulator", f"./log")
from dataset import get_default_multimodal_dataset, get_uneven_fl_augment_dataset, get_uneven_fl_dataset
from snn_simulator.dataset.single_dataset import ImageDataset, MultimodalDataset, AugmentImageDataset
from snn_simulator.config.config_manager import ConfigManager
from snn_simulator.core import simulator


def split_dataset(dataset, sizes, seed=42):
    """
    Split dataset into multiple parts based on the given sizes.

    Args:
        dataset (Dataset): The dataset to split.
        sizes (list): List of sizes for each split.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list: List of datasets corresponding to each split.
    """
    assert math.isclose(sum(sizes), 1)
    lengths = [int(len(dataset) * size) for size in sizes]
    indices = np.random.RandomState(seed).permutation(len(dataset))
    splits = np.split(indices, np.cumsum(lengths)[:-1])
    return [Subset(dataset, split) for split in splits]


def single_run(epochs, dataset_portion, save_root="./log"):
    dataset1, dataset2, dataset3, test_dataset = get_uneven_fl_augment_dataset(dataset_portion)
    # train_dataset = AugmentImageDataset(raw_dataset,train_indices,transform=train_transform)
    dataset1 = MultimodalDataset(dataset1, **config["dataset"], debug=False)
    dataset2 = MultimodalDataset(dataset2, **config["dataset"], debug=False)
    dataset3 = MultimodalDataset(dataset3, **config["dataset"], debug=False)
    test_dataset = MultimodalDataset(test_dataset, **config["dataset"])
    train_dataset_list = [dataset1, dataset2, dataset3]

    trainer = simulator.FLTimeStepTrainer(epochs, config, train_dataset_list, test_dataset, debug=False,
                                          save_root=save_root)
    trainer.run_fl(merge=False)


def single_run_without_augment(epochs, dataset_portion, save_root="./log"):
    dataset1, dataset2, dataset3, test_dataset = get_uneven_fl_dataset(dataset_portion)
    # train_dataset = AugmentImageDataset(raw_dataset,train_indices,transform=train_transform)
    dataset1 = MultimodalDataset(dataset1, **config["dataset"], debug=False)
    dataset2 = MultimodalDataset(dataset2, **config["dataset"], debug=False)
    dataset3 = MultimodalDataset(dataset3, **config["dataset"], debug=False)
    test_dataset = MultimodalDataset(test_dataset, **config["dataset"])
    train_dataset_list = [dataset1, dataset2, dataset3]

    trainer = simulator.FLTimeStepTrainer(epochs, config, train_dataset_list, test_dataset, debug=False,
                                          save_root=save_root)
    trainer.run_fl(merge=True)


if __name__ == "__main__":
    config_path = "./example/vehicle-cognition/multimodal-mlp.yaml"
    # config_path = "./example/vehicle-cognition/multimodal-cnn.yaml"
    cfm = ConfigManager(config_path)
    config = cfm.load_config()
    # dataset_portion = [
    #     [0.2, 0.7, 0.1],
    #     [0.7, 0.1, 0.2],
    #     [0.1, 0.2, 0.7],
    # ]
    dataset_portion = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ]
    # dataset_portion = [
    #     [0.9, 0.05, 0.05],
    #     [0.05, 0.9, 0.05],
    #     [0.05, 0.05, 0.9],
    # ]
    # dataset_portion = [
    #     [0.3, 0.4, 0.3],
    #     [0.4, 0.3, 0.3],
    #     [0.3, 0.3, 0.4],
    # ]
    # dataset_portion = [
    #     [0.3, 0.2, 0.2],
    #     [0.2, 0.3, 0.2],
    #     [0.2, 0.2, 0.3],
    #     [0.3, 0.3, 0.3]
    # ]
    # dataset_portion = [
    #     [0.5, 0.1, 0.1],
    #     [0.1, 0.5, 0.1],
    #     [0.1, 0.1, 0.5],
    #     [0.3, 0.3, 0.3]
    # ]

    TIMES = 20
    EPOCHS = 20
    # save_root = "./log/0.1-0.8-0.1--0.8-0.1-0.1--0.1-0.1-0.8"
    save_root = f"./log/{dataset_portion[0][0]}-{dataset_portion[0][1]}-{dataset_portion[0][2]}--{dataset_portion[1][0]}-{dataset_portion[1][1]}-{dataset_portion[1][2]}--{dataset_portion[2][0]}-{dataset_portion[2][1]}-{dataset_portion[2][2]}"
    # save_root = f"./log/0.3-0.4-0.3--0.4-0.3-0.3--0.3-0.3-0.4"
    for i in range(TIMES):
        logger.info(f"Run {i + 1} / {TIMES}")
        single_run(EPOCHS, dataset_portion, save_root)
        # single_run_without_augment(EPOCHS,dataset_portion, save_root)
