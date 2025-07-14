import random

import math
import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split,Subset
import librosa
from collections import defaultdict


class VehicleDatasetBase(Dataset):
    def __init__(self, root, data_types: [str]):
        self.root = root
        self.data_types = data_types
        self.data_path = []
        self.labels = []
        self.classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        self.classes_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self._load_filepath()

    def _load_filepath(self):
        for cls in self.classes_to_idx:
            cls_path = os.path.join(self.root, cls)
            for img_path in os.listdir(cls_path):
                tail = os.path.splitext(img_path)[1]
                if tail in self.data_types:
                    self.data_path.append(os.path.join(cls_path, img_path))
                    self.labels.append(self.classes_to_idx[cls])
        print(f"load data from {self.root} total :{len(self.data_path)}")

    def __len__(self):
        return len(self.data_path)


class VehicleImageDataset(VehicleDatasetBase):

    def __init__(self, root, width, height, data_type=None):
        if data_type is None:
            data_type = [".jpg"]
        super(VehicleImageDataset, self).__init__(root, data_type)
        self.width = width
        self.height = height
        print("image dataset loaded")

    def __getitem__(self, idx):
        img_path = self.data_path[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.width, self.height))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        img = np.transpose(img, (2, 0, 1))  # Change to [C, H, W]
        # img = torch.tensor(img,dtype=torch.float32).unsqueeze(0)
        return img, label


class VehicleAudioDataset(VehicleDatasetBase):
    SAMPLE_RATE = 22050
    DURATION = 2  # 音频截取事件长度，单位s
    """trans audio data into featuremap [frame_number,channel_number]"""

    def __init__(self, root, data_types=None):
        if data_types is None:
            data_types = [".wav"]
        super(VehicleAudioDataset, self).__init__(root, data_types)
        print("audio dataset loaded")
        [self.frame_number, self.channel_number] = self.get_featuremap_shape()

    def __getitem__(self, idx):
        audio_path = self.data_path[idx]
        label = self.labels[idx]
        audio_data, sample_rate = librosa.load(audio_path)
        # norm
        feature_map = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=80)
        # feature_map = self._norm_mfcc(feature_map)
        feature_map = np.mean(feature_map, axis=1)
        # feature_map_norm = (feature_map - np.mean(feature_map)) / np.std(feature_map)
        feature_map_minmax = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        # feature_map = torch.tensor(feature_map_minmax, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Add batch dimension
        feature_map = np.expand_dims(np.expand_dims(feature_map_minmax, 0), 0)
        return feature_map, label

    def get_featuremap_shape(self):
        audio_path = self.data_path[0]
        audio_data, sample_rate = librosa.load(audio_path, duration=1)
        feature_map = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
        return feature_map.shape

    @staticmethod
    def _norm_mfcc(feature_map):
        min_val = np.min(feature_map, axis=1, keepdims=True)
        max_val = np.max(feature_map, axis=1, keepdims=True)
        return (feature_map - min_val) / (max_val - min_val + 1e-6)


class VehicleFuseDataset:
    def __init__(self, img_dataset: Dataset, audio_dataset: Dataset):
        self.img_dataset = img_dataset
        self.audio_dataset = audio_dataset
        print("fuse dataset loaded")

    def __getitem__(self, idx):
        img, img_label = self.img_dataset[idx]
        audio, audio_label = self.audio_dataset[idx]
        assert img_label == audio_label
        return [img, audio], img_label

    def __len__(self):
        return len(self.audio_dataset)

class VehicleFuseAugmentedDataset:
    """
    Due to the augment all the train data is augmented 10 times
    """
    def __init__(self, img_dataset: Dataset, audio_dataset: Dataset, augment_times=10):
        self.img_dataset = img_dataset
        self.audio_dataset = audio_dataset
        self.augment_times = augment_times
        print("fuse augmented dataset loaded")

    def __getitem__(self, idx):
        img, img_label = self.img_dataset[idx]
        audio, audio_label = self.audio_dataset[int(idx/self.augment_times)]
        assert img_label == audio_label
        return [img, audio], img_label

    def __len__(self):
        return len(self.img_dataset)

class VehicleImgFeatureDataset(Dataset):
    def __init__(self,feature_path,label_path):
        self.features = np.loadtxt(feature_path, delimiter=",")
        self.labels = np.loadtxt(label_path, delimiter=",", dtype=np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature_norm = (feature - feature.min()) / (feature.max() - feature.min() + 1e-6)
        label = self.labels[idx]
        return feature_norm, label


def get_default_img_dataset():
    current_dir = os.path.dirname(__file__)
    dataset = VehicleImageDataset(root=os.path.join(current_dir, "vehicle/image"), width=28, height=28)
    return dataset


def get_default_img_feature_dataset():
    current_dir = os.path.dirname(__file__)
    feature_path = os.path.join(current_dir, "vehicle/img-vgg-feature/features.csv")
    label_path = os.path.join(current_dir, "vehicle/img-vgg-feature/labels.csv")
    dataset = VehicleImgFeatureDataset(feature_path, label_path)
    return dataset


def get_default_audio_dataset():
    current_dir = os.path.dirname(__file__)
    dataset = VehicleAudioDataset(root=os.path.join(current_dir, "vehicle/sound"))
    return dataset


def get_default_multimodal_dataset():
    current_dir = os.path.dirname(__file__)
    img_feature_root = os.path.join(current_dir, "vehicle/img-vgg-feature")
    audio_root = os.path.join(current_dir, "vehicle/sound")
    feature_path = os.path.join(img_feature_root, "features.csv")
    label_path = os.path.join(audio_root, "labels.csv")
    img_dataset = VehicleImgFeatureDataset(feature_path,label_path)
    audio_dataset = VehicleAudioDataset(audio_root)
    fuse_dataset = VehicleFuseDataset(img_dataset, audio_dataset)
    return fuse_dataset

def get_augmented_img_feature_dataset():
    current_dir = os.path.dirname(__file__)
    img_feature_root = os.path.join(current_dir, "vehicle/img-vgg-feature")
    train_feature_path = os.path.join(img_feature_root, "train_features.csv")
    train_label_path = os.path.join(img_feature_root, "train_labels.csv")
    test_feature_path = os.path.join(img_feature_root, "test_features.csv")
    test_label_path = os.path.join(img_feature_root, "test_labels.csv")
    train_dataset = VehicleImgFeatureDataset(train_feature_path, train_label_path)
    test_dataset = VehicleImgFeatureDataset(test_feature_path, test_label_path)
    return train_dataset, test_dataset

def get_augmented_multimodal_dataset():
    train_feature_dataset,test_feature_dataset = get_augmented_img_feature_dataset()
    current_dir = os.path.dirname(__file__)
    train_audio_root = os.path.join(current_dir, "vehicle-aug/sound/train")
    test_audio_root = os.path.join(current_dir, "vehicle-aug/sound/test")
    train_audio_dataset = VehicleAudioDataset(train_audio_root)
    test_audio_dataset = VehicleAudioDataset(test_audio_root)
    train_fuse_dataset = VehicleFuseAugmentedDataset(train_feature_dataset, train_audio_dataset,augment_times=10)
    test_fuse_dataset = VehicleFuseDataset(test_feature_dataset, test_audio_dataset)
    return train_fuse_dataset, test_fuse_dataset

def get_uneven_fl_dataset(label_size, seed=42):
    """
    Get a dataset for federated learning with uneven distribution.
    The dataset is split into multiple parts based on the given sizes.
    And the labels are unevenly distributed across the dataset.
    Args:
        label_size: np.ndarray shape is [num_clients, num_label]
        seed:

    Returns:

    """
    assert np.allclose(np.sum(label_size, axis=0), 1.)
    num_clients = len(label_size)
    current_dir = os.path.dirname(__file__)
    img_feature_root = os.path.join(current_dir, "vehicle/img-vgg-feature")
    audio_root = os.path.join(current_dir, "vehicle/sound")
    feature_path = os.path.join(img_feature_root, "features.csv")
    label_path = os.path.join(img_feature_root, "labels.csv")
    img_dataset = VehicleImgFeatureDataset(feature_path,label_path)
    audio_dataset = VehicleAudioDataset(audio_root)
    fuse_dataset = VehicleFuseDataset(img_dataset, audio_dataset)
    label_to_indices = defaultdict(list)
    sample_per_label = defaultdict(int)
    for idx, (_, label) in enumerate(fuse_dataset):
        label_to_indices[label].append(idx)
        sample_per_label[label] += 1
    r = random.Random(seed)
    for v in label_to_indices.values():
        r.shuffle(v)
    client_label_to_indices = defaultdict(list)
    for i in range(num_clients):
        for label, indices in label_to_indices.items():
            num_samples = int(label_size[i][label] * sample_per_label[label])
            client_label_to_indices[i].extend(indices[:num_samples])
            label_to_indices[label] = indices[num_samples:]
    subset_list = []
    for val in client_label_to_indices:
        subset_list.append(Subset(fuse_dataset, client_label_to_indices[val]))
    return subset_list


def get_uneven_fl_augment_dataset(label_size, seed=42):
    """
    Get a dataset for federated learning with uneven distribution.
    The dataset is split into multiple parts based on the given sizes.
    And the labels are unevenly distributed across the dataset.
    Args:
        label_size: np.ndarray shape is [num_clients, num_label]
        seed:

    Returns:

    """
    assert np.allclose(np.sum(label_size, axis=0), 1.)
    num_clients = len(label_size)
    current_dir = os.path.dirname(__file__)
    train_feature_dataset,test_feature_dataset = get_augmented_img_feature_dataset()
    train_audio_root = os.path.join(current_dir, "vehicle-aug/sound/train")
    test_audio_root = os.path.join(current_dir, "vehicle-aug/sound/test")
    train_audio_dataset = VehicleAudioDataset(train_audio_root)
    test_audio_dataset = VehicleAudioDataset(test_audio_root)
    train_fuse_dataset = VehicleFuseAugmentedDataset(train_feature_dataset, train_audio_dataset)
    test_fuse_dataset = VehicleFuseDataset(test_feature_dataset, test_audio_dataset)
    label_to_indices = defaultdict(list)
    sample_per_label = defaultdict(int)
    for idx, (_, label) in enumerate(train_fuse_dataset):
        label_to_indices[label].append(idx)
        sample_per_label[label] += 1
    r = random.Random(seed)
    for v in label_to_indices.values():
        r.shuffle(v)
    client_label_to_indices = defaultdict(list)
    for i in range(num_clients):
        for label, indices in label_to_indices.items():
            num_samples = int(label_size[i][label] * sample_per_label[label])
            client_label_to_indices[i].extend(indices[:num_samples])
            label_to_indices[label] = indices[num_samples:]
    subset_list = []
    for val in client_label_to_indices:
        subset_list.append(Subset(train_fuse_dataset, client_label_to_indices[val]))
    subset_list.append(test_fuse_dataset)
    return subset_list

if __name__ == '__main__':
    # dataset = VehicleImageDataset(root='./vehicle/image')
    dataset = get_default_audio_dataset()
    number = len(dataset)
    id = np.random.randint(number)
    img, label = dataset[id]
    print(img.shape)
    plt.imshow(img)
    plt.show()
