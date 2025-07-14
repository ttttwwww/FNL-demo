import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Type, cast, Sized
from .base import BaseSingleDataset, BaseEncoder
from snn_simulator.utils.config_utils import get_object_from_str
from . import logger
from PIL import Image

class AugmentImageDataset(Dataset):
    """
    Dataset class for image data with augmentation.
    Input raw dataset and transform function.Then return augmented image and label.
    """
    def __init__(self,raw_dataset,indices,transform=None):
        self.raw_dataset = raw_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, label = self.raw_dataset[self.indices[index]]
        img = (img*255).astype(np.uint8)
        img = np.transpose(img,(1,2,0))
        img = Image.fromarray(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class ImageDataset(BaseSingleDataset):
    """
    Single modality image dataset class.
    For raw dataset. The image shape should be [batch_size, height, width] or [batch_size, height, width, channel].
    Besides the input data should be [torch.Tensor].
    The label should be [batch_size].
    Output data shape is [time_step,H*W].
    """
    def __init__(self, raw_dataset: Dataset, encoder_cls: str, encoder_params: dict,debug:bool=False):
        """
        Load customary dataset and encode params.
        Args:
            raw_dataset:Raw dataset.
            encoder_cls: Text of encoder class.
            encoder_params: Parameters of encoder.
            debug: Debug mode.
        """
        super().__init__(raw_dataset,debug)
        encoder_cls = get_object_from_str(encoder_cls)
        try:
            self.encoder = encoder_cls(**encoder_params)
        except Exception as e:
            logger.error(f"{e}")
            logger.error(f"Encoder params : {encoder_params} are not compatible for {encoder_cls}")
            if self.debug:
                raise e

    def __getitem__(self, index):
        img, label = self.raw_dataset[index]
        if isinstance(img, np.ndarray):
            img = torch.tensor(img).float()
        if img.ndim == 3 or img.ndim == 1:
            img = img.unsqueeze(0) #[C,H,W] -> [B,C,H,W] or [feature_in] -> [B,feature_in]
        encoded_img = self.encoder.encode(img).squeeze(0)
        # return [encoded_img, label]
        return encoded_img,label

class MultimodalDataset(BaseSingleDataset):
    """
    Multi-modality dataset class.
    For raw dataset. The image shape should be [batch_size, height, width] or [batch_size, height, width, channel].
    Besides the input data should be [torch.Tensor].
    The label should be [batch_size].
    Output data shape is [time_step,H*W].
    """
    def __init__(self, raw_dataset: Dataset, encoder_cls: str, encoder_params: dict,debug:bool=False):
        """
        Load customary dataset and encode params.
        Args:
            raw_dataset:Raw dataset.
            encoder_cls: Text of encoder class.
            encoder_params: Parameters of encoder.
            debug: Debug mode.
        """
        super().__init__(raw_dataset,debug)
        encoder_cls = get_object_from_str(encoder_cls)
        try:
            self.encoder = encoder_cls(**encoder_params)
        except Exception as e:
            logger.error(f"{e}")
            logger.error(f"Encoder params : {encoder_params} are not compatible for {encoder_cls}")
            if self.debug:
                raise e

    def __getitem__(self, index):
        raw_multimodal_data, label = self.raw_dataset[index]
        img,audio = raw_multimodal_data
        img = torch.tensor(img).float()
        audio = torch.tensor(audio).float()
        if img.ndim == 3 or img.ndim == 1:
            img = img.unsqueeze(0) #[C,H,W] -> [B,C,H,W], # or [feature_in] -> [B,feature_in]
        encoded_img = self.encoder.encode(img).squeeze(0)
        T = encoded_img.shape[0]
        img_flat = encoded_img.view(T,-1)
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)#[C,H,W] -> [B,C,H,W]
        encoded_audio = self.encoder.encode(audio).squeeze(0)
        T,C,H,W = encoded_audio.shape
        audio_flat = encoded_audio.view(T,-1)
        encoded_data = torch.cat((img_flat, audio_flat), dim=1)
        # return [encoded_img, label]
        return [encoded_img,encoded_audio],label