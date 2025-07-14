from typing import Callable

import torch

from . import logger
from .base import BaseEncoder
from snn_simulator.utils.encode import ttfs_encode

def rgb_to_gray(image:torch.Tensor):
    gray = 0.299*image[0]+0.587*image[1]+0.114*image[2]
    return gray


class S4NNImageEncoder(BaseEncoder):
    """
    Encoder for image data in S4NN way.
    """
    def __init__(self,max_spike_time:int):
        super().__init__()
        self.max_time_step = max_spike_time
        self.encode_fun = ttfs_encode

    def encode(self, image:torch.Tensor):
        """
        Encode image into spike train in S4NN way.
        Args:
            image:shape [B,C, H, W] and if not gray the color order should be Red,Green,Blue.
        Returns:
            spike_train: shape [B,num_timestep,C, H, W]
        """
        if image.ndim == 4:
            if not (image.shape[1] == 3 or image.shape[1] == 1):
                logger.error(f"Size of img is {image.shape} are not compatible with S4NN encoder."
                             f"Channel should be 3 or 1.")
                raise ValueError("Raw dataset are not compatible with S4NN encoder.")
        elif image.ndim == 2:
            pass
        else:
            logger.error(f"Size of img is {image.shape} are not compatible with S4NN encoder."
                         f"Image should be 4D tensor [B,C,H,W].")
            raise ValueError("Raw dataset are not compatible with S4NN encoder.")
        # if image.shape[1]== 3:
        #     image = rgb_to_gray(image)
        out = self.encode_fun(image,self.max_time_step)
        return out

class S4NN1DFeatureEncoder(BaseEncoder):
    """
    Encoder for 1D feature in S4NN way.
    """
    def __init__(self,max_spike_time:int):
        super().__init__()
        self.max_time_step = max_spike_time
        self.encode_fun = ttfs_encode

    def encode(self, image:torch.Tensor):
        """
        Encode image into spike train in S4NN way.
        Args:
            image:shape [B,feature_in]
        Returns:
            spike_train: shape [B,feature_out]
        """
        if not image.ndim == 2:
            logger.error(f"Size of img should be [B,feature_in] however is {image.shape}.")
            raise ValueError(f"Raw dataset are not compatible with {self.__class__.__name__}.")
        out = self.encode_fun(image,self.max_time_step)
        return out

class ExponentialTTFSEncoder(BaseEncoder):
    """
    Encode input image into fire time in exponential way.
    """
    def __init__(self,max_spike_time:int):
        super().__init__()
        self.max_spike_time = max_spike_time

    def encode(self,image:torch.Tensor):
        """
        Args:
            image: shape [B,C,H,W] and if not gray the color order should be Red,Green,Blue.
        Returns:
            spike_train: shape [B,num_timestep,C,H,W]
        """
        if not image.ndim == 4:
            logger.error(f"Size of img is {image.shape} are not compatible with ExponentialTimeEncoder."
                         f"Image should be 4D tensor [B,C,H,W].")
            raise ValueError("Raw dataset are not compatible with ExponentialTimeEncoder.")
        if not (image.shape[1] == 3 or image.shape[1] == 1):
            logger.error(f"Size of img is {image.shape} are not compatible with ExponentialTimeEncoder."
                         f"Channel should be 3 or 1.")
            raise ValueError("Raw dataset are not compatible with ExponentialTimeEncoder.")
        if image.shape[1] == 3:
            image = rgb_to_gray(image)
        assert image.max() <= 1 and image.min() >= 0, "Image values should be in [0,1]"
        spike_time = (1.-image) * self.max_spike_time
        ex_spike_time = torch.exp(spike_time)
        return ex_spike_time