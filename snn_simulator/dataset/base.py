
from . import logger
from abc import ABC, abstractmethod
from typing import cast, Sized

import torch
from torch.utils.data import Dataset


class BaseEncoder(ABC):
    """
    Encoder base class.
    Input:Tensor Data in specific format.For example image shape is [C,H,W].
    Output:Spike train Data with additional time dimension.[num_timestep,C,H,W]
    """

    def __init__(self, *args, **kwargs):
        """
        Args are defined in subclass
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Should be implemented in subclass.
        Args:
            x: input Tensor.
        Returns:
            x: output Tensor.
        """
        raise NotImplementedError


class BaseSingleDataset(Dataset, ABC):
    """
    Preprocess input dataset.By decorating __getitem__ makes it compatible for SNN.
    To leverage multi-threading in dataloader for acceleration, lazy processing is adopted.
    Actual processing is deferred until data is read
    """

    def __init__(self, raw_dataset: Dataset, debug: bool = False):
        """
        Special parameters should be implemented in subclass.
        Args:
            raw_dataset: Raw dataset.
            debug: Debug mode.
        """
        logger.info(f"Creating {self.__class__.__name__}")
        logger.info(f"Raw_dataset is : {raw_dataset.__class__.__name__}")
        logger.info(f"Dataset has {len(cast(Sized, raw_dataset))} images")
        self.raw_dataset = raw_dataset
        self.debug = debug

    def __len__(self):
        return len(cast(Sized, self.raw_dataset))

    def __getitem__(self, idx):
        """
        Should be implemented in subclass.
        """
        raise NotImplementedError
