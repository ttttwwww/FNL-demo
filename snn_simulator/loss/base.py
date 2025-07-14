import torch.nn as nn
from . import logger

class BaseLoss(nn.Module):
    """
    Base loss class.Same to loss class in pytorch.
    """
    def __init__(self):
        super().__init__()
        logger.info(f'Loss {self.__class__.__name__} is initialized.')

