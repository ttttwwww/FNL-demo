
import torch.nn as nn

class BaseLayerSurrogate(nn.Module):
    """
    Define the base layer surrogate.Layer surrogate will cut the computation map for special backward computation.
    Such as temporal coding network.The base layer surrogate is created to make sure weight is included in surrogate
    """
    def __init__(self):
        super().__init__()

    def forward(self,x,weight):
        raise NotImplementedError