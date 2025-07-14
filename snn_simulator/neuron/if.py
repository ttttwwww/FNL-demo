import torch

from snn_simulator.utils.config_utils import get_object_from_str
from .base import BaseNeuron

class IdealIF(BaseNeuron):
    def __init__(self,threshold:float,volt_reset:float,surrogate_cls,debug:bool=False):
        """
        Ideal IF neuron, implements ideal charge fire function.
        Besides, surrogate function is used for gradient backward propagation.
        """
        super().__init__(debug)
        self.v = None
        self.threshold = threshold
        self.volt_reset = volt_reset
        surrogate_cls = get_object_from_str(surrogate_cls)
        self.surrogate_function = surrogate_cls

    def forward(self, x: torch.Tensor):
        if self.v is None:
            self.v = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        self.charge(x)
        spike = self.fire()
        self.reset()
        return spike

    def charge(self, x: torch.Tensor):
        self.v = self.v + x

    def fire(self):
        spike = self.surrogate_function(self.v-self.threshold)
        return spike

    def reset(self):
        self.v = torch.where(self.v > self.threshold, torch.full_like(self.v, self.volt_reset), self.v)

    def batch_reset(self):
        self.v = None