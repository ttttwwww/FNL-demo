from . import logger
from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class BaseNeuron(ABC, nn.Module):
    """
    Neuron base class, describing the neuron behavior including fire,charge and reset
    One neuron class contains neurons in single layer, which means the class only has one dimension.
    The shape of input and output tensor is [batch_size, num_neurons]
    """

    def __init__(self, debug: bool = False):
        """
        Concrete property is supposed to be implemented by subclasses.

        Args:
            debug: Debug mode.
        """
        super().__init__()
        self.debug = debug
        self.v = None  # Neuron's membrane voltage.Shape is [batch_size,num_neurons] change dynamically with batch_size
        if self.debug:
            logger.debug(f"neuron {self.__class__.__name__} debug mode is on")

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """
        Torch forward function.Should be implemented by subclasses.
        Including:charge,fire,reset

        Args:
            x: Input Tensor shape is [batch_size, num_neurons]

        Returns:
            Output Tensor shape is [batch_size, num_neurons]
        """
        pass

    @abstractmethod
    def charge(self, x: torch.Tensor):
        """
        Charge operation in neuron simulation. Integrates the input tensor and change the neuron's voltage.
        Specific function should be implemented by subclasses.

        Args:
            x: Input Tensor shape is [batch_size, num_neurons]
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset operation in neuron simulation. Reset the neuron's voltage if spike is generated.
        Specific function should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def batch_reset(self):
        """
        Reset the parameters like voltage after every batch.
        To make sure every simulation goes with same initial stated.
        """
        pass

    def get_v(self)-> torch.Tensor:
        """
        Get the membrane voltage of neuron.
        """
        return self.v
