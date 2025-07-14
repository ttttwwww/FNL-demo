import torch
import torch.nn as nn
from snn_simulator.utils.config_utils import get_object_from_str
from .base import BaseNeuron
from typing import Union
from . import logger

class BaseLIF(BaseNeuron):
    """
    Ideal LIF neuron, implements ideal charge fire function.
    Besides, surrogate function is used for gradient backward propagation.
    """

    def __init__(self,volt_threshold: Union[float, torch.Tensor],
                 volt_reset: Union[float, torch.Tensor], alpha: Union[float, torch.Tensor],
                 surrogate_cls: str,debug:bool=False):
        """
        Ideal LIF parameters. Surrogate function used for gradient backward propagation.
        The parameters could be "int" type so that setting of every neuron is same.
        Or they could be "torch.Tensor" type so that setting of every neuron differs.
        Parameter Example:
        [
            num_neurons = 10,
            volt_threshold = 0.1,
            volt_reset = 0,
            alpha = 0.9,
        ]
        Args:
            volt_threshold: Threshold voltage of Ideal LIF
            volt_reset: Reset voltage of Ideal LIF
            alpha: Decay factor of Ideal LIF
            surrogate_cls: Surrogate function for backward propagation.Heaviside function for layers that takes the backward propagation
            debug: Debug mode
        """

        super().__init__(debug)
        self.volt_threshold = volt_threshold
        self.volt_reset = volt_reset
        self.alpha = alpha
        surrogate_cls:nn.Module = get_object_from_str(surrogate_cls)
        self.surrogate_function = surrogate_cls()

        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function in nn.Module
        Args:
            x: Input tensor shape is [batch_size, num_neurons]

        Returns:
            spike: Binary tensor, shape is [batch_size, num_neurons].
        """
        # init self.v
        if self.v is None:
            self.v = torch.zeros_like(x).to(x.device, dtype=x.dtype)
        self.charge(x)
        spike = self.fire()
        self.reset()
        return spike

    def charge(self, x: torch.Tensor):
        """
        Charge in neuron simulation.Including integrate and leak.

        Args:
            x: Input tensor shape is [batch_size, num_neurons]
        """
        #TODO Recover this
        # self.v = (1-self.alpha) * x + self.alpha * self.v
        self.v =  x +  self.alpha * self.v

    def fire(self) -> torch.Tensor:
        """
        Fire in neuron simulation. Check if neuron generates spike.
        Returns:
            spike: Binary tensor, shape is [batch_size, num_neurons]. The spikes generated in this time step.
        """
        spike = self.surrogate_function(self.v - self.volt_threshold)
        return spike

    def reset(self):
        """
        Reset in neuron simulation. Reset voltage of neuron into v_reset for neuron has spiked

        """
        self.v = torch.where(self.v>self.volt_threshold, torch.full_like(self.v,self.volt_reset), self.v)

    def batch_reset(self):
        """
        Reset neuron, so that different batch begins with same neuron state.
        """
        self.v = None



class RCLIF(BaseLIF):
    def __init__(self, volt_threshold: Union[float, torch.Tensor],
                 volt_reset: Union[float, torch.Tensor], tau: Union[float, torch.Tensor],
                 R1: Union[float, torch.Tensor], Rh: Union[float, torch.Tensor], Rl: Union[float, torch.Tensor],
                 C: [Union[float, torch.Tensor]], surrogate_function:str,debug:bool=False):
        """
        LIF neuron, including circuit parameters.Alpha is computed by RC circuit.
        Args:
            volt_threshold: See BaseLIF.
            volt_reset: See BaseLIF.
            tau: Duration of 1 step.
            R1: Resistance of load Resistor.
            Rh: Resistance of memristor in high resistant state of load Resistor.
            Rl: Resistance of memristor in low resistant state of load Resistor.
            C: Capacitance of C paralleled memristor.
            debug: For debug mode.
            surrogate_function: See BaseLIF.
        """
        self.tau = tau
        self.R1 = R1
        self.Rh = Rh
        self.Rl = Rl
        self.C = C
        alpha = torch.exp(torch.tensor(-self.tau / ((self.R1 + self.Rh) * self.C)))
        super().__init__(volt_threshold, volt_reset, alpha, surrogate_function,debug)
