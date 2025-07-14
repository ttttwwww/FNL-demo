import torch

from . import logger
from .base import BaseLoss
from snn_simulator.utils.loss_utils import s4nn_mes_loss


class S4NNLoss(BaseLoss):
    def __init__(self, max_spike_time: int, delay_timestep: int = 3):
        """
        Loss for S4NN.
        Args:
            max_spike_time: Maximum spike time
            delay_timestep: Delay timestep for incorrect channels
        """
        super().__init__()
        self.max_spike_time = max_spike_time
        self.delay_timestep = delay_timestep

    def forward(self, output:torch.Tensor, label:torch.Tensor) -> torch.Tensor:
        """
        Forward propagation for S4NNLoss.
        Args:
            output: Network output
            label: Dataset label

        Returns:
            loss: S4NN loss
        """
        loss = s4nn_mes_loss(output, label, self.max_spike_time, self.delay_timestep)
        return loss

class TTFSCrossEntropyLoss(BaseLoss):
    def __init__(self):
        """
        Cross entropy loss for TTFS network.
        For TTFS network, the smallest spike time in each channel is considered as the prediction.
        So the logits are negated to match the cross entropy loss calculation.
        """
        super().__init__()

    def forward(self, output:torch.Tensor, label:torch.Tensor) -> torch.Tensor:
        """
        Forward propagation for TTFSCrossEntropyLoss.
        Args:
            output: Network output
            label: Dataset label

        Returns:
            loss: Cross entropy loss
        """
        logits = -output
        loss = torch.nn.functional.cross_entropy(logits, label)
        return -loss