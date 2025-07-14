"""
Function layer for SNN.Consists of two main parts:
1. Assistant layers for SNN, such as convolution, pooling, and linear layers.The compute method is modified for spike train.
2. Neuron that has closed formulation, the input and output could be converted from spike to special variant
like spike time or spike number
"""
from typing import Union, Callable

import numpy as np
import torch
from torch import nn, Tensor
from snn_simulator.models.base import BaseLayer
from snn_simulator.surrogate.layer_surrogate import S4NNSpikeTrainToSpikeTime
from . import logger
from snn_simulator.utils.spike_convert import spike_train_to_spike_time, spike_time_to_spike_train
import torch.nn.functional as F


def resume_ann_forward(x_seq: torch.Tensor,
                       stateless_module: nn.Module or list or tuple or nn.Sequential or Callable) -> torch.Tensor:
    """
    Function for custom functional layer in SNN.Such as conv layer and pool layer.
    Compared to ANN layer, spike train has extra time dimension, so the SNN layer would modify the input shape
    And use this function for the ANN compute like convolution or pool
    Args:
        x_seq:Input Tensor[B,T,C,H,W]
        stateless_module: Ann Module

    Returns:
        x:Output Tensor
    """
    y_shape = [x_seq.shape[0], x_seq.shape[1]]
    y = x_seq.flatten(0, 1)
    if isinstance(stateless_module, (list, tuple, nn.Sequential)):
        for m in stateless_module:
            y = m(y)
    else:
        y = stateless_module(y)
    y_shape.extend(y.shape[1:])
    return y.view(y_shape)


class SpikeToTimeLayer(BaseLayer):
    """
    A layer for TTFS encoded net. Convert Last spike train to spike time.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Spike to time layer init function.
        Args:
            debug: debug mode
        """
        super().__init__(debug)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return S4NNSpikeTrainToSpikeTime.apply(x)


class Conv2d(nn.Conv2d):
    """
    A 2d convolution layer for TTFS encoded net. Run convolution for spike train.
    This layer will do 2d convolution each time frame in spike train.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = "zeros", debug: bool = False) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of 2d Convolution layer.
        Args:
            x: Input tensor shape is [batch_size, num_timestep, H,W]
        Returns:
            spike: Binary tensor, shape is [batch_size, num_neurons].
        """
        x = resume_ann_forward(x, super().forward)
        return x


class MaxPool2d(nn.MaxPool2d):
    """
    A 2d max pooling layer for TTFS encoded net. Run max pooling for spike train.
    This layer will do 2d max pooling each time frame in spike train.
    """

    def __init__(self, kernel_size: Union[int, tuple], stride: Union[int, tuple] = None,
                 padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1,
                 return_indices: bool = False, ceil_mode: bool = False, debug: bool = False) -> None:
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x: Tensor):
        x = resume_ann_forward(x, super().forward)
        return x


class Flatten(nn.Flatten):
    def __init__(self, debug: bool = False) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = resume_ann_forward(x, super().forward)
        return x


class BatchNorm2d(nn.BatchNorm2d):
    """
    A 2d batch normalization layer for TTFS encoded net. Run batch normalization for spike train.
    This layer will do batch normalization each time frame in spike train.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, debug: bool = False) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = resume_ann_forward(x, super().forward)
        return x


class LayerNorm(nn.Module):
    """
    A custom layer normalization layer for SNN. Run layer normalization for spike train after convolution.
    This layer will do layer normalization each time frame in spike train.
    """

    def __init__(self, num_features, eps: float = 1e-5, elementwise_affine: bool = True,
                 debug: bool = False) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:[T,B,C,H,W]
        Returns:

        """
        x = x.permute(0, 1, 3, 4, 2)  # ->[T,B,H,W,C]
        x = self.ln(x)
        x = x.permute(0, 1, 4, 2, 3)  # ->[T,B,C,H,W]
        return x


def lif_impulse_spike_time_compute(weight, inputs, threshold, max_spike_time):
    # TODO move the compute method from LIFImpulseSpikeTimeLinear to here.
    pass


class _AverTTFSPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_trains: torch.Tensor, max_spike_time: int, kernel_size: int, stride: int, padding: int = 0):
        """
        Convert spike trains to spike time, then do the pool in time domain.Finally revert spike trains by pooled spike time
        Args:
            spike_trains:[B,T,C,H,W]
        Returns:
            out_trains:[B,T,C,(H-kernel_size+1+2*padding)//stride,(W-kernel_size+1+2*padding)//stride]
        Returns:

        """
        # TODO check the dataflaw
        with torch.no_grad():
            B, T, C, H, W = spike_trains.shape
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.padding = padding
            ctx.input_shape = [B, T, C, H, W]
            spike_flatten = spike_trains.view(B, T, -1)  # [B,T,C*H*W]
            spike_times = spike_train_to_spike_time(spike_flatten)[:, :, 0]  # [B,C*H*W]
            spike_times = spike_times.view(B, C, H, W)  # [B,C,H,W]
            # Do the average pool in time domain
            pooled_spike_times = F.avg_pool2d(spike_times, kernel_size, stride,
                                              padding)  # [B,C,(H-kernel_size+1+2*padding)//stride,(W-kernel_size+1+2*padding)//stride]
            _, _, H_out, W_out = pooled_spike_times.shape
            pooled_times_flatten = pooled_spike_times.view(B, -1).unsqueeze(-1)
            output_spike = spike_time_to_spike_train(pooled_times_flatten, max_spike_time)  # [B,T,C_out]
            output_spike = output_spike.view(B, max_spike_time, C, H_out, W_out)  # [B,T,C_out,H_out,W_out]

        return output_spike

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function of AverTTFSPool.Evenly distribute the output grad into corresponding input
        Args:
            grad_output:[T,B,C,new_H,new_W]

        Returns:
            grad_in:[T,B,C,H,W]
        """
        B, T, C, H_out, W_out = grad_output.shape
        grad_output = grad_output[:, 0, :, :, :]  # [B,C_out,H_out,W_out]
        B, T, C, H, W = ctx.input_shape
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        grad_output_unfold = grad_output.view(B, C, -1)  # [B,C_out,L]
        grad_output_unfold = grad_output_unfold.unsqueeze(2).expand(-1, -1, kernel_size * kernel_size, -1)
        grad_output_unfold = grad_output_unfold / (kernel_size * kernel_size)  # [B,C_out,kH*kW,L]
        grad_output_unfold = grad_output_unfold.view(B, C * kernel_size * kernel_size, -1)  # [B,C_out*kH*kW,L]
        grad_in = F.fold(grad_output_unfold, output_size=(H, W), kernel_size=kernel_size, stride=stride,
                         padding=padding)  # [B,C,H,W]
        grad_in = grad_in.unsqueeze(1).expand(-1, T, -1, -1, -1)
        return grad_in, None, None, None, None


class AverTTFSPool2d(nn.Module):
    """
        Average pool for time coding SNN.

    """

    def __init__(self, kernel_size: int, stride: int, max_spike_time: int, padding: int = 0,
                 debug: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_spike_time = max_spike_time

    def forward(self, spike_trains):
        """
        Convert spike trains to spike time, then do the pool in time domain.Finally revert spike trains by pooled spike time
        Args:
            spike_trains:[B,T,C,H,W]
        Returns:
            out_trains:[B,T,C,(H-kernel_size+1+2*padding)//stride,(W-kernel_size+1+2*padding)//stride]
        """
        return _AverTTFSPoolFunction.apply(spike_trains, self.max_spike_time, self.kernel_size, self.stride,
                                           self.padding)


class LIFImpulseSpikeTimeLinear(nn.Linear):
    # TODO This method won't work.Need a new method to compute the LIF fire time
    """
    A linear layer for temporal encoded net. The LIF model and delta impulse is used for forward propagation.
    Dataflow inside the layer would be spike time.
    Spike time layer will take the first spike time of each channel and compute the first spike time as output.
    """

    def __init__(self, in_features: int, out_features: int, ex_max_spike_time: float, tau: float,
                 threshold: float = 1.0,
                 bias: bool = True, debug: bool = False) -> None:
        """
        Spike time linear layer init function.
        Args:
            in_features: input dim
            out_features: output dim
            ex_max_spike_time: unit ms
            tau: unit ms, the time constant of LIF model
            threshold: unit V, the threshold of LIF model
            bias: bias of linear
            debug: debug mode
        """
        super().__init__(in_features, out_features, bias)
        logger.error(
            "LIFImpulseSpikeTimeLinear is waitting for modifying since the spike time cannot be calculated correctly")
        self.debug = debug
        self.threshold = threshold
        self.ex_max_spike_time = ex_max_spike_time
        self.tau = tau
        torch.nn.init.uniform_(self.weight, 0.0, 10 / self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input is the spike time tensor in pre layer.
        This layer will compute output spike time by input spike time and weight.
        The grad will be automatically calculated by autograd which consists of the derivative of exp function.
        Args:
            x: the exponential form of input spike time tensor$\exp(t/\taul)$, whose shape is [B,feature_in]
        Returns:
            x : the exponential form of output spike time tensor$\exp(t/\taul)$, whose shape is [B,feature_out]
        """

        # Since the output spike time only depends on the spike before spike time, and is not clear at first.
        # So we will take every time step of input spike into consideration for the smallest output spike time.
        # To make it easier, the input spike time will be sorted in ascending order.And then torch.cumsum will be conducted
        # So that, the membrain potential will be cumulated in time order.Then we will check the output spike time for
        # each input, and the smallest output spike time will be selected as the output spike time.

        B = x.shape[0]
        sorted_input, indices = torch.sort(x)  # Both [B,feature_in],

        # Extend the input and weight for compute
        sorted_input = sorted_input.unsqueeze(-1)  # [B,feature_in,1]
        sorted_input = torch.tile(sorted_input, (1, 1, self.out_features))  # [B,feature_in,feature_out]

        indices = indices.unsqueeze(-1).requires_grad_(False)  # [B,feature_in,1]
        indices = torch.tile(indices, (1, 1, self.out_features)).requires_grad_(False)  # [B,feature_in,feature_out]

        weight_extend = self.weight.unsqueeze(0).permute(0, 2, 1)  # [1,feature_in,feature_out]
        weight_extend = torch.tile(weight_extend, (B, 1, 1))  # [B,feature_in,feature_out]
        weight_sorted = torch.gather(weight_extend, 1, indices)

        # Compute the output spike time
        # Since the input spike is delta impulse, the neuron model is lif.
        # The membrane potential is $$V_j(t) = \sum w_{ij} \exp(\frac{t_i-t}{\tau})$$.i denotes the input spike.
        # Assume threshold is $\theta$. The output spike time will be $$t_j = \tau\ln(\frac{\sum_i w_{ij}\exp(t_i/\tau)}{\theta})$$
        # To make the compute simple ,the spike time could be delivered in exp form.The formulation could be rewritten as:
        # $$\exp(\frac{t_j}{\tau}) = (\frac{\sum_i w_{ij}\exp(t_i/\tau)}{\theta})$$
        # However, for a LIF neuron, the membrane potential is not monolithic ascending,
        # So the exact spike time cannot be computed by this way

        input_weight_mul = torch.multiply(weight_sorted, sorted_input)
        input_weight_sum = torch.cumsum(input_weight_mul, dim=1)
        output_spike_time = torch.div(input_weight_sum, self.threshold)  # [B,feature_in,feature_out]

        # Select the first spike time for each feature
        output_spike_time_valid = torch.where(output_spike_time < 1, self.ex_max_spike_time, output_spike_time)
        # TODO finish the spike time pick.
        out, _ = torch.min(output_spike_time_valid, dim=1)  # [B,feature_out]

        # search the nearest input spike time
        out_expanded = out.unsqueeze(1)  # [B,1,feature_out]
        mask = sorted_input < out_expanded  # [B,feature_in,feature_out]
        sorted_input_masked = torch.where(mask, sorted_input, -torch.inf)
        nearest_input, _ = torch.max(sorted_input_masked, dim=1)

        # Use the old gradient path
        out_final = nearest_input.detach() + out - out.detach()
        return out_final


class IFTTFSLinear(nn.Linear):
    """
    A linear layer for TTFS encoded net.
    The data flow inside the layer is spike time instead of spike train.
    The shape out spike is exp(-t), for convenience the data is exp(t) instead of t.
    """

    def __init__(self, in_features: int, out_features: int, threshold: float, max_spike_time, tau: float = 1,
                 bias: bool = None, debug: bool = False) -> None:
        """
        Init function for IFTTFSLinear layer.
        Args:
            in_features: input features channel
            out_features: output feature channel
            threshold: voltage threshold for IF neuron
            max_spike_time: latest spike time in the compute
            tau: decay factor of the output spike.
            bias: not implemented yet
            debug: debug mode
        """
        super().__init__(in_features, out_features, bias)
        self.debug = debug
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.threshold = threshold
        self.tau = tau
        self.max_spike_time = max_spike_time
        torch.nn.init.uniform_(self.weight, 0.0, 10 / self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact spike time of IF model could be computed by the formula:
        $$
        \exp(t_out) = \frac{\sum_i w_{ij}\exp(t_i)}{(\sum_i w_{ij}-\theta)}
        $$
        See paper "Rethinking skip connections in Spiking Neural Networks with Time-To-First-Spike coding" for more detail
        Args:
            x: exponential spike time shape is [B,C_in],if used in conv layer could be [B,L,C_in]
        Returns:
            out: exponential spike time shape is [B,C_out]
        """
        exp_max_time = np.exp(self.max_spike_time)
        L = None
        if x.ndim == 3:
            B, L, C_in = x.shape
        else:
            B, C_in = x.shape

        # sort input to get the first spike time
        sorted_input, indices = torch.sort(x)
        indices = torch.unsqueeze(indices, -1).requires_grad_(False)
        weight_extend = self.weight.unsqueeze(0).permute(0, 2, 1)  # [1,C_in,C_out]
        if L is not None:
            indices = torch.tile(indices, (1, 1, 1, self.out_features))  # [B,L,C_in,C_out]
            sorted_input = sorted_input.unsqueeze(-1).expand(-1, -1, -1, self.out_features)  # [B,L,C_in,C_out]
            weight_extend = torch.tile(weight_extend, (B, L, 1, 1))  # [B,L,C_in,C_out]
        else:
            indices = torch.tile(indices, (1, 1, self.out_features))  # [B,C_in,C_out]
            sorted_input = sorted_input.unsqueeze(-1).expand(-1, -1, self.out_features)  # [B,C_in,C_out]
            weight_extend = torch.tile(weight_extend, (B, 1, 1))  # [B,C_in,C_out]

        weight_sorted = torch.gather(weight_extend, -2, indices)

        weigh_input_mul = torch.multiply(weight_sorted, sorted_input)  # [B,C_in,C_out]
        weight_sum = torch.cumsum(weight_sorted, dim=-2)
        weight_input_sum = torch.cumsum(weigh_input_mul, dim=-2)  # [B,C_in,C_out]

        out_all = torch.div(weight_input_sum, torch.clamp(weight_sum - self.threshold, 1e-10, 1e10))
        # incase negative weight
        out_spike = torch.where(weight_sum > self.threshold, out_all, torch.full_like(out_all, exp_max_time))
        # don't know why seems impossible to be lower
        out_spike = torch.where(out_spike < sorted_input, torch.full_like(out_spike, exp_max_time), out_spike)
        out_spike_time_valid = torch.where(out_spike < sorted_input, torch.full_like(out_spike, exp_max_time),
                                           out_spike)
        # incase new input emits an earlier spike,feels useless too
        sorted_input_slice = sorted_input[..., (1 - self.in_features):, :]
        one_tensor = torch.ones_like(sorted_input_slice)[..., 0:1, :].requires_grad_(False)
        sorted_input_cat = torch.cat((sorted_input_slice, one_tensor), dim=-2)
        out_spike = torch.where(out_spike >= sorted_input_cat, torch.full_like(out_spike, exp_max_time), out_spike, )

        # earliest spike time
        out_spike_time, _ = torch.min(out_spike, dim=-2)
        return out_spike_time


class IFConv2D(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, stride: int, padding: int,
                 max_spike_time: int, threshold: float):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_spike_time = max_spike_time
        self.threshold = threshold
        self.kernel = IFTTFSLinear(in_channel * kernel_size * kernel_size, out_channel, threshold, max_spike_time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of IFConv2D layer.
        Args:
            x: Input tensor shape is [B,C,H,W]
        Returns:
            spike: Binary tensor, shape is [B,C_out,H_out,W_out].
        """
        B, C_in, H, W = x.shape
        patches = F.unfold(x, self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2, 1)
        patches = torch.where(torch.lt(patches, 0.1), torch.full_like(patches, np.exp(self.max_spike_time)), patches)
        out = self.kernel(patches).transpose(-1, -2)
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(B, self.out_channel, H_out, W_out)
        return out


class SpikeLinear(nn.Linear):
    """
    A linear layer for TTFS encoded net. Run linear for spike train.
    This layer will do linear each time frame in spike train.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, debug: bool = False) -> None:
        super().__init__(in_features, out_features, bias)
# TODO to be finished
