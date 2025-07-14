"""
Define the surrogate function for entire layer.
"""
from . import logger
import torch
import torch.nn as nn
from snn_simulator.neuron.base import BaseNeuron
from snn_simulator.utils.spike_convert import spike_train_to_spike_time, spike_time_to_spike_train
from .base import BaseLayerSurrogate
from snn_simulator.utils.hook import TensorCollector
import torch.nn.functional as F


class S4NNSurrogate(BaseLayerSurrogate):
    _counter = 0  # Class counter to generate different tag for different surrogate for collecting params in debug
    """
    A surrogate function for temporal network, especially for TTFS(Time-To-First-Spike) network.
    The surrogate function use the first spike time to calculate the gradiant.
    See details in paper "Temporal Backpropagation for Spiking Neural Networks with One Spike per Neuron"
    """

    def __init__(self, neuron: BaseNeuron, max_spike_time: int, tensor_collector: TensorCollector = None):
        """
        Extra parameters for forward and backward propagation.
        Args:
            neuron: The neuron to use.
            max_spike_time: Max timestep during the simulation.
            tensor_collector: The gradient collector.
        """
        super(S4NNSurrogate, self).__init__()
        self.id = S4NNSurrogate._counter
        S4NNSurrogate._counter += 1
        self.neuron = neuron
        self.max_spike_time = max_spike_time
        self.tensor_collector = tensor_collector

    def forward(self, x: torch.Tensor, weight) -> torch.Tensor:
        id = f"{self.__class__.__name__}_{self.id}"
        return S4NNSurrogateFun.apply(x, weight, self.neuron, self.max_spike_time, id, self.tensor_collector)


class S4NNSurrogateFun(torch.autograd.Function):
    # TODO Could be implemented better to reduce check time.
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, neuron: BaseNeuron, max_spike_time: int, collect_id: str,
                tensor_collector: TensorCollector = None, ) -> torch.Tensor:
        """
        Forward propagation.
        Args:
            ctx: context object.
            x: Input spike trains.[batch_size,num_time_step,num_pre_neuron]
            weight: Weight of the neuron layer. Shape is [num_pre_neuron,num_post_neuron]
            neuron: The neuron to use.
            max_spike_time: Max timestep of spike trains.
            collect_id: ID for each surrogate function in debug.
            tensor_collector: collector for debug
        Returns:
            x: Output spike trains.[batch_size,num_time_step,num_curr_neuron]
        """
        if tensor_collector is None:
            flag_collect = False
        else:
            flag_collect = True
        ctx.flag_collect = flag_collect
        with torch.no_grad():  # To cut the torch's calculation map
            # Record input spikes time
            in_spike_time = spike_train_to_spike_time(x)[:, :, 0]
            in_spike_time = torch.where(in_spike_time >= max_spike_time, max_spike_time, in_spike_time)
            # Multiply the weight
            # [batch_size,num_time_step,num_pre_neuron] -> [batch_size,num_time_step,num_curr_neuron]
            x = x.matmul(weight)
            # Input spike train to neurons and record outputs
            out = []
            for i in range(x.shape[1]):
                out.append(neuron(x[:, i, :]))
            out = torch.stack(out).transpose(0, 1)
            # out shape is [batch_size,time_step,num_post_neuron]
            # Convert output trains to output spikes time
            # [:,:,0] For the first spike
            out_spike_time = spike_train_to_spike_time(out)[:, :, 0]
            out_spike_time = torch.where(out_spike_time < max_spike_time, out_spike_time + 1, max_spike_time)
            reversed_out = spike_time_to_spike_train(out_spike_time.unsqueeze(-1), max_spike_time)
            # Save for backward
            ctx.save_for_backward(in_spike_time, out_spike_time)
            ctx.num_time_step = x.shape[1]
            ctx.weight = weight.detach()
            ctx.ex_max_spike_time = max_spike_time
            ctx.neuron = neuron
            ctx.collect_id = collect_id
            ctx.tensor_collector = tensor_collector
        return reversed_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward propagation.Replace the neuron's backward propagation.
        Args:
            ctx: context
            grad_output: gradient from backward.

        Returns:
            grad_input: gradient pass to forward.
        """
        flag_collect = ctx.flag_collect
        with torch.no_grad():
            # TODO 确定一下inf时间步到底怎么修改,以及最后一层的时间步怎么弄.确定一下未发放神经元之间的联系
            """
            Forward propagation tensor shape is [batch_size, num_time_step, num_pre_neuron]
            However, backward propagation is broken by S4NN layer.The grad needed is for weight whose shape is [num_pre_neuron,num_post_neuron].
            So there is an extra dimension in grad_output.We expand the grad_in from [batch_size, num_pre_neuron] to
            [batch_size, num_time_step,num_pre_neuron] by repeating.So we only need the first tensor in dim1.
            """
            collect_id = ctx.collect_id
            if flag_collect:
                tensor_collector = ctx.tensor_collector
            grad_output = grad_output[:, 0, :]
            if flag_collect:
                tensor_collector.store_temp_tensor(f"{collect_id}_grad_output_before_norm", grad_output)
            norm = grad_output.norm(dim=1,keepdim=True)
            grad_output = grad_output / (norm+1e-8)
            # normalization. The paper says it could improve accuracy.
            if flag_collect:
                tensor_collector.store_temp_tensor(f"{collect_id}_grad_output", grad_output)
            in_spike_time, out_spike_time, = ctx.saved_tensors
            if flag_collect:
                tensor_collector.store_temp_tensor(f"{collect_id}_in_spike_time", in_spike_time)
                tensor_collector.store_temp_tensor(f"{collect_id}_out_spike_time", out_spike_time)
            # if neuron.debug:
            #     neuron.hook_in_spike_time.append(in_spike_time.detach().cpu().numpy())
            #     neuron.hook_out_spike_time.append(out_spike_time.detach().cpu().numpy())
            weight = ctx.weight
            in_spike_time_expanded = in_spike_time.unsqueeze(-1)  # shape: (batch_size, dim_in, 1)
            out_spike_time_expanded = out_spike_time.unsqueeze(1)  # shape: (batch_size, 1, dim_out)

            # Compare spike order. Could change the value of has_spike for better performance.
            has_spiked = torch.where(out_spike_time_expanded - in_spike_time_expanded > 0, 1, 0)
            if flag_collect:
                tensor_collector.store_temp_tensor(f"{collect_id}_has_spiked", has_spiked)

            grad_tmp = torch.einsum("bj,bij->bij", grad_output, has_spiked)
            if flag_collect:
                tensor_collector.store_temp_tensor(f"{collect_id}_grad_tmp", grad_tmp)
            grad_weight = torch.einsum("bij->ij", grad_tmp)
            if flag_collect:
                tensor_collector.store_temp_tensor(f"{collect_id}_grad_weight", grad_weight)
            grad_in = torch.einsum("bij,ij->bi", grad_tmp, weight)
            norm = grad_in.norm(dim=1,keepdim=True)
            grad_in = grad_in / (norm+1e-8)
            grad_in = grad_in.unsqueeze(1)
            num_time_step = ctx.num_time_step
            grad_in = grad_in.expand(-1, num_time_step, -1)
            if flag_collect:
                tensor_collector.store_temp_tensor(f"{collect_id}_grad_in", grad_in)
        return grad_in, grad_weight, None, None, None, None


class S4NNSpikeTrainToSpikeTime(torch.autograd.Function):
    """
    The loss computation need spike time.This surrogate provide a transfer interface for spike trains and grads.
    """

    @staticmethod
    def forward(ctx, spike_train: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_train: Input spike trains.shape is [batch_size,num_time_step,num_neuron]

        Returns:
            spike_time: Output spike time.shape is [batch_size,num_channels,max_num_spike]
        """
        spike_time = spike_train_to_spike_time(spike_train)[:, :, 0]
        ctx.time_step = spike_train.shape[1]
        return spike_time

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        time_step = ctx.time_step
        grad_output = grad_output.unsqueeze(1).expand(-1, time_step, -1)
        return grad_output, None


class TemConvSurrogate(BaseLayerSurrogate):
    _cnt = 0

    def __init__(self, neuron: BaseNeuron, weight: nn.Parameter, kernel_size: int, padding: int, max_spike_time: int, stride: int = 1,
                 bias: torch.Tensor = None, tensor_collector: TensorCollector = None,):
        """
        Extra parameters for forward and backward propagation.
        Args:
            neuron: The neuron to use.
            kernel_size: Kernel size of the convolution.
            padding: Padding of the convolution.
            stride: Stride of the convolution.
            tensor_collector: The gradient collector.
            max_spike_time: Max timestep during the simulation.
        """
        super().__init__()
        self.neuron = neuron
        self.weight = weight
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.tensor_collector = tensor_collector
        self.max_spike_time = max_spike_time
        self.id = TemConvSurrogate._cnt
        self.bias = bias
        # TODO add bias for the convolution
        TemConvSurrogate._cnt += 1

    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        collect_id = f"{self.__class__.__name__}_{self.id}"
        return TemConvFunction.apply(spike_train, self.weight, self.neuron, self.bias, self.padding,
                                     self.max_spike_time,
                                     self.kernel_size, collect_id, self.stride, self.tensor_collector)


class TemConvFunction(torch.autograd.Function):
    """
    Convolution kernel for TemConv Layer
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, neuron: BaseNeuron, bias, padding: int, max_spike_time: int,
                kernel_size: int, collect_id: str, stride: int = 1,
                tensor_collector: TensorCollector = None) -> torch.Tensor:
        """
        Forward of the time convolution.
        Args:
            x:input spike train [B,T,C,H,W]
        """
        ctx.collect_id = collect_id
        ctx.flag_collect = False
        if tensor_collector is not None:
            ctx.tensor_collector = tensor_collector
            ctx.flag_collect = True

        with torch.no_grad():
            [B, T, C, H, W] = x.shape
            input_spikes = x.view(B, T, -1)
            x = x.reshape(B * T, C, H, W)  # [T*B,C,H,W]
            x = F.conv2d(x, weight=weight, bias=bias, stride=stride, padding=padding)
            _, C_out, H_out, W_out = x.shape
            x = x.view(B, T, C_out, H_out, W_out)  # [T,B,C,H,W]
            conv_spikes = x.view(B, T, -1)
            output_spikes = torch.zeros(B, T, C_out, H_out, W_out).to(x.device)
            for t in range(T):
                output_spikes[:, t, :, :, :] = neuron(conv_spikes[:, t, :]).view(B, C_out, H_out, W_out)
            input_spike_time = spike_train_to_spike_time(input_spikes)[:, :, 0]
            input_spike_time = input_spike_time.view(B, C, H, W)
            output_spike_time = spike_train_to_spike_time(output_spikes.view(B, T, -1))[:, :, 0]
            output_spike_time = torch.where(output_spike_time==torch.inf,max_spike_time,output_spike_time)
            output_spike_time = torch.where(output_spike_time < max_spike_time, output_spike_time + 1,
                                            max_spike_time)

            reversed_out = spike_time_to_spike_train(output_spike_time.unsqueeze(-1), max_spike_time)  # [B,T,channels]
            y = reversed_out.view(B, T, C_out, H_out, W_out)
            output_spike_time = output_spike_time.view(B, C_out, H_out, W_out)

            ctx.save_for_backward(input_spike_time, output_spike_time, weight.detach())
            ctx.kernel_size = kernel_size
            ctx.padding = padding
            ctx.stride = stride
            ctx.input_shape = [B, T, C, H, W]
            ctx.output_shape = y.shape
            ctx.max_spike_time = max_spike_time
            ctx.tensor_collector = tensor_collector
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO check if the grad goes correct
        collect_id = ctx.collect_id
        flag_collect = ctx.flag_collect
        tensor_collector = ctx.tensor_collector
        kernel_size = ctx.kernel_size
        padding = ctx.padding
        stride = ctx.stride
        max_spike_time = ctx.max_spike_time
        B, T, C, H, W = ctx.input_shape
        B, T, C_out, H_out, W_out = ctx.output_shape
        # Compute the grad.Since there is no exact spike time, grad can not be computed.
        # Grad_in will be the outcome of grad_output dot weight
        # Grad_weight will be the sum of corresponding grad_output
        # To be noted,  the grad of pre-neuron who spikes later than post-neuron will be 0.
        input_spike_time, output_spike_time, weight = ctx.saved_tensors  # [B,C,H,W],[B,C_out,H_out,W_out],[C_out,C,kH,kW]
        if flag_collect:
            tensor_collector.store_temp_tensor(f"{collect_id}_input_spike_time", input_spike_time)
            tensor_collector.store_temp_tensor(f"{collect_id}_output_spike_time", output_spike_time)
        input_spike_time_unfold = F.unfold(input_spike_time, kernel_size=kernel_size, stride=stride,
                                           padding=padding)  # [B,C*kH*kW,L]
        output_spike_time_unfold = output_spike_time.view(B, C_out, -1)  # [B,out_channel,L]
        #TODO try what will happen if the add the value of spike time into gradiant bp
        has_spiked = output_spike_time_unfold.unsqueeze(2) - input_spike_time_unfold.unsqueeze(
            1)  # [B,out_channel,C*kH*kW,L]
        has_spiked = torch.where(has_spiked > 0, 1, 0)
        # [B,C_out,C*kH*kW,L]
        weight_spike =output_spike_time_unfold.unsqueeze(2) - input_spike_time_unfold.unsqueeze(
            1)/max_spike_time*10

        if flag_collect:
            tensor_collector.store_temp_tensor(f"{collect_id}_has_spiked", has_spiked)
            tensor_collector.store_temp_tensor(f"{collect_id}_weight_spike", weight_spike)
        grad_output = grad_output[:,0,::]  # [B,T,out_channel,H,W]->[B,out_channel,H,W]
        grad_output_unfold = grad_output.view(B, C_out, -1)  # [B,out_channel,L]
        grad_tmp = torch.einsum("bij,bikj->bikj", grad_output_unfold, has_spiked)  # [B,C_out,C*kH*kW,L]
        weight_tmp = weight_spike * grad_tmp
        # grad_weight = torch.einsum("bikj->ik", grad_tmp)  # [C_out,C*kH*kW]
        grad_weight = torch.einsum("bikj->ik", weight_tmp)  # [C_out,C*kH*kW]

        grad_weight = grad_weight.view(weight.shape)
        if flag_collect:
            tensor_collector.store_temp_tensor(f"{collect_id}_grad_tmp", grad_tmp)
            tensor_collector.store_temp_tensor(f"{collect_id}_grad_weight", grad_weight)
            tensor_collector.store_temp_tensor(f"{collect_id}_weight_temp", weight_tmp)
            tensor_collector.store_temp_tensor(f"{collect_id}_grad_output_unfold", grad_output_unfold)
            tensor_collector.store_temp_tensor(f"{collect_id}_grad_output", grad_output)
        weight_flatten = weight.view(C_out,C*kernel_size*kernel_size)  # [C_out,C*kH*kW]
        grad_in = torch.einsum("bikj,ik->bkj", grad_tmp, weight_flatten)#[B,C*kH*kW,L]
        grad_in = F.fold(grad_in, output_size=(H,W),kernel_size=kernel_size, stride=stride, padding=padding) #[B,C,H,W]
        grad_in = grad_in.unsqueeze(1).expand(-1, T, -1, -1, -1)

        return grad_in, grad_weight, None, None, None, None,None,None,None,None
