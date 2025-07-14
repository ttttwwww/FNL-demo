"""
Network assembles the layers provided in layer module and from torch.
To bo more specific the neuron layer should be in layer module.
And the linear layer and convention layer could come from nn.Module

Custom networks should be created for different task
Here are some examples.

To define diverse network with more combo of layers or neurons. The network and net part in config should be modified.
"""
import numpy as np
import torch

from snn_simulator.utils.spike_convert import spike_train_to_spike_time
from .neuron_layer import TemNeuronLayer
from snn_simulator.utils.config_utils import merge_config, get_object_from_str
from .base import BaseTimeStepNetwork
from snn_simulator.surrogate.layer_surrogate import S4NNSpikeTrainToSpikeTime
from snn_simulator.utils.hook import TensorCollector

from torch import nn


class FlexibleTimeStepNet(BaseTimeStepNetwork):
    def __init__(self, layer_configs, tensor_collector: TensorCollector, debug: bool = False, **kwargs) -> None:
        super().__init__(debug, **kwargs)
        self.tensor_collector = tensor_collector
        self.layers = nn.ModuleList()
        self.layer_configs = layer_configs
        self.net_init()

    def net_init(self):
        """
        Init network parameters.
        The network comprises multiple linear neuron layers.
        To make parameters searching easier, the framework of net work could be modified by config file.
        """
        for i, cfg in enumerate(self.layer_configs):
            layer_cls = get_object_from_str(cfg["type"])
            if cfg["args"] is None:
                cfg["args"] = {}
            if "snn_simulator" in cfg["type"]:
                cfg["args"]["debug"] = self.debug
            if "neuron_layer" in cfg["type"]:
                if self.debug:
                    cfg["args"]["surrogate_params"]["tensor_collector"] = self.tensor_collector
            layer = layer_cls(**cfg["args"])
            self.layers.append(layer)

    def forward(self, x):
        """
        Forward propagation for flexible network.
        Args:
            x: Input data [B,T,C,H,W] or [B,T,C]
        Returns:
            output: Output data
        """
        for layer in self.layers:
            x = layer(x)
        return x


class ImgTimeStepNet(BaseTimeStepNetwork):
    # TODO quantizer need be to add
    # TODO change the init method make it same as other classes such as optimizer
    def __init__(self, input_size: int, output_size: int, hidden_sizes: [int], init_weight_uppers: [int],
                 neuron_cls: str, neuron_params: dict, surrogate_cls: str, surrogate_params: dict,
                 tensor_collector: TensorCollector, debug: bool = False, **kwargs):
        super().__init__(debug, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.init_weight_uppers = init_weight_uppers
        self.neuron_cls = neuron_cls
        self.neuron_params = neuron_params
        self.surrogate_cls = surrogate_cls
        self.surrogate_params = surrogate_params
        self.tensor_collector = tensor_collector
        self.net_init()

    def net_init(self) -> None:
        """
        Init network parameters.
        The network comprises multiple linear neuron layers.
        To make parameters searching easier, the framework of net work could be modified by config file.
        """
        input_size = self.input_size
        output_size = self.output_size
        hidden_sizes = self.hidden_sizes
        init_weight_uppers = self.init_weight_uppers
        neuron_cls = self.neuron_cls
        neuron_params = self.neuron_params
        surrogate_cls = self.surrogate_cls
        surrogate_params = self.surrogate_params
        surrogate_params["tensor_collector"] = self.tensor_collector
        if hidden_sizes is None:
            hidden_sizes = []
        previous_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layer = TemNeuronLayer(input_size=previous_size, output_size=hidden_size, neuron_cls=neuron_cls,
                                   neuron_params=neuron_params,
                                   surrogate_cls=surrogate_cls, surrogate_params=surrogate_params,
                                   init_weight_upper=init_weight_uppers[i], debug=self.debug)
            self.layers.append(layer)
            previous_size = hidden_size
        output_layer = TemNeuronLayer(input_size=previous_size, output_size=output_size, neuron_cls=neuron_cls,
                                      neuron_params=neuron_params,
                                      surrogate_cls=surrogate_cls, surrogate_params=surrogate_params,
                                      init_weight_upper=init_weight_uppers[-1], debug=self.debug)
        self.layers.append(output_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return S4NNSpikeTrainToSpikeTime.apply(x)


class FlexibleMultimodalTimeStepNet(BaseTimeStepNetwork):
    def __init__(self, layer_configs, tensor_collector: TensorCollector, debug: bool = False, **kwargs) -> None:
        super().__init__(debug, **kwargs)
        self.tensor_collector = tensor_collector
        self.layers = nn.ModuleList()
        self.img_layers = nn.ModuleList()
        self.audio_layers = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()
        self.layer_configs = layer_configs
        self.net_init()

    def net_init(self):
        """
        Init network parameters.
        The network comprises multiple linear neuron layers.
        To make parameters searching easier, the framework of net work could be modified by config file.
        """
        img_configs = self.layer_configs["img_layers"]
        audio_configs = self.layer_configs["audio_layers"]
        fuse_configs = self.layer_configs["fuse_layers"]
        # img layers
        for i, cfg in enumerate(img_configs):
            layer_cls = get_object_from_str(cfg["type"])
            if cfg["args"] is None:
                cfg["args"] = {}
            if "snn_simulator" in cfg["type"]:
                cfg["args"]["debug"] = self.debug
            if "neuron_layer" in cfg["type"]:
                if self.debug:
                    cfg["args"]["surrogate_params"]["tensor_collector"] = self.tensor_collector
            layer = layer_cls(**cfg["args"])
            self.img_layers.append(layer)
        # audio layers
        for i, cfg in enumerate(audio_configs):
            layer_cls = get_object_from_str(cfg["type"])
            if cfg["args"] is None:
                cfg["args"] = {}
            if "snn_simulator" in cfg["type"]:
                cfg["args"]["debug"] = self.debug
            if "neuron_layer" in cfg["type"]:
                if self.debug:
                    cfg["args"]["surrogate_params"]["tensor_collector"] = self.tensor_collector
            layer = layer_cls(**cfg["args"])
            self.audio_layers.append(layer)
        # fuse layers
        for i, cfg in enumerate(fuse_configs):
            layer_cls = get_object_from_str(cfg["type"])
            if cfg["args"] is None:
                cfg["args"] = {}
            if "snn_simulator" in cfg["type"]:
                cfg["args"]["debug"] = self.debug
            if "neuron_layer" in cfg["type"]:
                if self.debug:
                    cfg["args"]["surrogate_params"]["tensor_collector"] = self.tensor_collector
            layer = layer_cls(**cfg["args"])
            self.fuse_layers.append(layer)

    def forward(self, multimodal_data):
        """
        Forward propagation for flexible network.
        Args:
            multimodal_data: Multimodal input data
        Returns:
            output: Output data
        """
        img_data, audio_data = multimodal_data
        for layer in self.img_layers:
            img_data = layer(img_data)#[B,T,C] output should be flattened
        for layer in self.audio_layers:
            audio_data = layer(audio_data)#[B,T,C] output should be flattened
        x = torch.cat([img_data, audio_data], dim=2)
        for layer in self.fuse_layers:
            x = layer(x)
        return x

    def get_vmem(self) -> list[torch.Tensor]:
        """
        Get the voltage membrane of each layer in the network.
        Returns:
            vmem_list: List of voltage membrane tensors for each layer.
        """
        vmem_list = []
        for layer in self.img_layers:
            if hasattr(layer, 'get_vmem'):
                vmem_list.append(layer.get_vmem())
        for layer in self.audio_layers:
            if hasattr(layer, 'get_vmem'):
                vmem_list.append(layer.get_vmem())
        for layer in self.fuse_layers:
            if hasattr(layer, 'get_vmem'):
                vmem_list.append(layer.get_vmem())
        return vmem_list

    def batch_reset(self)->None:
        """
        Reset neuron's volt, for different batches.
        """
        for layer in self.img_layers:
            if hasattr(layer, 'batch_reset'):
                layer.batch_reset()
        for layer in self.audio_layers:
            if hasattr(layer, 'batch_reset'):
                layer.batch_reset()
        for layer in self.fuse_layers:
            if hasattr(layer, 'batch_reset'):
                layer.batch_reset()