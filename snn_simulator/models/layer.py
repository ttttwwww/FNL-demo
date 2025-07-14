# from . import logger
# import torch.nn as nn
# from .base import BaseLayer
# from snn_simulator.neuron.base import BaseNeuron
# from typing import Type, Union
# import torch
# from snn_simulator.utils.config_utils import get_object_from_str
# from snn_simulator.utils.hook import TensorCollector
# from snn_simulator.surrogate.layer_surrogate import S4NNSpikeTrainToSpikeTime
#
#
# class BaseNeuronLayer(BaseLayer):
#     """
#     Implement basic function of neuron layer.Including
#         Init neurons
#         Define data flow
#     """
#
#     def __init__(self, input_size, output_size, neuron_cls: str, neuron_params: dict, device:str,debug: bool = False):
#         """
#         Neuron init function.
#         Args:
#             input_size: Equal to neuron number
#             output_size: Equal to neuron number
#             neuron_cls: Class of neuron in layer
#             neuron_params: Parameter to create neurons in layer
#         """
#         super().__init__(device=device, debug=debug)
#         self.input_size = input_size
#         self.output_size = output_size
#         self.neuron_cls = get_object_from_str(neuron_cls)
#         self.neuron_params = neuron_params
#         self.neuron_params["debug"] = debug
#         try:
#             self.neuron = self.neuron_cls(**neuron_params)
#         except Exception as e:
#             logger.error(e)
#             logger.error("check if the neuron parameters are compatible for neuron class")
#             if self.debug:
#                 raise e
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         """
#         Forward function of neuron layer.Should be implemented in subclass.
#         Args:
#             input: Input tensor
#
#         Returns:
#             output: Output tensor
#         """
#         raise NotImplementedError
#
#     def batch_reset(self) -> None:
#         """
#         Entrance for resetting voltage of neurons.
#         """
#         self._reset_membrane()
#
#     def _reset_membrane(self) -> None:
#         """
#         Default reset method
#         """
#         self.neuron.batch_reset()
#
#
# class FreNeuronLayer(BaseNeuronLayer):
#     """
#     Special Neuron Layer for frequency encoded SNN.
#     Normally frequency-encoded network will be updated by BPTT.So this layer will run in step manner.
#     Which means each time call this function, the simulation will last for one time step.
#     And the backward will go as the calculation map maintained by pytorch.
#     So the surrogate function is implemented in neuron.
#     """
#
#     def __init__(self, input_size, output_size, neuron_cls: str, neuron_params: dict):
#         """
#         Fre Neuron init function.
#         Args:
#             input_size:See BaseNeuronLayer.
#             output_size: See BaseNeuronLayer.
#             neuron_cls: See BaseNeuronLayer.
#             neuron_params: See BaseNeuronLayer.
#         """
#         super().__init__(input_size, output_size, neuron_cls, neuron_params)
#         if input_size != output_size:
#             logger.error(
#                 f"neuron layer {self.__class__.__name__} input size {input_size} "
#                 f"does not equal to output size {output_size}")
#         assert input_size == output_size
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Define dataflow in frequency-encoded network.Backward propagation goes as calculation map.
#         The surrogate function will be implemented in neuron.
#         Args:
#             x: input tensor shape is [batch_size,neuron_number]
#
#         Returns:
#             output: output tensor shape is [batch_size,neuron_number]
#         """
#         x = self.neuron(x)
#         return x
#
#
# class TemNeuronLayer(BaseNeuronLayer):
#     """
#     Temporal Neuron Layer for temporal encoded SNN.
#     Usually custom surrogate function will be used for backward propagation.
#     And the spike time of pre neuron and post neuron is used for gradiant calculation.
#     So the calculation map inside layer will be cut by surrogate function, and weight will be built in the neuron layer.
#     """
#
#     def __init__(self, input_size: int, output_size: int, neuron_cls: str, neuron_params: dict,
#                  surrogate_cls: str, surrogate_params: dict, init_weight_upper: int, device: str,
#                  debug: bool = False) -> None:
#         """
#         Temporal Neuron init function.Init surrogate function as well.
#         In order to make sure only x is passed into surrogate function in forward propagation.
#         Args:
#             input_size: See BaseNeuronLayer.
#             output_size: See BaseNeuronLayer.
#             neuron_cls: See BaseNeuronLayer.
#             neuron_params: See BaseNeuronLayer.
#             surrogate_cls: Surrogate function class. Be defined in module surrogate.
#             surrogate_params: parameters for surrogate function.
#             init_weight_upper: Initial weight upper for surrogate function.
#
#             device: Device for surrogate function.
#             debug: Debug mode.
#         """
#         super().__init__(input_size, output_size, neuron_cls, neuron_params, device,debug)
#         try:
#             surrogate_params["neuron"] = self.neuron
#             surrogate_cls = get_object_from_str(surrogate_cls)
#             self.surrogate = surrogate_cls(**surrogate_params)
#         except Exception as e:
#             logger.error(e)
#             logger.error(f"surrogate {surrogate_cls.__name__} params are not compatible")
#             if self.debug:
#                 raise e
#         self.weight = nn.Parameter(torch.randn(input_size, output_size).to(device))
#         self.init_weight_upper = init_weight_upper
#         self.init_weights()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         forward propagation of neuron layer.
#         Args:
#             x: Input spike train shape is [batch_size,num_timestep,neuron_number]
#
#         Returns:
#             x: Output spike train shape is [batch_size,num_timestep,neuron_number]
#         """
#         return self.surrogate(x, self.weight)
#
#     def init_weights(self) -> None:
#         """
#         Initialize weights using uniform distribution.The lower bond is 0 and the upper bond is init_weight_upper.
#         """
#         self.weight.data.uniform_(0, self.init_weight_upper)
#
#     def get_weight(self) -> torch.Tensor:
#         return self.weight
#
#
#
