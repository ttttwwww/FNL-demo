from . import logger
import torch
import torch.nn as nn




class BaseLayer(nn.Module):
    """
    Base layer for all layers.
    Including basic module like debug or device
    """
    def __init__(self, debug:bool=False,**kwargs):
        """
        Only cover dimension,rest will be implemented in subclass.
        Args:
            device: Dimension of inputs
            debug: Dimension of outputs
        """
        super(BaseLayer, self).__init__()
        self.debug = debug
        if kwargs:
            logger.warning(f"unused params {kwargs} in layer {self.__class__.__name__}")

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward process.Will be implemented in subclass.
        Args:
            x: Input tensor

        Returns:
            output: Output tensor
        """
        raise NotImplementedError

class BaseTimeStepNetwork(nn.Module):
    """
    BaseNet for time step based simulation.Basic function like reset,get_weight is implemented.
    Specific framework should be implemented in subclass.
    """

    def __init__(self,debug: bool = False,**kwargs):
        """
        Detailed parameters should be implemented in subclass.
        Args:
            debug:Debug mode
            device:Device which should be used
            kwargs:Extra unused params
        """
        super().__init__()
        logger.info(f"net {self.__class__.__name__} is initialized")
        self.debug = debug
        if self.debug:
            logger.debug(f"net {self.__class__.__name__} debug mode is on")
        self.layers = nn.ModuleList()
        self.neurons = []
        if kwargs:
            logger.warning(f"Ignoring unused params {kwargs}  in net {self.__class__.__name__}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def batch_reset(self)->None:
        """
        Reset neuron's volt, for different batches.
        """
        for layer in self.layers:
            if hasattr(layer, 'batch_reset'):
                layer.batch_reset()

    def net_init(self):
        """
        Initialize the neuron layer.Must be implemented in subclass.
        Returns:

        """
        raise NotImplementedError("Should be implemented in subclass")

    def get_weight(self) -> list[torch.Tensor]:
        """
        Get weight of each layer
        Returns:
            weight_list: weight of each layer
        """
        weight_list = []
        for layer in self.layers:
            if hasattr(layer, 'get_weight'):
                weight_list.append(layer.get_weight())
        return weight_list

    def get_config(self) -> dict:
        """
        Get the actual network params
        Returns:
            config: The dict used to build the network
        """
        config = {"net": self.net_param}
        return config

    def get_vmem(self) -> list[torch.Tensor]:
        """
        Get the membrane voltage of the last layer.
        Returns:
            vmem: Membrane voltage of the last layer
        """
        v_mem_list = []
        for layer in self.layers:
            if hasattr(layer, 'get_vmem'):
                v_mem_list.append(layer.get_vmem())

        return v_mem_list