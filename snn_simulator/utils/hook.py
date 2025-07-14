import torch


class TensorCollector:
    def __init__(self,debug:bool=True):
        """
        Use dict to collect gradients in hook way
        Args:
            debug: Only activate in debug mode
        """
        self.grads = {}
        self.custom_tensors = {}
        self.debug = debug

    def store_grad_param(self, key:str, grad:torch.Tensor)->None:
        """
        store gradient
        Args:
            key: tensor name
            grad: tensor
        """
        if self.debug:
            self.grads[key] = grad.detach().cpu().numpy()

    def store_temp_tensor(self,key:str,tensor:torch.Tensor)->None:
        if self.debug:
            self.custom_tensors[key] = tensor.detach().cpu().numpy()


