import torch
import torch.nn as nn

class MyHeaviside(nn.Module):
    def __init__(self):
        super().__init__()
        self.surrogate = MyHeavisideFunction
    def forward(self, x):
        x = self.surrogate.apply(x)
        return x

class MyHeavisideFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.heaviside(x,torch.tensor(0.).to(device=x.device,dtype=x.dtype))
    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output,device=grad_output.device)