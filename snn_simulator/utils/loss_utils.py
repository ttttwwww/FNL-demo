"""
This Module define the loss and acc compute function.
To meet the simulator requirements, the output might be different.
So all the function should handler the output direct from the network in front of the loss or acc compute.

"""
from typing import Any

import torch
from snn_simulator.utils.encode import s4nn_encode_label

def cross_entropy_acc_count(output: torch.Tensor, label: torch.Tensor,ctx=None) -> [int, torch.Tensor]:
    prediction = torch.argmin(output, dim=1)
    correct = torch.sum(prediction == label).float().item()
    return correct,output

def ttfs_acc_count(output: torch.Tensor, label: torch.Tensor, ctx: dict[Any]) -> [int, torch.Tensor]:
    """
    Count correct number in a batch data.Output is from TTFS net shape is [batch_size,num_classes].
    The value denotes the spike time in each channel.So the channel has the first spike is the prediction.
    Label is int value denotes the correct channel number.
    Args:
        output:Network output shape is [batch_size,num_classes].
        label: Dataset label shape is [batch_size].
        ctx: context during training for extra information including:
        {
            'model': model,  # The model used for training
            "max_spike_time": max_spike_time,  # Maximum spike time for S4NN denotes all the channel didn't spike.
        }
    Returns:
        num_correct:Number of correct samples.
        prediction:Prediction index of samples.
    """
    #TODO 修改正确判断再尝试一下
    model = ctx["model"]
    max_spike_time = ctx["max_spike_time"]
    vmem = model.get_vmem()[-1]
    min_times,prediction = output.min(1)
    vmem_pred = torch.argmax(vmem, dim=1)
    final_pred = torch.where(min_times>= max_spike_time, vmem_pred,prediction)

    acc = torch.sum(final_pred == label).float().item()

    # min_time = torch.min(output, dim=1)[0]
    # label_time = output[:,label][:,0]
    # acc = torch.sum(label_time == min_time).float().item()
    return acc,prediction


def s4nn_mes_loss(output: torch.Tensor, label: torch.Tensor, max_spike_time: int,delay_timestep:int) -> torch.Tensor:
    """
    MSE loss function for S4NN.The target spike time is adjusted by the S4NN way.See the details in paper.
    Args:
        output:Network output shape is [batch_size,num_classes].
        label:Dataset label shape is [batch_size].
        max_spike_time: maximum spike time.
        delay_timestep: delay timestep for incorrect channels.
    Returns:
        loss:Loss for prediction and label.
    """
    target_time = s4nn_encode_label(output, label, max_spike_time, delay_timestep)
    loss = -torch.mean(0.5 * (target_time - output) ** 2)
    return loss

