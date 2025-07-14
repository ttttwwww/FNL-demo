import torch
import torch.nn.functional as F


def rgb_to_gray(image:torch.Tensor):
    gray = 0.299*image[0]+0.587*image[1]+0.114*image[2]
    return gray

def ttfs_encode(x: torch.Tensor(), max_spike_time: int):
    """
    Encode data in TTFS way.Input x shape is [batch_size,a,b,....] value should be in [0,1].
    The spike train shape will be [batch_size,max_spike_time,a,b,...].
    The value whose indices is int((1-x)* max_spike_time) in time axis will be set to 1.
    Args:
        x: shape[B,*dim] value should be in [0,1].
        max_spike_time: maximum spike time.

    Returns:
        out: shape[T,B,*dim] .
    """
    assert x.max() <= 1 and x.min() >= 0
    # flatten and insert time axis
    shape = x.shape
    x_flatten = x.flatten()
    timestep_indices = ((1 - x_flatten) * (max_spike_time-1)).long()
    out = torch.zeros_like(x_flatten).unsqueeze(0).expand(max_spike_time, -1).clone()
    raw_indices = torch.arange(len(x_flatten)).expand_as(out).clone()
    out[timestep_indices, raw_indices] = 1
    # reshape
    out = out.view(max_spike_time, *shape)
    out = out.transpose(0, 1)
    return out
    # batch_size, dim = x.shape
    # out = torch.zeros([batch_size, max_delay_time + 1, dim], device=x.device)
    # batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1)
    # dim_indices = torch.arange(dim, device=x.device)
    # delay_indices = ((1 - x) * max_delay_time).long()
    # out[batch_indices, delay_indices, dim_indices] = 1
    # return out


def _ttfs_decode_test():
    """
    A test function for TTFS encoding.Output y should be same as x.
    """
    x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    x = x / 20
    y = ttfs_encode(x, 100)
    y = y.view(2, 100, -1)
    print(y)


def s4nn_encode_label(output: torch.Tensor, label: torch.Tensor, max_spike_time: int, delay_timestep: int):
    """
    Transform label[batch_size] into targe spike time[batch_size,num_channel].
    The precise spike time is determined in S4NN way.
    Generally, if the prediction is correct, the target time of correct channel is the spike time
    and the target time of other channels is the spike time plus delay.
    if the prediction is wrong, the target time of correct channel is the earliest spike time
    and the target time of other channels is the latest spike time.
    Args:
        output:The network's direct shape is [batch_size,num_channel].
        label:Dataset Label shape is [batch_size].
        max_spike_time: Maximum spike time.
        delay_timestep: Number of delay step for incorrect target.
    Returns:
        target_time: target spike time for each channel shape is [batch_size,num_channel].
    """
    with torch.no_grad():
        # timestep of first spike in each channel shape is [batch_size]
        min_time: torch.Tensor = output.min(dim=1)[0]
        # incase timestep beyond max spike time or all the channel has no spike.
        min_time = torch.where(min_time >= torch.tensor(max_spike_time), min_time - delay_timestep, min_time)
        # shape to [batch_size,num_channel]
        delayed_time = (delay_timestep + min_time).unsqueeze(1).expand_as(output).clone()
        target_time = torch.where(output < delayed_time, delayed_time, output)
        target_indices = [torch.arange(target_time.shape[0]), label]
        target_time[target_indices] = min_time
    return target_time


def encode_poisson(x: torch.Tensor, max_spike_time: int):
    """
    Method encodes single value data into poisson spikes train.
    Args:
        x: Input data, shape is [batch_size,data_len]
        max_spike_time:The maximum spike time
    Returns:
        out: Poisson spikes train
    """
    assert len(x.shape) == 2
    batch_size = x.shape[0]
    data_len = x.shape[1]
    x = x.repeat(1, max_spike_time).reshape(batch_size, max_spike_time, data_len)
    x_normalized = torch.zeros_like(x)
    for i in range(x.shape[0]):
        x_normalized[i] = F.normalize(x[i])
    out_spike = torch.rand_like(x_normalized).le(x_normalized).to(x_normalized)
    return out_spike


if __name__ == "__main__":
    pass
