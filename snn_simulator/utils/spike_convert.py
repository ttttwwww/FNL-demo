import torch


def spike_train_to_spike_time(spike_train: torch.Tensor)->torch.Tensor:
    """
    Input spike train [B,T,C].Extract the spike time[B,C,N]
    Value in N dim is the index+1 in T dim,which means spike at [B,0,C]->spike_train[B,C,0]=1

    Extract spike timestep from spike train.The channel doesn't spike will have torch.inf as spike time.
    To accelerate computations, tensor operations are used to replace for loops.
    Particularly, the advanced indexing operations used here might cause difficulty understanding.
    Args:
        spike_train: Input spike train shape is [B,T,C]

    Returns:
        spike_times(torch.Tensor): Extracted spike times shape is [batch_size,num_channels,max_num_spike]
        max_num_spike denotes the maximum number of spike of all channel
    """
    spike_train = torch.where(spike_train>0,1,0)
    batch_size, time_step, num = spike_train.shape
    spike_train = spike_train.transpose(1,2)
    max_num_spike = max(1, int(torch.max(spike_train.sum(2)).item()))
    # make every channel spike at max_timestep
    spike_time = torch.full([batch_size, num, max_num_spike],torch.inf, dtype=torch.float32).to(spike_train.device)

    # get spike indices
    indices = torch.where(spike_train > 0)
    batch_indices, num_channels, time_indices = indices

    # calculate spike number of each channel
    unique_pairs, inverse_indices = torch.unique(torch.stack([batch_indices, num_channels], dim=1), dim=0,
                                                 return_inverse=True)
    num_spike = torch.zeros(len(unique_pairs), dtype=spike_train.dtype).to(spike_train.device)

    num_spike.scatter_add_(0, inverse_indices,spike_train[indices])

    # fill time indices into the array.
    # advanced indexing is used instead of for loop.
    offset = torch.arange(0, max_num_spike, device=spike_train.device).expand(len(unique_pairs), max_num_spike)
    mask = offset < num_spike.unsqueeze(-1)
    neuron_spike_time = torch.full([len(unique_pairs), max_num_spike],torch.inf, dtype=torch.float32).to(
        spike_train.device)
    neuron_spike_time[mask] = time_indices.float()
    batch_ids,neuron_ids = unique_pairs.unbind(1)
    spike_time[batch_ids, neuron_ids] = neuron_spike_time + 1
    return spike_time

def spike_time_to_spike_train(spike_time: torch.Tensor,max_time_step:int)->torch.Tensor:
    """
    Create spike train from spike time.
    Args:
        spike_time: Spike time of each channel.Shape is [batch_size,num_channels,max_num_spike]
        max_time_step: Maximum timestep in spike_train.

    Returns:
        spike_train: Spike train shape is [batch_size,timestep,num_channels]

    """
    batch_size, num_channels,max_num_spike = spike_time.shape
    spike_train = torch.zeros([batch_size, max_time_step, num_channels], dtype=torch.float32, device=spike_time.device)
    indices = torch.where(spike_time != torch.inf)
    batch_indices, num_channels,spike_indices = indices
    spike_train[batch_indices,spike_time[indices].long()-1,  num_channels] = 1
    return spike_train

if __name__ == '__main__':
    a = torch.tensor([[0,0,0,0],[0,1,0,1],[0,1,1,0],[1,0,0,1]],dtype=torch.float32).unsqueeze(0).transpose(2,1)
    spike_time = spike_train_to_spike_time(a)
    print(spike_time)
    b = spike_time_to_spike_train(spike_time,4)
    if torch.equal(a,b):
        print(torch.equal(a,b))
    else:
        print("a=")
        print(a)
        print("b=")
        print(b)