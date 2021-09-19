import gc
import torch
import numpy as np
from typing import Tuple

def drop_agent_data(drop_probability, *args) -> Tuple[torch.Tensor]:
    """
    simulate connection drops between cars or non-transmitting cars
    input:
        - rgbs:         1 x agent_count x 3 x H x W
        - labels:       1 x agent_count x H x W
        - masks:        1 x agent_count x H x W
        - transforms:   1 x agent_count x 16 x 16
    """
    bsize = args[0].shape[1]
    # don't drop for single batches
    if bsize == 1:
        return (arg[0, ...] for arg in args)
    drop_probs = torch.ones((bsize, ), dtype=torch.float32) * drop_probability
    drops = torch.bernoulli(drop_probs).long()
    # if randomed all ones (everything dropped), return everything
    if drops.sum() == bsize:
        return (arg[0, ...] for arg in args)
    return (arg[0, drops != 1, ...] for arg in args)

def squeeze_all(*args) -> Tuple[torch.Tensor]:
    """
    squeezes all given parameters
    """
    return (arg.squeeze(0) for arg in args)

def to_device(device, *args) -> Tuple[torch.Tensor]:
    """
    sends the tensors to the given device
    """
    return (arg.to(device) for arg in args)

def get_noisy_transforms(transforms: torch.Tensor, dx_std, dy_std, th_std) -> torch.Tensor:
    """
    return a noisy version of the transforms given the noise parameters
    """
    batch_size = transforms.shape[0]
    se2_noise = torch.zeros_like(transforms)
    if th_std != 0.0:
        rand_t = torch.normal(mean=0.0, std=th_std, size=(batch_size,)) * (np.pi / 180.0)
        se2_noise[:, 0, 0] = torch.cos(rand_t)
        se2_noise[:, 0, 1] = -torch.sin(rand_t)
        se2_noise[:, 1, 0] = torch.sin(rand_t)
        se2_noise[:, 1, 1] = torch.cos(rand_t)
    else:
        se2_noise[:, 0, 0] = 1
        se2_noise[:, 1, 1] = 1
    if dx_std != 0.0:
        se2_noise[:, 0, 3] = torch.normal(mean=0.0, std=dx_std, size=(batch_size,))
    if dy_std != 0.0:
        se2_noise[:, 1, 3] = torch.normal(mean=0.0, std=dy_std, size=(batch_size,))
    se2_noise[:, 2, 2] = 1
    se2_noise[:, 3, 3] = 1
    return transforms @ se2_noise

def separate_masks(masks: torch.Tensor, boundary_pixel: int = 172):
    """
    seperates the mask into vehicle and FoV masks.
    if there is a vehicle right in front of the current one, the boundary
    pixel is violated. less that 5% ? who cares. masks size: Bx256x205
    """
    vehicle_masks = masks.clone()
    vehicle_masks[:, :boundary_pixel] = 0
    masks[:, boundary_pixel:] = 0
    return vehicle_masks, masks

def mem_report(source: str='cpu'):
    '''
    Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported
    https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe
    source can be 'cpu', 'gpu' or 'all'
    '''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print(f'Storage on {mem_type}')
        print('-' * LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print(f'{element_type}\t\t{size}\t\t{mem:.2f}')
        print('-' * LEN)
        print(f'Total Tensors: {total_numel} \tUsed Memory Space: {total_mem:.2f} MBytes')
        print('-' * LEN)

    LEN = 65
    print('=' * LEN)
    objects = gc.get_objects()
    print('Element type\tSize\t\t\tUsed MEM(MBytes)')
    tensors = [obj for obj in objects if isinstance(obj, torch.Tensor)]
    source = source.lower()
    if source == 'cpu':
        host_tensors = [t for t in tensors if not t.is_cuda]    
        _mem_report(host_tensors, 'CPU')
    elif source == 'gpu':
        cuda_tensors = [t for t in tensors if t.is_cuda]
        _mem_report(cuda_tensors, 'GPU')
    elif source == 'all':
        host_tensors = [t for t in tensors if not t.is_cuda]
        cuda_tensors = [t for t in tensors if t.is_cuda]
        _mem_report(host_tensors, 'CPU')
        _mem_report(cuda_tensors, 'GPU')
    else:
        raise ValueError(f'unknown source: {source}')
    print('=' * LEN)

# dicts for plotting batches based on agent count
newline_dict = {
    1: '',
    2: '',
    3: '',
    4: '',
    5: '\n',
    6: '\n\n',
    7: '\n',
    8: '\n'
}

font_dict = {
    1: 17,
    2: 25,
    3: 30,
    4: 32,
    5: 37,
    6: 40,
    7: 45,
    8: 45
}