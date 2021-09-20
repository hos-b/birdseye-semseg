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

def get_relative_noise(gt_transforms: torch.Tensor, noisy_transforms: torch.Tensor) -> torch.Tensor:
    """
    return the relative noise of the noisy transforms compared to the gt transforms
    """
    agent_count = gt_transforms.shape[0]
    relative_noise_params = torch.zeros((agent_count, agent_count, 3),
                                        dtype=gt_transforms.dtype,
                                        device=gt_transforms.device)
    # extracting relative 2D gt transforms
    gt_rel_3d_tf = gt_transforms.repeat(agent_count, 1, 1)
    gt_rel_3d_tf = gt_transforms.inverse().repeat_interleave(agent_count, dim=0) @ gt_rel_3d_tf
    gt_rel_2d_tf = torch.eye(3).unsqueeze(0).repeat(agent_count * agent_count, 1, 1)
    gt_rel_2d_tf[:, :2,  2] = gt_rel_3d_tf[:, :2,  3]
    gt_rel_2d_tf[:, :2, :2] = gt_rel_3d_tf[:, :2, :2]
    gt_rel_2d_tf = gt_rel_2d_tf.reshape(agent_count, agent_count, 3, 3)
    # extracting relative 2D noisy transforms
    nz_rel_3d_tf = noisy_transforms.repeat(agent_count, 1, 1)
    nz_rel_3d_tf = noisy_transforms.inverse().repeat_interleave(agent_count, dim=0) @ nz_rel_3d_tf
    nz_rel_2d_tf = torch.eye(3).unsqueeze(0).repeat(agent_count * agent_count, 1, 1)
    nz_rel_2d_tf[:, :2,  2] = nz_rel_3d_tf[:, :2,  3]
    nz_rel_2d_tf[:, :2, :2] = nz_rel_3d_tf[:, :2, :2]
    nz_rel_2d_tf = nz_rel_2d_tf.reshape(agent_count, agent_count, 3, 3)
    # calculating noise parameters
    relative_noise_params[:, :, :2] = gt_rel_2d_tf[:, :, :2, 2] - nz_rel_2d_tf[:, :, :2, 2]
    relative_noise_params[:, :,  2] = torch.atan2(gt_rel_2d_tf[:, :, 1, 0], gt_rel_2d_tf[:, :, 0, 0]) - \
                                      torch.atan2(nz_rel_2d_tf[:, :, 1, 0], nz_rel_2d_tf[:, :, 0, 0])
    return relative_noise_params

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