import torch
import numpy as np
from typing import Tuple
from data.mask_warp import get_single_relative_img_transform

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

def get_se2_noise_transforms(batch_size, device, dx_std, dy_std, th_std) -> torch.Tensor:
    """
    returns a batch_size x batch_size x 4 x 4 tensor of SE2 noise
    the ego-transforms are the identity.
    """
    bs_sq = batch_size * batch_size
    se2_noise = torch.zeros((bs_sq, 4, 4), dtype=torch.float32, device=device)
    if th_std != 0.0:
        rand_t = torch.normal(mean=0.0, std=th_std, size=(bs_sq,)) * (np.pi / 180.0)
        se2_noise[:, 0, 0] = torch.cos(rand_t)
        se2_noise[:, 0, 1] = -torch.sin(rand_t)
        se2_noise[:, 1, 0] = torch.sin(rand_t)
        se2_noise[:, 1, 1] = torch.cos(rand_t)
    else:
        se2_noise[:, 0, 0] = 1
        se2_noise[:, 1, 1] = 1
    if dx_std != 0.0:
        se2_noise[:, 0, 3] = torch.normal(mean=0.0, std=dx_std, size=(bs_sq,))
    if dy_std != 0.0:
        se2_noise[:, 1, 3] = torch.normal(mean=0.0, std=dy_std, size=(bs_sq,))
    se2_noise[:, 2, 2] = 1
    se2_noise[:, 3, 3] = 1
    # set the diagonal to eye
    indices = [i + (i * batch_size) for i in range(batch_size)]
    se2_noise[indices] = torch.eye(4, dtype=torch.float32, device=device)
    return se2_noise.reshape((batch_size, batch_size, 4, 4))

def get_se2_diff(mat_1: torch.Tensor, mat_2: torch.Tensor):
    """
    input:
        * matrix 1: A x 4 x 4
        * matrix 2: A x 4 x 4
    output:
        * A x 3 [dx, dy, dtheta]
    """
    agent_count = mat_1.shape[0]
    relative_diff = torch.zeros(
        (agent_count, 3), dtype=mat_1.dtype,
        device=mat_1.device
    )
    relative_diff[:, :2] = mat_1[:, :2, 3] - mat_2[:, :2, 3]
    relative_diff[:,  2] = torch.atan2(mat_1[:, 1, 0], mat_1[:, 0, 0]) - \
                           torch.atan2(mat_2[:, 1, 0], mat_2[:, 0, 0])
    # limit angle range to -pi, pi
    for a in range(agent_count):
        while relative_diff[a, 2] <= np.pi:
            relative_diff[a, 2] += 2 * np.pi
        while relative_diff[a, 2] > np.pi:
            relative_diff[a, 2] -= 2 * np.pi
    return relative_diff

def get_relative_noise(gt_transforms: torch.Tensor, noisy_transforms: torch.Tensor, agent_index: int) -> torch.Tensor:
    """
    return the relative noise of the noisy transforms compared to the gt transforms for
    the given agent index
    input:
        * gt_transforms:    A x 4 x 4
        * noisy_transforms: A x 4 x 4
    output:
        * relative_noise:   A x 3 [x, y, theta]
    """
    gt_relative_tfs = gt_transforms[agent_index].inverse() @ gt_transforms
    # T_j' w.r.t. T_i
    nz_relative_tfs = noisy_transforms[agent_index].inverse() @ noisy_transforms

    return get_se2_diff(gt_relative_tfs, nz_relative_tfs)

def separate_masks(masks: torch.Tensor, boundary_pixel: int = 172):
    """
    seperates the mask into vehicle and FoV masks.
    if there is a vehicle right in front of the current one, the boundary
    pixel is violated. less than 5% ? who cares. masks size: Bx256x205
    """
    vehicle_masks = masks.clone()
    vehicle_masks[:, :boundary_pixel] = 0
    masks[:, boundary_pixel:] = 0
    return vehicle_masks, masks

def get_transform_loss(gt_transforms: torch.Tensor, noisy_transforms: torch.Tensor,
                       estiamted_noise: torch.Tensor, loss_func, agent_count):
    """
    calculates the loss of the noisy transforms compared to the gt transforms
    """
    t_loss = 0
    for i in range(agent_count):
        gt_relative_tfs = gt_transforms[i].inverse() @ gt_transforms
        estimated_relative_tfs = (noisy_transforms[i].inverse() @ noisy_transforms) @ estiamted_noise[i]
        t_loss += loss_func(estimated_relative_tfs, gt_relative_tfs)
    return t_loss

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