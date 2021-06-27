import cv2
from numpy.core.fromnumeric import size
import torch
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

def drop_agent_data(rgbs, labels, masks, transforms, drop_probability) -> Tuple[torch.Tensor]:
    """
    simulate connection drops between cars or non transmitting cars
    input:
        - rgbs:         batch_size x agent_count x 3 x H x W
        - labels:       batch_size x agent_count x H x W
        - masks:        batch_size x agent_count x H x W
        - transforms:   batch_size x agent_count x 16 x 16
    """
    # don't drop for single batches
    if rgbs.shape[1] == 1:
        return rgbs[0, ...], labels[0, ...], masks[0, ...], transforms[0, ...]
    drop_probs = torch.ones((rgbs.shape[1], ), dtype=torch.float32) * drop_probability
    drops = torch.bernoulli(drop_probs).long()
    # if randomed all ones (everything dropped), return everything
    if drops.sum() == rgbs.shape[1]:
        return rgbs[0, ...], labels[0, ...], masks[0, ...], transforms[0, ...]
    return rgbs[0, drops != 1, ...], labels[0, drops != 1, ...], \
           masks[0, drops != 1, ...], transforms[0, drops != 1, ...]

def squeeze_all(rgbs, labels, masks, transforms) -> Tuple[torch.Tensor]:
    """
    squeezes all given parameters
    """
    return rgbs.squeeze(0), labels.squeeze(0), masks.squeeze(0), transforms.squeeze(0)

def get_matplotlib_image(tensor_img: torch.Tensor, figsize=(4, 5)) -> np.ndarray:
    """
    returns the plot of a mask as an image
    """
    org_h, org_w = tensor_img.shape
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    plt.imshow(tensor_img)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    image = cv2.resize(image, dsize=(org_w, org_h), interpolation=cv2.INTER_NEAREST)
    fig.clear()
    plt.close(fig)
    return image

def to_device(rgbs, labels, masks, car_transforms, device) -> Tuple[torch.Tensor]:
    """
    sends the tensors to the given device
    """
    return rgbs.to(device), labels.to(device), \
           masks.to(device), car_transforms.to(device)

def get_noisy_transforms(transforms: torch.Tensor, dx_std, dy_std, th_std) -> torch.Tensor:
    """
    return a noisy version of the transforms given the noise parameters
    """
    batch_size = transforms.shape[0]
    se3_noise = torch.zeros_like(transforms)
    if th_std != 0.0:
        rand_t = torch.normal(mean=0.0, std=th_std, size=(batch_size,)) * (np.pi / 180.0)
        se3_noise[:, 0, 0] = torch.cos(rand_t)
        se3_noise[:, 0, 1] = -torch.sin(rand_t)
        se3_noise[:, 1, 0] = torch.sin(rand_t)
        se3_noise[:, 1, 1] = torch.cos(rand_t)
    else:
        se3_noise[:, 0, 0] = 1
        se3_noise[:, 1, 1] = 1
    if dx_std != 0.0:
        se3_noise[:, 0, 3] = torch.normal(mean=0.0, std=dx_std, size=(batch_size,))
    if dy_std != 0.0:
        se3_noise[:, 1, 3] = torch.normal(mean=0.0, std=dy_std, size=(batch_size,))
    se3_noise[:, 2, 2] = 1
    se3_noise[:, 3, 3] = 1
    return transforms @ se3_noise

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