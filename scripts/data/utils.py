import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

def drop_agent_data(rgbs, labels, masks, transforms, drop_probability):
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

def squeeze_all(rgbs, labels, masks, transforms):
    """
    squeezes all given parameters
    """
    return rgbs.squeeze(0), labels.squeeze(0), masks.squeeze(0), transforms.squeeze(0)

def get_matplotlib_image(tensor_img: torch.Tensor, figsize=(4, 5)):
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

def to_device(rgbs, labels, masks, car_transforms, device):
    """
    sends the tensors to the given device
    """
    return rgbs.to(device), labels.to(device), \
           masks.to(device), car_transforms.to(device)

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