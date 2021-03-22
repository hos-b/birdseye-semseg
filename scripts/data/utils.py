import cv2
import torch
import wandb
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data.config import TrainingConfig

def drop_agent_data(rgbs, labels, masks, transforms, drop_probability):
    """
    simulate connection drops between cars or non transmitting cars
    input:
        - rgbs:         batch_size x agent_count x 3 x H x W
        - labels:       batch_size x agent_count x H x W
        - masks:        batch_size x agent_count x H x W
        - transforms:   batch_size x agent_count x 16 x 16
    """
    drop_probs = torch.ones((rgbs.shape[1], ), dtype=torch.float32) * drop_probability
    drops = torch.bernoulli(drop_probs).long()
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

def to_device(rgbs, labels, masks, car_transforms, device, non_blocking):
    """
    sends the tensors to the given device
    """
    return rgbs.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking), \
           masks.to(device, non_blocking=non_blocking), car_transforms.to(device, non_blocking=non_blocking)

def init_wandb(name: str, train_cfg: TrainingConfig):
    wandb.init(
        project='birdseye-semseg',
        entity='hos-b',
        group='initial models',
        name=name,
        dir=train_cfg.log_dir,
        config={
            'model_size': train_cfg.model_size,
            'segmentation_loss': train_cfg.loss_function,
            'epochs': 1000,
            'learning_rate': train_cfg.learning_rate,
            'output_h': train_cfg.output_h,
            'output_w': train_cfg.output_w,
            'classes': train_cfg.classes,
            'mask_det_threshold': train_cfg.mask_detection_thresh,
            'agent_drop_probability': train_cfg.drop_prob,
            'initial_difficulty': train_cfg.initial_difficulty,
            'maximum_difficulty': train_cfg.maximum_difficulty,
            'strategy': train_cfg.strategy,
            'strategy_parameter': train_cfg.strategy_parameter,
            'world_size': train_cfg.world_size,
            'torch_seed': train_cfg.torch_seed
    })
