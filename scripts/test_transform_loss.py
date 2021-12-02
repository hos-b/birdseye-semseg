import os
import torch
import torch.nn as nn
import torch.optim

# just for random seed
import random
import numpy as np

import wandb

import matplotlib
matplotlib.use('Agg')

from agent.agent_pool import CurriculumPool
from data.config import SemanticCloudConfig, TrainingConfig
import data.color_map as color_map
from data.dataset import MassHDF5
from data.logging import init_wandb
from data.utils import get_noisy_transforms, get_se2_noise_transforms
from data.utils import to_device, get_transform_loss
from model.factory import get_model
from evaluate import plot_full_batch

def test(**kwargs):

    train_cfg: TrainingConfig = kwargs.get('train_cfg')
    NEW_SIZE, CENTER, PPM = kwargs.get('geom_properties')
    # network & cuda
    device = kwargs.get('device')
    model = kwargs.get('model')
    agent_pool: CurriculumPool = kwargs.get('agent_pool')
    # losses & optimization
    optimizer: torch.optim.Adam = kwargs.get('optimizer')
    mask_loss: nn.L1Loss = kwargs.get('mask_loss')
    transform_loss: nn.MSELoss = kwargs.get('transform_loss')
    semseg_loss = kwargs.get('semseg_loss')
    mask_loss_weight = kwargs.get('mask_loss_weight')
    sseg_loss_weight = kwargs.get('sseg_loss_weight')
    trns_loss_weight = kwargs.get('trns_loss_weight')
    # dataset
    train_set: MassHDF5 = kwargs.get('train_set')
    segmentation_classes = kwargs.get('segmentation_classes')
    # logging
    enable_logging = False
    if enable_logging:
        init_wandb('transform loss test', train_cfg)

    # start
    batch_ids = [0, 1, 2, 3, 4, 5] #, 1, 2, 3, 4, 5
    for ep in range(10000):
        total_t_loss = 0.0
        sample_count = 0
        model.train()
        for batch_id in batch_ids:
            rgbs, labels, car_masks, fov_masks, car_transforms, _ = train_set.__getitem__(batch_id)
            batch_size = rgbs.shape[0]
            sample_count += batch_size
            rgbs, labels, car_masks, fov_masks, car_transforms = to_device(device, rgbs, labels, car_masks,
                                                        fov_masks, car_transforms)
            solo_masks = car_masks + fov_masks
            agent_pool.generate_connection_strategy(solo_masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            # fwd-bwd
            optimizer.zero_grad()
            # add se2 noise
            noisy_transforms = get_noisy_transforms(car_transforms,
                                                    train_cfg.se2_noise_dx_std,
                                                    train_cfg.se2_noise_dy_std,
                                                    train_cfg.se2_noise_th_std)
            # forward pass
            solo_sseg_preds, solo_mask_preds, aggr_sseg_preds, aggr_mask_preds = \
                model(rgbs, noisy_transforms, agent_pool.adjacency_matrix, car_masks)
            t_loss = get_transform_loss(car_transforms, noisy_transforms,
                                        model.feat_matching_net.estimated_noise,
                                        agent_pool.adjacency_matrix, transform_loss)
            # weighted losses
            # (0.5 * m_loss * torch.exp(-mask_loss_weight) + 0.5 * mask_loss_weight +
            #  0.5 * t_loss * torch.exp(-trns_loss_weight) + 0.5 * trns_loss_weight +
            #        s_loss * torch.exp(-sseg_loss_weight) + 0.5 * sseg_loss_weight).backward()
            (t_loss).backward()

            optimizer.step()
            total_t_loss += t_loss.item()
            # end of batch

        # log batch loss
        if ep % train_cfg.log_every == 0:
            print(f'\nepoch loss: {ep}: {(total_t_loss / sample_count):.6f} transform')
            if enable_logging:
                batch_img = plot_full_batch(rgbs, labels, solo_sseg_preds, aggr_sseg_preds,
                                            solo_mask_preds, aggr_mask_preds,
                                            solo_masks, agent_pool.combined_masks,
                                            plot_dest='image', semantic_classes=train_cfg.classes,
                                            title=f'E: {ep + 1}, B#: idk')

if __name__ == '__main__':
    # parsing config file
    geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    train_cfg = TrainingConfig('config/training.yml')
    if not train_cfg.se2_noise_enable:
        print('se2 noise is disabled'); exit()
    device = torch.device('cuda')
    torch.manual_seed(train_cfg.torch_seed)
    random.seed(train_cfg.torch_seed)
    np.random.seed(train_cfg.torch_seed)
    # image size and center coordinates --------------------------------------------------------
    new_size = (train_cfg.output_h, train_cfg.output_w)
    center = (geom_cfg.center_x(new_size[1]), geom_cfg.center_y(new_size[0]))
    ppm = geom_cfg.pix_per_m(new_size[0], new_size[1])
    # dataset ----------------------------------------------------------------------------------
    train_set = MassHDF5(dataset=train_cfg.validset_name, path=train_cfg.dset_dir,
                         hdf5name=train_cfg.validset_file, size=new_size,
                         classes=train_cfg.classes, jitter=[0, 0, 0, 0],
                         mask_gaussian_sigma=train_cfg.gaussian_mask_std,
                         guassian_kernel_size=train_cfg.gaussian_kernel_size)
    segmentation_classes = color_map.__our_classes
    # snapshot dir -----------------------------------------------------------------------------
    train_cfg.snapshot_dir = train_cfg.snapshot_dir.format(train_cfg.training_name)
    if not os.path.exists(train_cfg.snapshot_dir):
        os.makedirs(train_cfg.snapshot_dir)
    # network stuff ----------------------------------------------------------------------------
    model = get_model('mcnnT3xNoisyRT', train_cfg.num_classes, new_size,
                      geom_cfg, train_cfg.aggregation_type,
                      mcnnt3x_path=train_cfg.extra_model_arg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    mask_loss_weight = torch.tensor([0.0], requires_grad=True, device=device)
    sseg_loss_weight = torch.tensor([0.0], requires_grad=True, device=device)
    trns_loss_weight = torch.tensor([0.0], requires_grad=True, device=device)
    optimizer.add_param_group({"params": [mask_loss_weight, sseg_loss_weight, trns_loss_weight]})
    agent_pool = CurriculumPool(train_cfg.initial_difficulty, train_cfg.maximum_difficulty,
                                train_cfg.max_agent_count, train_cfg.enforce_adj_calc, device)
    # losses -----------------------------------------------------------------------------------
    semseg_loss = nn.CrossEntropyLoss(reduction='none')
    mask_loss = nn.L1Loss(reduction='mean')
    transform_loss = nn.MSELoss(reduction='none')
    # send to gpu
    semseg_loss = semseg_loss.to(device)
    mask_loss = mask_loss.to(device)
    transform_loss = transform_loss.to(device)
    # begin ------------------------------------------------------------------------------------
    print('starting transform loss test')
    test(train_cfg=train_cfg, device=device, model=model, optimizer=optimizer,
         agent_pool=agent_pool, mask_loss=mask_loss, semseg_loss=semseg_loss,
         transform_loss=transform_loss, geom_properties=(new_size, center, ppm),
         mask_loss_weight=mask_loss_weight, sseg_loss_weight=sseg_loss_weight,
         trns_loss_weight=trns_loss_weight, train_set=train_set,
         segmentation_classes=segmentation_classes)