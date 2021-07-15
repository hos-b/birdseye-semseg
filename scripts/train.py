import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from kornia.losses.focal import FocalLoss

# just for random seed
import random
import numpy as np

import wandb
import subprocess

import matplotlib
matplotlib.use('Agg')

from agent.agent_pool import CurriculumPool
from data.config import SemanticCloudConfig, TrainingConfig
import data.color_map as color_map
from data.dataset import MassHDF5
from data.logging import init_wandb
from metrics.iou import get_iou_per_class, get_mask_iou
from data.utils import drop_agent_data, squeeze_all
from data.utils import get_noisy_transforms
from data.utils import to_device
from model.large_mcnn import TransposedMCNN, MaxoutMCNNT
from model.noisy_mcnn import NoisyMCNN
from model.pyrocc.pyrocc import PyramidOccupancyNetwork
from evaluate import plot_full_batch

def train(**kwargs):
    train_cfg: TrainingConfig = kwargs.get('train_cfg')
    NEW_SIZE, CENTER, PPM = kwargs.get('geom_properties')
    log_enable = kwargs.get('log_enable')
    # network & cuda
    device = kwargs.get('device')
    model = kwargs.get('model')
    agent_pool: CurriculumPool = kwargs.get('agent_pool')
    # losses & optimization
    scheduler: lr_scheduler.LambdaLR = kwargs.get('scheduler')
    optimizer: torch.optim.Adam = kwargs.get('optimizer')
    mask_loss: nn.L1Loss = kwargs.get('mask_loss')
    semseg_loss = kwargs.get('semseg_loss')
    mask_loss_weight = kwargs.get('mask_loss_weight')
    sseg_loss_weight = kwargs.get('sseg_loss_weight')
    last_snapshot_metric = 1e6
    # dataset
    train_loader = kwargs.get('train_loader')
    valid_loader = kwargs.get('valid_loader')
    segmentation_classes = kwargs.get('segmentation_classes')
    epochs = train_cfg.epochs
    # starting epoch
    start_ep = kwargs.get('start_ep')
    for ep in range(start_ep, epochs):
        total_train_m_loss = 0.0
        total_train_s_loss = 0.0
        sample_count = 0
        # training
        model.train()
        for batch_idx, (rgbs, labels, masks, car_transforms, _) in enumerate(train_loader):
            sample_count += rgbs.shape[1]
            rgbs, labels, masks, car_transforms = to_device(rgbs, labels, masks,
                                                            car_transforms, device)
            # simulate connection drops
            rgbs, labels, masks, car_transforms = drop_agent_data(rgbs, labels,
                                                                  masks, car_transforms,
                                                                  train_cfg.drop_prob)
            agent_pool.generate_connection_strategy(masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            # fwd-bwd
            optimizer.zero_grad()
            # add se2 noise
            if train_cfg.se2_noise_enable:
                car_transforms = get_noisy_transforms(car_transforms,
                                                      train_cfg.se2_noise_dx_std,
                                                      train_cfg.se2_noise_dy_std,
                                                      train_cfg.se2_noise_th_std)
            sseg_preds, mask_preds = model(rgbs, car_transforms, agent_pool.adjacency_matrix)
            m_loss = mask_loss(mask_preds.squeeze(1), masks)
            s_loss = torch.mean(semseg_loss(sseg_preds, labels) * agent_pool.combined_masks,
                                dim=(0, 1, 2))
            # semseg & mask batch loss
            batch_train_m_loss = m_loss.item()
            batch_train_s_loss = s_loss.item()
            # weighted losses
            (m_loss * torch.exp(-mask_loss_weight) + mask_loss_weight +
             s_loss * torch.exp(-sseg_loss_weight) + sseg_loss_weight).backward()

            optimizer.step()

            # log batch loss
            if (batch_idx + 1) % train_cfg.log_every == 0:
                if log_enable:
                    wandb.log({
                        'loss/batch train mask': batch_train_m_loss,
                        'loss/batch train sseg': batch_train_s_loss
                    })
                print(f'\repoch: {ep + 1}/{epochs}, '
                      f'training batch: {batch_idx + 1} / {len(train_loader)}', end='')
            total_train_m_loss += batch_train_m_loss
            total_train_s_loss += batch_train_s_loss
            # end of batch

        # log train epoch loss
        if log_enable:
            wandb.log({
                'loss/total train mask': total_train_m_loss / sample_count,
                'loss/total train sseg': total_train_s_loss / sample_count,
                'misc/epoch': ep + 1
            })
        print(f'\nepoch loss: {(total_train_m_loss / sample_count)} mask, '
              f'{(total_train_s_loss / sample_count)} segmentation')
        # validation ---------------------------------------------------------------------------
        model.eval()
        visualized = False
        total_valid_m_loss = 0.0
        total_valid_s_loss = 0.0
        sseg_ious = torch.zeros((train_cfg.num_classes, 1), dtype=torch.float64).to(device)
        mask_ious = 0.0
        sample_count = 0
        for batch_idx, (rgbs, labels, masks, car_transforms, batch_no) in enumerate(valid_loader):
            print(f'\repoch: {ep + 1}/{epochs}, '
                  f'validation batch: {batch_idx + 1} / {len(valid_loader)}', end='')
            sample_count += rgbs.shape[1]
            rgbs, labels, masks, car_transforms = squeeze_all(rgbs, labels, masks, car_transforms)
            rgbs, labels, masks, car_transforms = to_device(rgbs, labels, masks,
                                                            car_transforms, device)
            agent_pool.generate_connection_strategy(masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            # add se2 noise to transforms
            if train_cfg.se2_noise_enable:
                car_transforms = get_noisy_transforms(car_transforms,
                                                      train_cfg.se2_noise_dx_std,
                                                      train_cfg.se2_noise_dy_std,
                                                      train_cfg.se2_noise_th_std)
            with torch.no_grad():
                sseg_preds, mask_preds = model(rgbs, car_transforms, agent_pool.adjacency_matrix)
            sseg_ious += get_iou_per_class(sseg_preds, labels, agent_pool.combined_masks,
                                       train_cfg.num_classes).to(device)
            mask_ious += get_mask_iou(mask_preds.squeeze(1), masks, train_cfg.mask_detection_thresh)
            # sum up losses
            total_valid_m_loss += mask_loss(mask_preds.squeeze(1), masks).item()
            total_valid_s_loss += torch.mean(semseg_loss(sseg_preds, labels) * agent_pool.combined_masks,
                                             dim=(0, 1, 2)).item()
            # visualize a random batch and all hard batches [if enabled]
            if not visualized and log_enable:
                validation_img_log_dict = {'misc/epoch': ep + 1}
                first_batch_img = plot_full_batch(rgbs, labels, sseg_preds, mask_preds, masks,
                                                  agent_pool, plot_dest='image',
                                                  semantic_classes=train_cfg.classes,
                                                  title=f'E: {ep + 1}, B#: {batch_no.item()}')
                validation_img_log_dict['media/results'] = \
                    wandb.Image(first_batch_img, caption='full batch predictions')
                wandb.log(validation_img_log_dict)
                visualized = True
            # end of batch

        # more wandb logging -------------------------------------------------------------------
        elevation_metric = 0.0
        log_dict = {}
        for key, val in segmentation_classes.items():
            log_dict[f'iou/{val.lower()}'] = (sseg_ious[key] / sample_count).item()
            if val != 'Misc' and val != 'Water':
                elevation_metric += sseg_ious[key] / sample_count
        log_dict['loss/total validation mask'] = (total_valid_m_loss / sample_count)
        log_dict['loss/total validation sseg'] = (total_valid_s_loss / sample_count)
        log_dict['iou/mask'] = (mask_ious / sample_count).item()
        log_dict['misc/epoch'] = ep + 1
        log_dict['misc/save'] = 0
        log_dict['curriculum/elevation metric'] = (elevation_metric / 5).item()
        log_dict['weight/sseg'] = torch.exp(-sseg_loss_weight).item()
        log_dict['weight/mask'] = torch.exp(-mask_loss_weight).item()
        print(f'\nepoch validation loss: {total_valid_m_loss / sample_count} mask, '
              f'{total_valid_s_loss / sample_count} segmentation')
        # saving the new model -----------------------------------------------------------------
        snapshot_tag = 'last'
        if log_dict['curriculum/elevation metric'] > last_snapshot_metric:
            print(f'best model @ epoch {ep + 1}')
            last_snapshot_metric = log_dict['curriculum/elevation metric']
            snapshot_tag = 'best'
            log_dict['misc/save'] = 1
        torch.save(optimizer.state_dict(), train_cfg.snapshot_dir +
                    f'/{snapshot_tag}_optimizer')
        torch.save(model.state_dict(), train_cfg.snapshot_dir +
                    f'/{snapshot_tag}_model.pth')
        # update curriculum difficulty ---------------------------------------------------------
        scheduler.step()
        if train_cfg.curriculum_activate:
            increase_diff = False
            if train_cfg.strategy == 'every-x-epochs':
                if (ep + 1) % int(train_cfg.strategy_parameter) == 0:
                    increase_diff = True
            elif train_cfg.strategy == 'metric':
                # elevation metric = avg[avg[important class IoU] + avg[mask IoU]]
                elevation_metric /= 6
                if elevation_metric >= train_cfg.strategy_parameter:
                    increase_diff = True
            if increase_diff:
                agent_pool.difficulty = min(agent_pool.difficulty + 1,
                                            agent_pool.maximum_difficulty)
                print(f'\n=======>> difficulty increased to {agent_pool.difficulty} <<=======')
        log_dict['curriculum/difficulty'] = agent_pool.difficulty
        if log_enable:
            wandb.log(log_dict)
    # end
    if log_enable:
        wandb.finish()

def parse_and_execute():
    # parsing config file
    geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    train_cfg = TrainingConfig('config/training.yml')
    # gpu selection ----------------------------------------------------------------------------
    device_str = train_cfg.device
    if train_cfg.device == 'cuda':
        torch.cuda.set_device(0)
        device_str += f':{0}'
    device = torch.device(device_str)
    torch.manual_seed(train_cfg.torch_seed)
    random.seed(train_cfg.torch_seed)
    np.random.seed(train_cfg.torch_seed)
    # image size and center coordinates --------------------------------------------------------
    new_size = (train_cfg.output_h, train_cfg.output_w)
    center = (geom_cfg.center_x(new_size[1]), geom_cfg.center_y(new_size[0]))
    ppm = geom_cfg.pix_per_m(new_size[0], new_size[1])
    print(f'output image size: {new_size}, vehicle center {center}')
    # dataset ----------------------------------------------------------------------------------
    train_set = MassHDF5(dataset=train_cfg.trainset_name, path=train_cfg.dset_dir,
                         hdf5name=train_cfg.trainset_file, size=new_size,
                         classes=train_cfg.classes, jitter=train_cfg.color_jitter)
    valid_set = MassHDF5(dataset=train_cfg.validset_name, path=train_cfg.dset_dir,
                         hdf5name=train_cfg.validset_file, size=new_size,
                         classes=train_cfg.classes, jitter=[0, 0, 0, 0])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=train_cfg.shuffle_data,
                                               num_workers=train_cfg.loader_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1,
                                               shuffle=train_cfg.shuffle_data,
                                               num_workers=train_cfg.loader_workers)
    if train_cfg.classes == 'carla':
        segmentation_classes = color_map.__carla_classes
    elif train_cfg.classes == 'ours':
        segmentation_classes = color_map.__our_classes
    elif train_cfg.classes == 'diminished':
        segmentation_classes = color_map.__diminished_classes
    # snapshot dir -----------------------------------------------------------------------------
    train_cfg.snapshot_dir = train_cfg.snapshot_dir.format(train_cfg.training_name)
    if not os.path.exists(train_cfg.snapshot_dir):
        os.makedirs(train_cfg.snapshot_dir)
    # network stuff ----------------------------------------------------------------------------
    if train_cfg.model_name == 'mcnnT':
        model = TransposedMCNN(train_cfg.num_classes, new_size,
                    geom_cfg, train_cfg.aggregation_type).to(device)
    elif train_cfg.model_name == 'mcnnTMax':
        model = MaxoutMCNNT(train_cfg.num_classes, new_size,
                    geom_cfg, train_cfg.aggregation_type).to(device)
    elif train_cfg.model_name == 'mcnnNoisy':
        model = NoisyMCNN(train_cfg.num_classes, new_size,
                    geom_cfg, train_cfg.aggregation_type).to(device)
    elif train_cfg.model_name == 'pyrocc':
        model = PyramidOccupancyNetwork(train_cfg.num_classes, new_size,
                    geom_cfg, train_cfg.aggregation_type).to(device)
    else:
        print('unknown network architecture {train_cfg.model_name}')
        exit()
    print(f'{(model.parameter_count() / 1e6):.2f}M trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    start_ep = train_cfg.resume_starting_epoch if train_cfg.resume_training else 0
    lr_lambda = lambda epoch: pow((1 - (((epoch + start_ep) - 1) / train_cfg.epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    mask_loss_weight = torch.tensor([0.0], requires_grad=True, device=device)
    sseg_loss_weight = torch.tensor([0.0], requires_grad=True, device=device)
    optimizer.add_param_group({"params": [mask_loss_weight, sseg_loss_weight]})
    agent_pool = CurriculumPool(train_cfg.initial_difficulty, train_cfg.maximum_difficulty,
                                train_cfg.max_agent_count, device)
    # loading the network parameters/optimizer state -------------------------------------------
    resume_tag = ''
    if train_cfg.resume_training:
        resume_tag = train_cfg.resume_tag + '-'
        snapshot_path = train_cfg.snapshot_dir + \
            f'/{train_cfg.resume_model_version}_model.pth'
        if not os.path.exists(snapshot_path):
            print(f'{snapshot_path} does not exist')
            exit()
        if train_cfg.resume_optimizer_state:
            optimizer_path = train_cfg.snapshot_dir + \
                f'/{train_cfg.resume_model_version}_optimizer'
            if not os.path.exists(optimizer_path):
                print(f'{optimizer_path} does not exist')
                exit()
            optimizer.load_state_dict(torch.load(optimizer_path))
        model.load_state_dict(torch.load(snapshot_path))
        agent_pool.difficulty = train_cfg.resume_difficulty
        print(f'resuming {train_cfg.training_name} '
              f'using {train_cfg.resume_model_version} model '
              f'at epoch {start_ep + 1}')
    # logging ----------------------------------------------------------------------------------
    name = train_cfg.training_name + '-' + resume_tag
    name += subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8')[:-1]
    # checking for --dirty
    git_diff = subprocess.Popen(['/usr/bin/git', 'diff', '--quiet'], stdout=subprocess.PIPE)
    ret_code = git_diff.wait()
    name += '-dirty' if ret_code != 0 else ''
    log_enable = train_cfg.training_name != 'debug'
    if log_enable:
        init_wandb(name, train_cfg)
    else:
        torch.autograd.set_detect_anomaly(True)
        print(f'disabled logging')
    # losses ------------------------------------------------------------------------------------
    if train_cfg.loss_function == 'cross-entropy':
        semseg_loss = nn.CrossEntropyLoss(reduction='none')
    elif train_cfg.loss_function == 'weighted-cross-entropy':
        semseg_loss = nn.CrossEntropyLoss(weight=torch.tensor(train_cfg.ce_weights), reduction='none')
    elif train_cfg.loss_function == 'focal':
        semseg_loss = FocalLoss(alpha=0.5, gamma=2.0, reduction='none')
    mask_loss = nn.L1Loss(reduction='mean')
    # send to gpu
    if train_cfg.device == 'cuda':
        semseg_loss = semseg_loss.to(device)
        mask_loss = mask_loss.to(device)
    # begin -------------------------------------------------------------------------------------
    train(train_cfg=train_cfg, device=device, log_enable=log_enable, model=model, optimizer=optimizer,
          agent_pool=agent_pool, scheduler=scheduler, mask_loss=mask_loss, semseg_loss=semseg_loss,
          geom_properties=(new_size, center, ppm), train_loader=train_loader, valid_loader=valid_loader,
          mask_loss_weight=mask_loss_weight, sseg_loss_weight=sseg_loss_weight, start_ep=start_ep,
          segmentation_classes=segmentation_classes)

if __name__ == '__main__':
    parse_and_execute()