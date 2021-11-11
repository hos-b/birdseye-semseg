import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

# just for random seed
import random
import numpy as np

import wandb
import subprocess

import matplotlib
matplotlib.use('Agg')

from agent.agent_pool import CurriculumPool
from data.config import SemanticCloudConfig, TrainingConfig
from data.dataset import MassHDF5
from data.logging import init_wandb
from metrics.iou import get_mask_iou
from data.utils import squeeze_all
from data.utils import to_device
from evaluate import plot_mask_batch
from model.maskcnn import MaskCNN

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
    last_snapshot_metric = 1e6
    # dataset
    train_loader = kwargs.get('train_loader')
    valid_loader = kwargs.get('valid_loader')
    epochs = train_cfg.epochs
    # starting epoch
    start_ep = kwargs.get('start_ep')
    for ep in range(start_ep, epochs):
        total_train_s_m_loss = 0.0
        total_train_a_m_loss = 0.0
        sample_count = 0
        # training
        model.train()
        for batch_idx, (rgbs, labels, car_masks, fov_masks, car_transforms, _) in enumerate(train_loader):
            sample_count += rgbs.shape[1]
            rgbs, labels, car_masks, fov_masks, car_transforms = to_device(device, rgbs, labels, car_masks,
                                                                        fov_masks, car_transforms)
            rgbs, labels, car_masks, fov_masks, car_transforms = squeeze_all(rgbs, labels, car_masks,
                                                                        fov_masks, car_transforms)
            solo_masks = car_masks + fov_masks
            agent_pool.generate_connection_strategy(solo_masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            # fwd-bwd
            optimizer.zero_grad()
            solo_pred, aggr_pred = model(rgbs, car_transforms, agent_pool.adjacency_matrix, car_masks)
            m_solo_loss = mask_loss(solo_pred.squeeze(1), solo_masks)
            m_aggr_loss = mask_loss(aggr_pred.squeeze(1), agent_pool.combined_masks)
            (m_solo_loss + m_aggr_loss).backward()
            # semseg & mask batch loss
            batch_train_s_m_loss = m_solo_loss.item()
            batch_train_a_m_loss = m_aggr_loss.item()
            optimizer.step()
            # log batch loss
            if (batch_idx + 1) % train_cfg.log_every == 0:
                if log_enable:
                    wandb.log({
                        'loss/batch train solo mask': batch_train_s_m_loss,
                        'loss/batch train aggr mask': batch_train_a_m_loss
                    })
                print(f'\repoch: {ep + 1}/{epochs}, '
                      f'training batch: {batch_idx + 1} / {len(train_loader)}', end='')
            total_train_s_m_loss += batch_train_s_m_loss
            total_train_a_m_loss += batch_train_a_m_loss
            # end of batch

        # log train epoch loss
        if log_enable:
            wandb.log({
                'loss/total train solo mask': total_train_s_m_loss / sample_count,
                'loss/total train aggr mask': total_train_a_m_loss / sample_count,
                'misc/epoch': ep + 1
            })
        print(f'\nepoch loss: {(total_train_s_m_loss / sample_count):.6f} solo, '
                            f'{(total_train_a_m_loss / sample_count):.6f} aggregated')
        # validation ---------------------------------------------------------------------------
        model.eval()
        visualized = False
        total_valid_s_m_loss = 0.0
        total_valid_a_m_loss = 0.0
        solo_mask_iou = 0.0
        aggr_mask_iou = 0.0
        sample_count = 0
        for batch_idx, (rgbs, labels, car_masks, fov_masks, car_transforms, batch_no) in enumerate(valid_loader):
            if (batch_idx + 1) % train_cfg.log_every == 0:
                print(f'\repoch: {ep + 1}/{epochs}, '
                    f'validation batch: {batch_idx + 1} / {len(valid_loader)}', end='')
            rgbs, labels, car_masks, fov_masks, car_transforms = to_device(device, rgbs, labels, car_masks,
                                                                        fov_masks, car_transforms)
            rgbs, labels, car_masks, fov_masks, car_transforms = squeeze_all(rgbs, labels, car_masks,
                                                                        fov_masks, car_transforms)
            sample_count += rgbs.shape[0]
            solo_masks = car_masks + fov_masks
            agent_pool.generate_connection_strategy(solo_masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            with torch.no_grad():
                solo_pred, aggr_pred = model(rgbs, car_transforms, agent_pool.adjacency_matrix, car_masks)
                m_solo_loss = mask_loss(solo_pred.squeeze(1), solo_masks)
                m_aggr_loss = mask_loss(aggr_pred.squeeze(1), agent_pool.combined_masks)
            solo_mask_iou += get_mask_iou(solo_pred.squeeze(1), solo_masks, 
                                          train_cfg.mask_detection_thresh)
            aggr_mask_iou += get_mask_iou(aggr_pred.squeeze(1), agent_pool.combined_masks,
                                          train_cfg.mask_detection_thresh)
            # sum up losses
            total_valid_s_m_loss += m_solo_loss.item()
            total_valid_a_m_loss += m_aggr_loss.item()
            # visualize a random batch and all hard batches [if enabled]
            if not visualized and log_enable:
                first_batch_img = plot_mask_batch(
                    rgbs, labels, solo_pred, aggr_pred, solo_masks,
                    agent_pool.combined_masks, plot_dest='image',
                    semantic_classes=train_cfg.classes,
                    title=f'E: {ep + 1}, B#: {batch_no.item()}'
                )
                wandb.log({
                    'misc/epoch': ep + 1,
                    'media/results': wandb.Image(first_batch_img,
                                                 caption='full batch predictions')
                })
                visualized = True
            # end of batch

        # more wandb logging -------------------------------------------------------------------
        log_dict = {}
        log_dict['loss/total validation solo mask'] = (total_valid_s_m_loss / sample_count)
        log_dict['loss/total validation aggr mask'] = (total_valid_a_m_loss / sample_count)
        log_dict['iou/solo mask'] = (solo_mask_iou / sample_count).item()
        log_dict['iou/aggr mask'] = (aggr_mask_iou / sample_count).item()
        log_dict['misc/epoch'] = ep + 1
        log_dict['misc/save'] = 0
        print(f'\nepoch validation loss: {(total_valid_s_m_loss / sample_count):.6f} solo, '
                                       f'{(total_valid_a_m_loss / sample_count):.6f} aggregated')
        # saving the new model -----------------------------------------------------------------
        snapshot_tag = 'last'
        new_snapshot_metric = log_dict['loss/total validation solo mask'] + \
                              log_dict['loss/total validation aggr mask']
        if new_snapshot_metric < last_snapshot_metric:
            print(f'best model @ epoch {ep + 1}')
            last_snapshot_metric = new_snapshot_metric
            snapshot_tag = 'best'
            log_dict['misc/save'] = 1
        torch.save(optimizer.state_dict(), train_cfg.snapshot_dir +
                    f'/{snapshot_tag}_optimizer')
        torch.save(model.state_dict(), train_cfg.snapshot_dir +
                    f'/{snapshot_tag}_model.pth')
        # update curriculum difficulty ---------------------------------------------------------
        scheduler.step()
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
                         classes=train_cfg.classes, jitter=train_cfg.color_jitter,
                         mask_gaussian_sigma=0)
    valid_set = MassHDF5(dataset=train_cfg.validset_name, path=train_cfg.dset_dir,
                         hdf5name=train_cfg.validset_file, size=new_size,
                         classes=train_cfg.classes, jitter=[0, 0, 0, 0],
                         mask_gaussian_sigma=0)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=train_cfg.shuffle_data,
                                               num_workers=train_cfg.loader_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1,
                                               shuffle=train_cfg.shuffle_data,
                                               num_workers=train_cfg.loader_workers)
    # snapshot dir -----------------------------------------------------------------------------
    train_cfg.snapshot_dir = train_cfg.snapshot_dir.format(train_cfg.training_name)
    if not os.path.exists(train_cfg.snapshot_dir):
        os.makedirs(train_cfg.snapshot_dir)
    # network stuff ----------------------------------------------------------------------------
    model = MaskCNN(new_size, geom_cfg, train_cfg.aggregation_type).to(device)
    print(f'{(model.parameter_count() / 1e6):.2f}M trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    start_ep = train_cfg.resume_starting_epoch if train_cfg.resume_training else 0
    lr_lambda = lambda epoch: pow((1 - (((epoch + start_ep) - 1) / train_cfg.epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    agent_pool = CurriculumPool(train_cfg.initial_difficulty, train_cfg.maximum_difficulty,
                                train_cfg.max_agent_count, train_cfg.enforce_adj_calc, device)
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
    mask_loss = nn.L1Loss(reduction='mean')
    # send to gpu
    if train_cfg.device == 'cuda':
        mask_loss = mask_loss.to(device)
    # begin -------------------------------------------------------------------------------------
    print('training masks only, most settings in training.yml are ignored')
    train_cfg.print_config()
    train(train_cfg=train_cfg, device=device, log_enable=log_enable, model=model, optimizer=optimizer,
          agent_pool=agent_pool, scheduler=scheduler, mask_loss=mask_loss,start_ep=start_ep,
          geom_properties=(new_size, center, ppm), train_loader=train_loader, valid_loader=valid_loader)

if __name__ == '__main__':
    parse_and_execute()