import os
import cv2
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
from model.factory import get_model
from evaluate import plot_full_batch

def train(**kwargs):
    train_cfg: TrainingConfig = kwargs.get('train_cfg')
    NEW_SIZE, CENTER, PPM = kwargs.get('geom_properties')
    debug_mode = kwargs.get('debug_mode')
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
    last_snapshot_metric = 0.0
    # dataset
    train_loader = kwargs.get('train_loader')
    valid_loader = kwargs.get('valid_loader')
    segmentation_classes = kwargs.get('segmentation_classes')
    epochs = train_cfg.epochs
    # wallhack mask
    if train_cfg.wallhack_prob > 0:
        wallhack_mask_np = np.zeros(shape=(train_cfg.output_h, train_cfg.output_w), dtype=np.uint8)
        fov_vertices = np.array([[[81, 155], [122, 155], [204, 20], [204, 0], [0, 0], [0, 20]]], dtype=np.int32)
        cv2.fillPoly(wallhack_mask_np, fov_vertices, 1)
        wallhack_mask = torch.from_numpy(wallhack_mask_np).unsqueeze(0).to(device)
    else:
        wallhack_mask = torch.zeros((1, train_cfg.output_h, train_cfg.output_w), device=device)
    # starting epoch
    start_ep = kwargs.get('start_ep')
    for ep in range(start_ep, epochs):
        total_train_m_loss = 0.0
        total_train_s_loss = 0.0
        sample_count = 0
        # training
        model.train()
        for batch_idx, (rgbs, labels, car_masks, fov_masks, car_transforms, _) in enumerate(train_loader):
            sample_count += rgbs.shape[1]
            rgbs, labels, car_masks, fov_masks, car_transforms = to_device(device, rgbs, labels, car_masks,
                                                        fov_masks, car_transforms)
            # simulate connection drops
            rgbs, labels, car_masks, fov_masks, car_transforms = drop_agent_data(train_cfg.drop_prob,
                                                        rgbs, labels, car_masks, fov_masks, car_transforms)
            solo_masks = car_masks + fov_masks
            agent_pool.generate_connection_strategy(solo_masks, car_transforms,
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
            # add mask wallhack for semantics
            if random.uniform(0, 1) < train_cfg.wallhack_prob:
                wallhack = wallhack_mask
            else:
                wallhack = torch.zeros_like(wallhack_mask, device=device)
            # forward for base mcnn models
            if model.output_count == 2:
                aggr_sseg_preds, solo_mask_preds = model(rgbs, car_transforms,
                                                         agent_pool.adjacency_matrix,
                                                         car_masks)
                m_loss = mask_loss(solo_mask_preds.squeeze(1), solo_masks)
                s_loss =  torch.mean(semseg_loss(aggr_sseg_preds, labels) * 
                                     torch.clamp(agent_pool.combined_masks + wallhack, 0.0, 1.0))
            # forward for full (4x) models
            else:
                solo_sseg_preds, solo_mask_preds, aggr_sseg_preds, aggr_mask_preds = \
                    model(rgbs, car_transforms, agent_pool.adjacency_matrix, car_masks)
                m_loss = mask_loss(solo_mask_preds.squeeze(1), solo_masks) + \
                         mask_loss(aggr_mask_preds.squeeze(1), agent_pool.combined_masks)
                s_loss = torch.mean(semseg_loss(solo_sseg_preds, labels) *
                                    torch.clamp(solo_masks + wallhack, 0.0, 1.0)) + \
                         torch.mean(semseg_loss(aggr_sseg_preds, labels) *
                                    torch.clamp(agent_pool.combined_masks + wallhack, 0.0, 1.0))
            # semseg & mask batch loss
            batch_train_m_loss = m_loss.item()
            batch_train_s_loss = s_loss.item()
            # weighted losses
            (0.5 * m_loss * torch.exp(-mask_loss_weight) + 0.5 * mask_loss_weight +
                   s_loss * torch.exp(-sseg_loss_weight) + 0.5 * sseg_loss_weight).backward()

            optimizer.step()

            # log batch loss
            if (batch_idx + 1) % train_cfg.log_every == 0:
                if not debug_mode:
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
        if not debug_mode:
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
        for batch_idx, (rgbs, labels, car_masks, fov_masks, car_transforms, dataset_idx) in enumerate(valid_loader):
            print(f'\repoch: {ep + 1}/{epochs}, '
                  f'validation batch: {batch_idx + 1} / {len(valid_loader)}', end='')
            sample_count += rgbs.shape[1]
            rgbs, labels, car_masks, fov_masks, car_transforms = to_device(device, rgbs, labels, car_masks,
                                                        fov_masks, car_transforms)
            rgbs, labels, car_masks, fov_masks, car_transforms = squeeze_all(rgbs, labels, car_masks,
                                                        fov_masks, car_transforms)
            solo_masks = car_masks + fov_masks
            agent_pool.generate_connection_strategy(solo_masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            # add se2 noise to transforms
            if train_cfg.se2_noise_enable:
                car_transforms = get_noisy_transforms(car_transforms,
                                                      train_cfg.se2_noise_dx_std,
                                                      train_cfg.se2_noise_dy_std,
                                                      train_cfg.se2_noise_th_std)
            with torch.no_grad():
                # forward for base mcnn models
                if model.output_count == 2:
                    aggr_sseg_preds, solo_mask_preds = model(rgbs, car_transforms,
                                                             agent_pool.adjacency_matrix,
                                                             car_masks)
                    total_valid_m_loss += mask_loss(solo_mask_preds.squeeze(1), solo_masks)
                    total_valid_s_loss +=  torch.mean(semseg_loss(aggr_sseg_preds, labels) *
                                                      agent_pool.combined_masks)
                    # aliases for visualization
                    aggr_mask_preds = solo_mask_preds
                    solo_sseg_preds = aggr_sseg_preds
                # forward for full (4x) models
                else:
                    solo_sseg_preds, solo_mask_preds, aggr_sseg_preds, aggr_mask_preds = \
                        model(rgbs, car_transforms, agent_pool.adjacency_matrix, car_masks)
                    total_valid_m_loss += (mask_loss(solo_mask_preds.squeeze(1), solo_masks) +
                                           mask_loss(aggr_mask_preds.squeeze(1), agent_pool.combined_masks)).item()
                    total_valid_s_loss += (torch.mean(semseg_loss(solo_sseg_preds, labels) * solo_masks) +
                                           torch.mean(semseg_loss(aggr_sseg_preds, labels) * agent_pool.combined_masks)).item()
            sseg_ious += get_iou_per_class(aggr_sseg_preds, labels, agent_pool.combined_masks,
                                           train_cfg.num_classes).to(device)
            mask_ious += get_mask_iou(aggr_mask_preds.squeeze(1), agent_pool.combined_masks,
                                      train_cfg.mask_detection_thresh)
            # visualize a random batch
            if not visualized:
                validation_img_log_dict = {'misc/epoch': ep + 1}
                first_batch_img = plot_full_batch(rgbs, labels, solo_sseg_preds, aggr_sseg_preds,
                                                  solo_mask_preds, aggr_mask_preds,
                                                  solo_masks, agent_pool.combined_masks,
                                                  plot_dest='image', semantic_classes=train_cfg.classes,
                                                  title=f'E: {ep + 1}, B#: {dataset_idx.item()}')
                validation_img_log_dict['media/results'] = \
                    wandb.Image(first_batch_img, caption='full batch predictions')
                if not debug_mode:
                    wandb.log(validation_img_log_dict)
                visualized = True
            # end of batch

        # more wandb logging -------------------------------------------------------------------
        avg_iou = 0.0
        log_dict = {}
        for key, val in segmentation_classes.items():
            log_dict[f'iou/{val.lower()}'] = (sseg_ious[key] / sample_count).item()
            if val != 'Misc' and val != 'Water':
                avg_iou += sseg_ious[key] / sample_count

        if train_cfg.classes == 'ours':
            # 2 classes are irrelevant, so are masks
            avg_iou /= train_cfg.num_classes - 2
        else:
            avg_iou /= train_cfg.num_classes

        log_dict['loss/total validation mask'] = (total_valid_m_loss / sample_count)
        log_dict['loss/total validation sseg'] = (total_valid_s_loss / sample_count)
        log_dict['iou/mask'] = (mask_ious / sample_count).item()
        log_dict['misc/epoch'] = ep + 1
        log_dict['misc/save'] = 0
        log_dict['curriculum/elevation metric'] = avg_iou.item()
        log_dict['weight/sseg'] = torch.exp(-sseg_loss_weight).item()
        log_dict['weight/mask'] = torch.exp(-mask_loss_weight).item()
        print(f'\nepoch validation loss: {total_valid_m_loss / sample_count} mask, '
              f'{total_valid_s_loss / sample_count} segmentation')
        # saving the new model -----------------------------------------------------------------
        snapshot_tag = 'last'
        if avg_iou > last_snapshot_metric:
            print(f'best model @ epoch {ep + 1}')
            last_snapshot_metric = avg_iou
            snapshot_tag = 'best'
            log_dict['misc/save'] = 1
        torch.save(optimizer.state_dict(), train_cfg.snapshot_dir +
                    f'/{snapshot_tag}_optimizer')
        torch.save(model.state_dict(), train_cfg.snapshot_dir +
                    f'/{snapshot_tag}_model.pth')
        scheduler.step()
        # update curriculum difficulty ---------------------------------------------------------
        if train_cfg.curriculum_activate:
            increase_diff = False
            if train_cfg.strategy == 'every-x-epochs':
                if (ep + 1) % int(train_cfg.strategy_parameter) == 0:
                    increase_diff = True
            elif train_cfg.strategy == 'metric':
                # elevation metric = avg[avg[important class IoU] + avg[mask IoU]]
                avg_iou /= 6
                if avg_iou >= train_cfg.strategy_parameter:
                    increase_diff = True
            if increase_diff:
                agent_pool.difficulty = min(agent_pool.difficulty + 1,
                                            agent_pool.maximum_difficulty)
                print(f'\n=======>> difficulty increased to {agent_pool.difficulty} <<=======')
        log_dict['curriculum/difficulty'] = agent_pool.difficulty
        if not debug_mode:
            wandb.log(log_dict)
    # end
    if not debug_mode:
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
    print(f'pixels per meter: {ppm}, vehicle center {center}')
    # dataset ----------------------------------------------------------------------------------
    train_set = MassHDF5(dataset=train_cfg.trainset_name, path=train_cfg.dset_dir,
                         hdf5name=train_cfg.trainset_file, size=new_size,
                         classes=train_cfg.classes, jitter=train_cfg.color_jitter,
                         mask_gaussian_sigma=train_cfg.gaussian_mask_std,
                         guassian_kernel_size=train_cfg.gaussian_kernel_size)
    valid_set = MassHDF5(dataset=train_cfg.validset_name, path=train_cfg.dset_dir,
                         hdf5name=train_cfg.validset_file, size=new_size,
                         classes=train_cfg.classes, jitter=[0, 0, 0, 0],
                         mask_gaussian_sigma=train_cfg.gaussian_mask_std,
                         guassian_kernel_size=train_cfg.gaussian_kernel_size)
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
    elif train_cfg.classes == 'ours+mask':
        print('ours+mask should not be used with this script'); exit()
    elif train_cfg.classes == 'diminished':
        segmentation_classes = color_map.__diminished_classes
    # snapshot dir -----------------------------------------------------------------------------
    train_cfg.snapshot_dir = train_cfg.snapshot_dir.format(train_cfg.training_name)
    if not os.path.exists(train_cfg.snapshot_dir):
        os.makedirs(train_cfg.snapshot_dir)
    # network stuff ----------------------------------------------------------------------------
    model = get_model(train_cfg.model_name, train_cfg.num_classes, new_size,
                      geom_cfg, train_cfg.aggregation_type).to(device)
    print(f'{(model.parameter_count() / 1e6):.2f}M trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    start_ep = train_cfg.resume_starting_epoch if train_cfg.resume_training else 0
    lr_lambda = lambda epoch: pow((1 - (((epoch + start_ep) - 1) / train_cfg.epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    mask_loss_weight = torch.tensor([0.0], requires_grad=True, device=device)
    sseg_loss_weight = torch.tensor([0.0], requires_grad=True, device=device)
    optimizer.add_param_group({"params": [mask_loss_weight, sseg_loss_weight]})
    agent_pool = CurriculumPool(train_cfg.initial_difficulty, train_cfg.maximum_difficulty,
                                train_cfg.max_agent_count, train_cfg.enforce_adj_calc, device)
    # loading the network parameters/optimizer state -------------------------------------------
    resume_tag = ''
    if train_cfg.resume_training:
        resume_tag = train_cfg.resume_tag + '-'
        snapshot_path = train_cfg.snapshot_dir + \
            f'/{train_cfg.resume_model_version}_model.pth'
        if not os.path.exists(snapshot_path):
            print(f'{snapshot_path} does not exist'); exit()
        if train_cfg.resume_optimizer_state:
            optimizer_path = train_cfg.snapshot_dir + \
                f'/{train_cfg.resume_model_version}_optimizer'
            if not os.path.exists(optimizer_path):
                print(f'{optimizer_path} does not exist'); exit()
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
    debug_mode = train_cfg.training_name == 'debug'
    if not debug_mode:
        init_wandb(name, train_cfg)
    else:
        torch.autograd.set_detect_anomaly(True)
        print(f'disabled logging')
    # losses -----------------------------------------------------------------------------------
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
    # begin ------------------------------------------------------------------------------------
    train_cfg.print_config()
    train(train_cfg=train_cfg, device=device, debug_mode=debug_mode, model=model, optimizer=optimizer,
          agent_pool=agent_pool, scheduler=scheduler, mask_loss=mask_loss, semseg_loss=semseg_loss,
          geom_properties=(new_size, center, ppm), train_loader=train_loader, valid_loader=valid_loader,
          mask_loss_weight=mask_loss_weight, sseg_loss_weight=sseg_loss_weight, start_ep=start_ep,
          segmentation_classes=segmentation_classes)

if __name__ == '__main__':
    parse_and_execute()