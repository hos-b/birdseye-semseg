import os
import cv2
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import wandb
import subprocess
from kornia.losses.focal import FocalLoss

from agent.agent_pool import CurriculumPool
from data.config import SemanticCloudConfig, TrainingConfig
from data.color_map import our_semantics_to_cityscapes_rgb
from data.color_map import __our_classes as segmentation_classes
from data.dataset import get_datasets
from data.logging import init_wandb
from data.utils import drop_agent_data, squeeze_all
from data.utils import get_matplotlib_image, to_device
from metrics.iou import iou_per_class, mask_iou
from model.mcnn import MCNN, MCNN4


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
    last_metric = 0.0
    # dataset
    train_loader = kwargs.get('train_loader')
    test_loader = kwargs.get('test_loader')
    epochs = train_cfg.epochs
    for ep in range(epochs):
        total_train_m_loss = 0.0
        total_train_s_loss = 0.0
        sample_count = 0
        # training
        model.train()
        for batch_idx, (ids, rgbs, labels, masks, car_transforms) in enumerate(train_loader):
            sample_count += rgbs.shape[1]
            print(f'\repoch: {ep + 1}/{epochs}, '
                  f'training batch: {batch_idx + 1} / {len(train_loader)}', end='')
            rgbs, labels, masks, car_transforms = to_device(rgbs, labels,
                                                            masks, car_transforms,
                                                            device, train_cfg.pin_memory)
            # simulate connection drops [disabled for now]
            rgbs, labels, masks, car_transforms = drop_agent_data(rgbs, labels,
                                                                  masks, car_transforms,
                                                                  train_cfg.drop_prob)
            # semseg & mask batch loss
            batch_train_m_loss = 0.0
            batch_train_s_loss = 0.0
            agent_pool.generate_connection_strategy(ids, masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            # agent count x fwd-bwd
            optimizer.zero_grad()
            sseg_pred, mask_pred = model(rgbs, car_transforms, agent_pool.adjacency_matrix)
            m_loss = mask_loss(mask_pred.squeeze(1), masks)
            s_loss = torch.mean(semseg_loss(sseg_pred, labels) * agent_pool.combined_masks,
                                dim=(0, 1, 2))
            batch_train_m_loss += m_loss.item()
            batch_train_s_loss += s_loss.item()
            # (m_loss + s_loss).backward()
            s_loss.backward()
            optimizer.step()

            # writing batch loss
            if (batch_idx + 1) % train_cfg.log_every == 0 and log_enable:
                wandb.log({
                    'batch train mask': batch_train_m_loss,
                    'batch train seg': batch_train_s_loss
                })
            total_train_m_loss += batch_train_m_loss
            total_train_s_loss += batch_train_s_loss
        if log_enable:
            wandb.log({
                'total train mask loss': total_train_m_loss / sample_count,
                'total train seg loss': total_train_s_loss / sample_count,
                'epoch': ep + 1
            })
        print(f'\nepoch loss: {(total_train_m_loss / sample_count)} mask, '
              f'{(total_train_s_loss / sample_count)} segmentation')
        # validation ---------------------------------------------------------------------------
        model.eval()
        visaulized = False
        total_valid_m_loss = 0.0
        total_valid_s_loss = 0.0
        sseg_ious = torch.zeros((train_cfg.num_classes, 1), dtype=torch.float64).cuda(0)
        mask_ious = 0.0
        sample_count = 0
        for batch_idx, (ids, rgbs, labels, masks, car_transforms) in enumerate(test_loader):
            print(f'\repoch: {ep + 1}/{epochs}, '
                  f'validation batch: {batch_idx + 1} / {len(test_loader)}', end='')
            sample_count += rgbs.shape[1]
            rgbs, labels, masks, car_transforms = squeeze_all(rgbs, labels, masks, car_transforms)
            rgbs, labels, masks, car_transforms = to_device(rgbs, labels,
                                                            masks, car_transforms,
                                                            device, train_cfg.pin_memory)
            agent_pool.generate_connection_strategy(ids, masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            with torch.no_grad():
                sseg_preds, mask_preds = model(rgbs, car_transforms, agent_pool.adjacency_matrix)
            sseg_ious += iou_per_class(sseg_preds, labels).cuda(0)
            mask_ious += mask_iou(mask_preds.squeeze(1), masks, train_cfg.mask_detection_thresh)
            m_loss = mask_loss(mask_preds.squeeze(1), masks)
            s_loss = semseg_loss(sseg_preds, labels) * agent_pool.combined_masks
            total_valid_m_loss += torch.mean(m_loss).detach()
            total_valid_s_loss += torch.mean(s_loss).detach()
            # visaluize the first agent from the first batch
            if not visaulized:
                # masked target semantics
                ss_trgt_img = our_semantics_to_cityscapes_rgb(labels[0].cpu()).transpose(2, 0, 1)
                ss_mask = agent_pool.combined_masks[0].cpu()
                ss_trgt_img[:, ss_mask == 0] = 0
                # predicted semantics
                _, ss_pred = torch.max(sseg_preds[0], dim=0)
                ss_pred_img = our_semantics_to_cityscapes_rgb(ss_pred.cpu()).transpose(2, 0, 1)
                # predicted & target mask
                pred_mask = get_matplotlib_image(mask_preds[0].squeeze().cpu())
                trgt_mask = get_matplotlib_image(masks[0].cpu())
                if log_enable:
                    wandb.log({
                        'input rgb' : [
                            wandb.Image(rgbs[0], caption='input image'),
                        ],
                        'output': [
                            wandb.Image(pred_mask, caption='predicted mask'),
                            wandb.Image(trgt_mask, caption='target mask'),
                            wandb.Image(ss_pred_img.transpose(1, 2, 0), caption='predicted semantics'),
                            wandb.Image(ss_trgt_img.transpose(1, 2, 0), caption='target semantics')
                        ],
                        'epoch': ep + 1
                    })
                visaulized = True

        # more wandb logging -------------------------------------------------------------------
        new_metric = 0.0
        log_dict = {}
        for key, val in segmentation_classes.items():
            log_dict[f'{val.lower()} iou'] = (sseg_ious[key] / sample_count).item()
            new_metric += sseg_ious[key] / sample_count
        new_metric += mask_ious / sample_count
        log_dict['total validation mask loss'] = (total_valid_m_loss / sample_count).item()
        log_dict['total validation seg loss'] = (total_valid_s_loss / sample_count).item()
        log_dict['mask iou'] = (mask_ious / sample_count).item()
        log_dict['epoch'] = ep + 1
        if log_enable:
            wandb.log(log_dict)
        print(f'\nepoch validation loss: {total_valid_s_loss / sample_count} mask, '
              f'{total_valid_s_loss / sample_count} segmentation')
        # saving the new model -----------------------------------------------------------------
        snapshot_tag = 'last'
        if new_metric > last_metric:
            print(f'best model @ epoch {ep + 1}')
            last_metric = new_metric
            snapshot_tag = 'best'
        torch.save(optimizer.state_dict(), train_cfg.snapshot_dir +
                    f'/{snapshot_tag}_optimizer')
        torch.save(model.state_dict(), train_cfg.snapshot_dir +
                    f'/{snapshot_tag}_model.pth')
        # update curriculum difficulty ---------------------------------------------------------
        scheduler.step()
        if train_cfg.strategy == 'every-x-epochs':
            agent_pool.update_difficulty(ep + 1)

    if log_enable:
        wandb.finish()


def parse_and_execute():
    # parsing config file
    geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    train_cfg = TrainingConfig('config/training.yml')
    debug = train_cfg.training_name == 'debug'
    if train_cfg.distributed:
        print('change training.distributed to false in the configs')
        exit()
    # gpu selection ----------------------------------------------------------------------------
    device_str = train_cfg.device
    if train_cfg.device == 'cuda':
        torch.cuda.set_device(0)
        device_str += f':{0}'
    device = torch.device(device_str)
    torch.manual_seed(train_cfg.torch_seed)
    # image size and center coordinates --------------------------------------------------------
    new_size = (train_cfg.output_h, train_cfg.output_w)
    center = (geom_cfg.center_x(new_size[1]), geom_cfg.center_y(new_size[0]))
    ppm = geom_cfg.pix_per_m(new_size[0], new_size[1])
    # dataset ----------------------------------------------------------------------------------
    train_set, test_set = get_datasets(train_cfg.dset_name, train_cfg.dset_dir,
                                       train_cfg.dset_file, (0.8, 0.2),
                                       new_size, train_cfg.classes)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=train_cfg.shuffle_data,
                                               pin_memory=train_cfg.pin_memory,
                                               num_workers=train_cfg.loader_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=train_cfg.shuffle_data,
                                              pin_memory=train_cfg.pin_memory,
                                              num_workers=train_cfg.loader_workers)
    # logging ----------------------------------------------------------------------------------
    name = train_cfg.training_name + '-'
    name += subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8')[:-1]
    # checking for --dirty
    git_diff = subprocess.Popen(['/usr/bin/git', 'diff', '--quiet'], stdout=subprocess.PIPE)
    ret_code = git_diff.wait()
    name += '-dirty' if ret_code != 0 else ''
    log_enable = train_cfg.training_name != 'debug'
    init_wandb(name, train_cfg) if log_enable else print(f'disabled logging')
    # saving snapshots -------------------------------------------------------------------------
    if not os.path.exists(train_cfg.snapshot_dir):
        os.makedirs(train_cfg.snapshot_dir)
    # network stuff ----------------------------------------------------------------------------
    model = MCNN(3, train_cfg.num_classes, new_size, geom_cfg).cuda(0)
    print(f'{(model.parameter_count() / 1e6):.2f}M trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    agent_pool = CurriculumPool(train_cfg.initial_difficulty, train_cfg.maximum_difficulty,
                                train_cfg.max_agent_count, train_cfg.strategy,
                                train_cfg.strategy_parameter, device)
    lr_lambda = lambda epoch: pow((1 - ((epoch - 1) / train_cfg.epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # losses -----------------------------------------------------------------------------------
    if train_cfg.loss_function == 'cross-entropy':
        semseg_loss = nn.CrossEntropyLoss(reduction='none')
    elif train_cfg.loss_function == 'focal':
        semseg_loss = FocalLoss(alpha=0.5, gamma=2.0, reduction='none')
    else:
        print(f'unknown loss function: {train_cfg.loss_function}')
        exit()
    mask_loss = nn.L1Loss(reduction='mean')
    # send to gpu
    if train_cfg.device == 'cuda':
        semseg_loss = semseg_loss.cuda(0)
        mask_loss = mask_loss.cuda(0)
    train(train_cfg=train_cfg, device=device, log_enable=log_enable, model=model, optimizer=optimizer,
          agent_pool=agent_pool, scheduler=scheduler, mask_loss=mask_loss, semseg_loss=semseg_loss,
          geom_properties=(new_size, center, ppm), train_loader=train_loader, test_loader= test_loader)

if __name__ == '__main__':
    parse_and_execute()