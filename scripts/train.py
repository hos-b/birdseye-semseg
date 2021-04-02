import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torchvision import transforms
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
# from model.mass_cnn import DistributedMassCNN
# from model.fast_scnn import FastSCNN
from model.mcnn import MCNN4

def main(gpu, geom_cfg: SemanticCloudConfig, train_cfg: TrainingConfig):
    # gpu selection ----------------------------------------------------------------------------
    device_str = train_cfg.device
    if train_cfg.device == 'cuda':
        torch.cuda.set_device(gpu)
        device_str += f':{gpu}'
    device = torch.device(device_str)
    torch.manual_seed(train_cfg.torch_seed)
    # image size and center coordinates --------------------------------------------------------
    NEW_SIZE = (train_cfg.output_h, train_cfg.output_w)
    CENTER = (geom_cfg.center_x(NEW_SIZE[1]), geom_cfg.center_y(NEW_SIZE[0]))
    PPM = geom_cfg.pix_per_m(NEW_SIZE[0], NEW_SIZE[1])
    # dataset ----------------------------------------------------------------------------------
    train_set, test_set = get_datasets(train_cfg.dset_name, train_cfg.dset_dir, train_cfg.dset_file,
                                      (0.8, 0.2), NEW_SIZE, train_cfg.classes)
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
    if ret_code != 0:
        name += '-dirty'
    if train_cfg.logger == 'wandb':
        init_wandb(name, train_cfg)
    else:
        print(f'unsupported logger')
    # saving snapshots -------------------------------------------------------------------------
    last_metric = 0.0
    # network stuff ----------------------------------------------------------------------------
    model = MCNN4(3, train_cfg.num_classes, NEW_SIZE, geom_cfg).cuda(gpu)
    print(f'{(model.parameter_count() / 1e6):.2f}M trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    agent_pool = CurriculumPool(train_cfg.initial_difficulty, train_cfg.maximum_difficulty,
                                train_cfg.max_agent_count, train_cfg.strategy,
                                train_cfg.strategy_parameter, device)
    lr_lambda = lambda epoch: pow((1 - ((epoch - 1) / train_cfg.epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # custom loss parameters
    mask_loss_weight = torch.tensor([0.0], requires_grad=True, device=device)
    sseg_loss_weight = torch.tensor([0.0], requires_grad=True, device=device)
    optimizer.add_param_group({"params": mask_loss_weight})
    optimizer.add_param_group({"params": sseg_loss_weight})
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
        semseg_loss = semseg_loss.cuda(gpu)
        mask_loss = mask_loss.cuda(gpu)
    epochs = train_cfg.epochs
    # training ---------------------------------------------------------------------------------
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
            (m_loss * torch.exp(-mask_loss_weight) + mask_loss_weight +
             s_loss * torch.exp(-sseg_loss_weight) + sseg_loss_weight).backward()
            optimizer.step()

            # writing batch loss
            if (batch_idx + 1) % train_cfg.log_every == 0:
                wandb.log({
                    'batch train mask': batch_train_m_loss,
                    'batch train seg': batch_train_s_loss
                })
            total_train_m_loss += batch_train_m_loss
            total_train_s_loss += batch_train_s_loss

        # syncing tensors for wandb logging ----------------------------------------------------
        # total_train_m_loss = sync_tensor(rank, 0, total_train_m_loss, train_cfg.world_size)
        # total_train_s_loss = sync_tensor(rank, 1, total_train_s_loss, train_cfg.world_size)
        # sample_count = sync_tensor(rank, 2, sample_count, train_cfg.world_size)
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
        sseg_ious = torch.zeros((train_cfg.num_classes, 1), dtype=torch.float64).cuda(gpu)
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
            sseg_ious += iou_per_class(sseg_preds, labels).cuda(gpu)
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
        log_dict['mask loss weight'] = torch.exp(-mask_loss_weight).item()
        log_dict['sseg loss weight'] = torch.exp(-sseg_loss_weight).item()
        log_dict['epoch'] = ep + 1
        wandb.log(log_dict)
        print(f'\nepoch validation loss: {total_valid_s_loss / sample_count} mask, '
              f'{total_valid_s_loss / sample_count} segmentation')
        # saving the new model if it's better --------------------------------------------------
        if new_metric > last_metric:
            print(f'saving snapshot at epoch {ep + 1}')
            last_metric = new_metric
            if not os.path.exists(train_cfg.snapshot_dir):
                os.makedirs(train_cfg.snapshot_dir)
            torch.save(optimizer.state_dict(), train_cfg.snapshot_dir + '/optimizer_dict')
            torch.save(model.state_dict(), train_cfg.snapshot_dir + '/model_snapshot.pth')
        # update curriculum difficulty ---------------------------------------------------------
        scheduler.step()
        if train_cfg.strategy == 'every-x-epochs':
            agent_pool.update_difficulty(ep + 1)

    wandb.finish()


if __name__ == '__main__':
    geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    train_cfg = TrainingConfig('config/training.yml')
    if train_cfg.distributed:
        print('change training.distributed to false in the configs')
        exit()
    main(0, geom_cfg, train_cfg)