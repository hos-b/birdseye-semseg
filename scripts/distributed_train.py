import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import wandb
import subprocess
from kornia.losses.focal import FocalLoss

from agent.agent_pool import CurriculumPool
from data.config import SemanticCloudConfig, TrainingConfig
from data.color_map import __our_classes as segmentation_classes
from data.dataset import get_datasets
from data.logging import log_string, init_wandb, log_wandb
from data.utils import drop_agent_data, squeeze_all
from data.utils import to_device
from evaluate import plot_batch
from metrics.iou import iou_per_class, mask_iou
from model.mcnn import MCNN, MCNN4

def train(gpu, *args):
    geom_cfg: SemanticCloudConfig = args[0]
    train_cfg: TrainingConfig = args[1]
    # distributed training ---------------------------------------------------------------------
    rank = gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=train_cfg.world_size,
        rank=rank
    )
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
    train_sampler = DistributedSampler(train_set, num_replicas=train_cfg.world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=train_cfg.world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, pin_memory=True,
                                               num_workers=0, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, pin_memory=True,
                                              num_workers=0, sampler=test_sampler)
    # logging ----------------------------------------------------------------------------------
    name = train_cfg.training_name + '-'
    name += subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8')[:-1]
    # checking for --dirty
    git_diff = subprocess.Popen(['/usr/bin/git', 'diff', '--quiet'], stdout=subprocess.PIPE)
    ret_code = git_diff.wait()
    name += '-dirty' if ret_code != 0 else ''
    log_enable = train_cfg.training_name != 'debug'
    if rank == 0 and log_enable:
        init_wandb(name, train_cfg)
    # saving snapshots -------------------------------------------------------------------------
    train_cfg.snapshot_dir = train_cfg.snapshot_dir.format(train_cfg.training_name)
    if not os.path.exists(train_cfg.snapshot_dir):
        os.makedirs(train_cfg.snapshot_dir)
    # network stuff ----------------------------------------------------------------------------
    if train_cfg.model_name == 'mcnn':
        model = MCNN(3, train_cfg.num_classes, NEW_SIZE,
                     geom_cfg, train_cfg.norm_keep_stats).cuda(gpu)
    elif train_cfg.model_name == 'mcnn4':
        model = MCNN4(3, train_cfg.num_classes, NEW_SIZE,
                      geom_cfg, train_cfg.norm_keep_stats).cuda(gpu)
    else:
        log_string(f'unknown network architecture {train_cfg.model_name}')
    log_string(f'{(model.parameter_count() / 1e6):.2f}M trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    agent_pool = CurriculumPool(train_cfg.initial_difficulty, train_cfg.maximum_difficulty,
                                train_cfg.max_agent_count, device)
    log_string(rank, f'{(model.parameter_count() / 1e6):.2f}M trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    ddp_model = DistributedDataParallel(model, device_ids=[gpu])
    # losses -----------------------------------------------------------------------------------
    if train_cfg.loss_function == 'cross-entropy':
        semseg_loss = nn.CrossEntropyLoss(reduction='none')
    elif train_cfg.loss_function == 'focal':
        semseg_loss = FocalLoss(alpha=0.5, gamma=2.0, reduction='none')
    else:
        log_string(f'unknown loss function: {train_cfg.loss_function}')
        exit()
    mask_loss = nn.L1Loss(reduction='mean')
    # send to gpu
    if train_cfg.device == 'cuda':
        semseg_loss = semseg_loss.cuda(gpu)
        mask_loss = mask_loss.cuda(gpu)
    epochs = train_cfg.epochs
    # training ---------------------------------------------------------------------------------
    for ep in range(epochs):
        train_sampler.set_epoch(ep)
        test_sampler.set_epoch(ep)
        total_train_m_loss = 0.0
        total_train_s_loss = 0.0
        sample_count = 0
        # training
        model.train()
        for batch_idx, (ids, rgbs, labels, masks, car_transforms) in enumerate(train_loader):
            sample_count += rgbs.shape[1]
            log_string(f'\repoch: {ep + 1}/{epochs}, '
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
            sseg_preds, mask_preds = model(rgbs, car_transforms, agent_pool.adjacency_matrix)
            m_loss = mask_loss(mask_preds.squeeze(1), masks)
            s_loss = torch.mean(semseg_loss(sseg_preds, labels) * agent_pool.combined_masks,
                                dim=(0, 1, 2))
            batch_train_m_loss += m_loss.item()
            batch_train_s_loss += s_loss.item()
            # (m_loss + s_loss).backward() TODO: uncomment
            s_loss.backward()
            optimizer.step()

            # writing batch loss
            if (batch_idx + 1) % train_cfg.log_every == 0 and log_enable:
                log_wandb({
                    'batch train mask': batch_train_m_loss,
                    'batch train seg': batch_train_s_loss
                })
            total_train_m_loss += batch_train_m_loss
            total_train_s_loss += batch_train_s_loss

        if log_enable:
            log_wandb({
                'total train mask loss': total_train_m_loss / sample_count,
                'total train seg loss': total_train_s_loss / sample_count,
                'epoch': ep + 1
            })
        log_string(f'\nepoch loss: {(total_train_m_loss / sample_count)} mask, '
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
            log_string(f'\repoch: {ep + 1}/{epochs}, '
                       f'validation batch: {batch_idx + 1} / {len(test_loader)}', end='')
            sample_count += rgbs.shape[1]
            rgbs, labels, masks, car_transforms = squeeze_all(rgbs, labels, masks, car_transforms)
            rgbs, labels, masks, car_transforms = to_device(rgbs, labels, masks, car_transforms,
                                                            device, train_cfg.pin_memory)
            agent_pool.generate_connection_strategy(ids, masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            with torch.no_grad():
                sseg_preds, mask_preds = model(rgbs, car_transforms, agent_pool.adjacency_matrix)
            sseg_ious += iou_per_class(sseg_preds, labels, masks).cuda(gpu)
            mask_ious += mask_iou(mask_preds.squeeze(1), masks, train_cfg.mask_detection_thresh)
            m_loss = mask_loss(mask_preds.squeeze(1), masks)
            s_loss = semseg_loss(sseg_preds, labels) * agent_pool.combined_masks
            total_valid_m_loss += torch.mean(m_loss).detach()
            total_valid_s_loss += torch.mean(s_loss).detach()
            # visaluize the first agent from the first batch
            if not visaulized and log_enable:
                img = plot_batch(rgbs, labels, sseg_preds, mask_preds, agent_pool, 'image')
                log_wandb({
                    'results': wandb.Image(img, caption='full batch predictions'),
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
            log_wandb(log_dict)
        log_string(f'\nepoch validation loss: {total_valid_s_loss / sample_count} mask, '
                   f'{total_valid_s_loss / sample_count} segmentation')
        # saving the new model -----------------------------------------------------------------
        snapshot_tag = 'last'
        if new_metric > last_metric:
            log_string(f'best model @ epoch {ep + 1}')
            last_metric = new_metric
            snapshot_tag = 'best'
        torch.save(optimizer.state_dict(), train_cfg.snapshot_dir +
                    f'/{snapshot_tag}_optimizer')
        torch.save(model.state_dict(), train_cfg.snapshot_dir +
                    f'/{snapshot_tag}_model.pth')
        # update curriculum difficulty ---------------------------------------------------------
        if train_cfg.strategy == 'every-x-epochs':
            agent_pool.update_difficulty(ep + 1)

    if rank == 0:
        wandb.finish()


if __name__ == '__main__':
    geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    train_cfg = TrainingConfig('config/training.yml')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    if not train_cfg.distributed:
        print('change training.distributed to true in the configs')
        exit()
    elif train_cfg.world_size <= 1:
        print(f'invalid world size for distributed training. expected >= 2, got {train_cfg.world_size}')
        exit()
    mp.spawn(train, nprocs=train_cfg.world_size, args=(geom_cfg, train_cfg))