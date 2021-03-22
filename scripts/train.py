import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import wandb
import subprocess
from tensorboardX import SummaryWriter
from kornia.losses.focal import FocalLoss

from agent.agent_pool import CurriculumPool
from data.dataset import get_datasets
from data.dist import sync_tensor
from data.config import SemanticCloudConfig, TrainingConfig
from data.color_map import our_semantics_to_cityscapes_rgb
from data.color_map import __our_classes as segmentation_classes
from data.logging import log_scalar, log_image, log_string
from data.utils import drop_agent_data, squeeze_all, init_wandb
from data.utils import get_matplotlib_image, to_device
from metrics.iou import iou_per_class, mask_iou
from model.mass_cnn import DistributedMassCNN

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
    writer = None
    if rank == 0:
        if train_cfg.logger == 'tensorboard':
            writer = SummaryWriter(os.path.join(train_cfg.log_dir, name))
        elif train_cfg.logger == 'wandb':
            init_wandb(name, train_cfg)
    # saving snapshots -------------------------------------------------------------------------
    last_metric = 0.0
    # network stuff ----------------------------------------------------------------------------
    model = DistributedMassCNN(geom_cfg, gpu,
                               num_classes=train_cfg.num_classes,
                               output_size=NEW_SIZE).cuda(gpu)
    print(f"{(model.parameter_count() / 1e6):.2f}M trainable parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    ddp_model = DistributedDataParallel(model, device_ids=[gpu])
    agent_pool = CurriculumPool(train_cfg.initial_difficulty,
                                train_cfg.maximum_difficulty,
                                train_cfg.max_agent_count)
    # losses -----------------------------------------------------------------------------------
    if train_cfg.loss_function == 'cross-entropy':
        semseg_loss = nn.CrossEntropyLoss(reduction='none')
    elif train_cfg.loss_function == 'focal':
        semseg_loss = FocalLoss(alpha=0.5, gamma=2.0, reduction='none')
    else:
        print(f'unknown loss function: {train_cfg.loss_function}')
        exit()
    mask_loss = nn.L1Loss(reduction='mean')
    # send to gpu if distributed
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
        ddp_model.train()
        for batch_idx, (ids, rgbs, labels, masks, car_transforms) in enumerate(train_loader):
            sample_count += rgbs.shape[1]
            log_string(rank, f'\repoch: {ep + 1}/{epochs}, '
                             f'training batch: {batch_idx + 1} / {len(train_loader)}', end='')
            rgbs, labels, masks, car_transforms = to_device(rgbs, labels,
                                                            masks, car_transforms,
                                                            device, True)
            # simulate connection drops [disabled for now]
            rgbs, labels, masks, car_transforms = drop_agent_data(rgbs, labels,
                                                                  masks, car_transforms,
                                                                  train_cfg.drop_prob)
            # semseg & mask loss
            batch_train_m_loss = 0.0
            batch_train_s_loss = 0.0
            optimizer.zero_grad()
            agent_pool.generate_connection_strategy(ids, masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            detached_features = None
            # async fwd-bwd
            with ddp_model.no_sync():
                for i in range(agent_pool.agent_count - 1):
                    detached_features, mask_pred, sseg_pred = ddp_model(rgbs, car_transforms,
                                                                        detached_features, i,
                                                                        agent_pool.adjacency_matrix,
                                                                        False)
                    m_loss = mask_loss(mask_pred.squeeze(), masks[i])
                    s_loss = torch.mean(semseg_loss(sseg_pred, labels[i].unsqueeze(0)) *
                                            agent_pool.combined_masks[i],
                                        dim=(0, 1, 2))
                    batch_train_m_loss += m_loss.item()
                    batch_train_s_loss += s_loss.item()
                    (m_loss + s_loss).backward()
            last_i = agent_pool.agent_count - 1
            # synchronzing fwd-bwd
            detached_features, mask_pred, sseg_pred = ddp_model(rgbs, car_transforms,
                                                                detached_features, last_i,
                                                                agent_pool.adjacency_matrix,
                                                                False)
            m_loss = mask_loss(mask_pred.squeeze(), masks[last_i])
            s_loss = torch.mean(semseg_loss(sseg_pred, labels[last_i].unsqueeze(0)) *
                                    agent_pool.combined_masks[last_i],
                                dim=(0, 1, 2))
            batch_train_m_loss += m_loss.item()
            batch_train_s_loss += s_loss.item()
            (m_loss + s_loss).backward()
            optimizer.step()
            # writing batch loss
            log_scalar(rank, batch_train_m_loss, 'loss/batch_train_msk',
                       ep * len(train_loader) + batch_idx, writer, train_cfg.logger)
            log_scalar(rank, batch_train_s_loss, 'loss/batch_train_seg',
                       ep * len(train_loader) + batch_idx, writer, train_cfg.logger)
            total_train_m_loss += batch_train_m_loss
            total_train_s_loss += batch_train_s_loss

        # syncing tensors for wandb logging ----------------------------------------------------
        # total_train_m_loss = sync_tensor(rank, 0, total_train_m_loss, train_cfg.world_size)
        # total_train_s_loss = sync_tensor(rank, 1, total_train_s_loss, train_cfg.world_size)
        # sample_count = sync_tensor(rank, 2, sample_count, train_cfg.world_size)
        log_scalar(rank, total_train_m_loss / sample_count, 'loss/total_train_msk',
                   ep + 1, writer, train_cfg.logger)
        log_scalar(rank, total_train_s_loss / sample_count, 'loss/total_train_seg',
                   ep + 1, writer, train_cfg.logger)
        log_string(rank, f'\nepoch loss: {(total_train_m_loss / sample_count).item()} mask, '
                         f'{(total_train_s_loss / sample_count).item()} segmentation')
        # validation ---------------------------------------------------------------------------
        ddp_model.eval()
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
                                                            device, True)
            agent_pool.generate_connection_strategy(ids, masks, car_transforms,
                                                    PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                    CENTER[0], CENTER[1])
            mask_preds, sseg_preds = ddp_model(rgbs, car_transforms,
                                              detached_features, -1,
                                              agent_pool.adjacency_matrix, True)
            sseg_ious += iou_per_class(sseg_preds, labels).cuda(gpu)
            mask_ious += mask_iou(mask_preds.squeeze(1), masks, train_cfg.mask_detection_thresh).item()
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
                log_image(rgbs[0],
                          f'validation/input_rgb[{rank}]',
                          ep + 1, writer, train_cfg.logger)
                log_image(torch.from_numpy(pred_mask).permute(2, 0, 1),
                          f'validation/mask_predicted[{rank}]',
                          ep + 1, writer, train_cfg.logger)
                log_image(torch.from_numpy(trgt_mask).permute(2, 0, 1),
                          f'validation/mask_target[{rank}]',
                          ep + 1, writer, train_cfg.logger)
                log_image(torch.from_numpy(ss_pred_img),
                          f'validation/segmentation_predicted[{rank}]',
                          ep + 1, writer, train_cfg.logger)
                log_image(torch.from_numpy(ss_trgt_img),
                          f'validation/segmentation_target[{rank}]',
                          ep + 1, writer, train_cfg.logger)
                visaulized = True

        new_metric = 0.0
        # total_valid_m_loss = sync_tensor(rank, 3, total_valid_m_loss, train_cfg.world_size)
        # total_valid_s_loss = sync_tensor(rank, 4, total_valid_s_loss, train_cfg.world_size)
        # sseg_ious = sync_tensor(rank, 5, sseg_ious, train_cfg.world_size)
        # mask_ious = sync_tensor(rank, 6, mask_ious, train_cfg.world_size)
        # sample_count = sync_tensor(rank, 7, sample_count, train_cfg.world_size)
        log_scalar(rank, total_valid_m_loss / sample_count, 'loss/total_valid_msk',
                   ep + 1, writer, train_cfg.logger)
        log_scalar(rank, total_valid_s_loss / sample_count, 'loss/total_valid_seg',
                   ep + 1, writer, train_cfg.logger)
        log_scalar(rank, mask_ious / sample_count, 'iou/mask_iou',
                   ep + 1, writer, train_cfg.logger)
        for key, val in segmentation_classes.items():
            log_scalar(rank, sseg_ious[key] / sample_count, f'iou/{val.lower()}_iou',
                       ep + 1, writer, train_cfg.logger)
            new_metric += sseg_ious[key] / sample_count
        new_metric += mask_ious / sample_count
        log_string(rank, f'\nepoch loss: {(total_valid_m_loss / sample_count).item()} mask, '
                         f'{(total_valid_s_loss / sample_count).item()} segmentation')

        if new_metric > last_metric and rank == 0:
            print(f'saving snapshot at epoch {ep}')
            last_metric = new_metric
            if not os.path.exists(train_cfg.snapshot_dir):
                os.makedirs(train_cfg.snapshot_dir)
            torch.save(optimizer.state_dict(), train_cfg.snapshot_dir + '/optimizer_dict')
            torch.save(ddp_model.state_dict(), train_cfg.snapshot_dir + '/model_snapshot.pth')

    if rank == 0:
        wandb.finish()


if __name__ == '__main__':
    geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    train_cfg = TrainingConfig('config/training.yml')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=train_cfg.world_size, args=(geom_cfg, train_cfg))