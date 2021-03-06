import os
import torch
import torch.nn as nn
import torch.functional as F
from tensorboardX import SummaryWriter

import subprocess
import matplotlib.pyplot as plt

from data.dataset import get_datasets
from data.config import SemanticCloudConfig
from data.color_map import our_semantics_to_cityscapes_rgb
from data.mask_warp import get_all_aggregate_masks
from data.utils import drop_agent_data, squeeze_all, get_matplotlib_image
from model.mass_cnn import MassCNN
from agent.agent_pool import AgentPool

def to_device(rgbs, labels, masks, car_transforms, device):
    return rgbs.to(device), labels.to(device), \
           masks.to(device), car_transforms.to(device)

# opening semantic cloud settings file
cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
DATASET_DIR = '/export/home/aiscar2/mass-data'
PKG_NAME = 'dataset_10k.hdf5'
DATASET = 'town-01'
TENSORBOARD_DIR = './tensorboard'
NEW_SIZE = (256, 205)

# image size and center coordinates
CENTER = (cfg.center_x(NEW_SIZE[1]), cfg.center_y(NEW_SIZE[0]))
PPM = cfg.pix_per_m(NEW_SIZE[0], NEW_SIZE[1])

# dataset
device = torch.device('cuda')
train_set, test_set = get_datasets(DATASET, DATASET_DIR, PKG_NAME, (0.8, 0.2), NEW_SIZE, 'ours')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4)

# logging
name = 'initial-'
name += subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8')[:-1]
writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, name))

# network stuff
drop_prob = 0.0
learning_rate = 5e-4
model = MassCNN(cfg, num_classes=7, output_size=NEW_SIZE).to(device)
agent_pool = AgentPool(model, device, NEW_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
semseg_loss = nn.CrossEntropyLoss(reduction='none')
mask_loss = nn.L1Loss(reduction='none')
epochs = 100
print(f"{(model.parameter_count() / 1e6):.2f}M trainable parameters")

for ep in range(epochs):
    total_train_m_loss = 0.0
    total_train_s_loss = 0.0
    total_valid_m_loss = 0.0
    total_valid_s_loss = 0.0
    # training
    model.train()
    for batch_idx, (_, rgbs, labels, masks, car_transforms) in enumerate(train_loader):
        print(f'\repoch: {ep + 1}/{epochs}, training batch: {batch_idx + 1} / {len(train_loader)}', end='')
        rgbs, labels, masks, car_transforms = to_device(rgbs, labels, masks, car_transforms, device)
        # simulate connection drops
        rgbs, labels, masks, car_transforms = drop_agent_data(rgbs, labels, masks, car_transforms, drop_prob)
        # masked loss
        aggregate_masks = get_all_aggregate_masks(masks, car_transforms, PPM, NEW_SIZE[0], \
                                                  NEW_SIZE[1], CENTER[0], CENTER[1])
        optimizer.zero_grad()
        agent_pool.calculate_detached_messages(rgbs)
        for i in range(agent_pool.agent_count):
            mask_pred = agent_pool.calculate_agent_mask(rgbs[i])
            sseg_pred = agent_pool.aggregate_messages(i, car_transforms)
            m_loss = torch.mean(mask_loss(mask_pred.squeeze(), masks[i]) * aggregate_masks[i], dim=(0, 1))
            s_loss = torch.mean(semseg_loss(sseg_pred, labels[i].unsqueeze(0)) * masks[i], dim=(0, 1, 2))
            (m_loss + s_loss).backward()
            optimizer.step()
            batch_train_m_loss = m_loss.item()
            batch_train_s_loss = s_loss.item()
        writer.add_scalar("loss/batch_train_msk", batch_train_m_loss, ep * len(train_loader) + batch_idx)
        writer.add_scalar("loss/batch_train_seg", batch_train_s_loss, ep * len(train_loader) + batch_idx)
        total_train_m_loss += batch_train_m_loss
        total_train_s_loss += batch_train_s_loss

    writer.add_scalar("loss/total_train_msk", total_train_m_loss / len(train_loader), ep + 1)
    writer.add_scalar("loss/total_train_seg", total_train_s_loss / len(train_loader), ep + 1)
    print(f'\nepoch loss: {total_train_m_loss / len(train_loader)} mask, '
                        f'{total_train_s_loss / len(train_loader)} segmentation')

    # validation
    model.eval()
    visaulized = False
    with torch.no_grad():
        for batch_idx, (_, rgbs, labels, masks, car_transforms) in enumerate(test_loader):
            print(f'\repoch: {ep + 1}/{epochs}, validation batch: {batch_idx + 1} / {len(test_loader)}', end='')
            rgbs, labels, masks, car_transforms = squeeze_all(rgbs, labels, masks, car_transforms)
            rgbs, labels, masks, car_transforms = to_device(rgbs, labels, masks, car_transforms, device)
            mask_preds, sseg_preds = model(rgbs, car_transforms)
            m_loss = mask_loss(mask_preds.squeeze(), masks.squeeze())
            s_loss = semseg_loss(sseg_preds, labels)
            batch_valid_m_loss = torch.mean(m_loss.view(1, -1)).item()
            batch_valid_s_loss = torch.mean(s_loss.view(1, -1)).item()
            writer.add_scalar("loss/batch_valid_msk", batch_valid_m_loss, ep * len(test_loader) + batch_idx)
            writer.add_scalar("loss/batch_valid_seg", batch_valid_s_loss, ep * len(test_loader) + batch_idx)
            total_valid_m_loss += batch_valid_m_loss
            total_valid_s_loss += batch_valid_s_loss

            # visaluize the first agent from the first batch
            if not visaulized:
                aggregate_masks = get_all_aggregate_masks(masks, car_transforms, PPM, NEW_SIZE[0], \
                                                          NEW_SIZE[1], CENTER[0], CENTER[1])
                ss_trgt_img = our_semantics_to_cityscapes_rgb(labels[0].cpu()).transpose(2, 0, 1)
                ss_mask = aggregate_masks[0].cpu()
                ss_trgt_img[:, ss_mask == 0] = 0
                _, ss_pred = torch.max(sseg_preds[0], dim=0)
                ss_pred_img = our_semantics_to_cityscapes_rgb(ss_pred.cpu()).transpose(2, 0, 1)
                pred_mask_img = get_matplotlib_image(mask_preds[0].squeeze().cpu())
                trgt_mask_img = get_matplotlib_image(masks[0].cpu())
                writer.add_image("validation/predicted_mask", torch.from_numpy(pred_mask_img).permute(2, 0, 1), ep + 1)
                writer.add_image("validation/target_mask", torch.from_numpy(trgt_mask_img).permute(2, 0, 1), ep + 1)
                writer.add_image("validation/predicted_segmentation", ss_pred_img, ep + 1)
                writer.add_image("validation/target_segmentation", torch.from_numpy(ss_trgt_img), ep + 1)
                visaulized = True

    writer.add_scalar("loss/total_valid_msk", total_valid_m_loss / len(test_loader), ep + 1)
    writer.add_scalar("loss/total_valid_seg", total_valid_s_loss / len(test_loader), ep + 1)
    print(f'\nepoch loss: {total_valid_m_loss / len(test_loader)} mask, {total_valid_s_loss / len(test_loader)} segmentation')

writer.close()