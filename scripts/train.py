import os
import cv2
import torch
import torch.nn as nn
import torch.functional as F
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

# from data.color_map import semantic_to_cityscapes
from data.dataloader import get_datasets
from data.config import SemanticCloudConfig
from data.color_map import our_semantics_to_cityscapes_rgb, carla_semantic_to_cityscapes_rgb
from model.mass_cnn import MassCNN

# opening semantic cloud settings file
cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
DATASET_DIR = '/home/hosein'
TENSORBOARD_DIR = './tensorboard'
PKG_NAME = "tp.hdf5"
NEW_SIZE = (256, 205)

# dataset
device = torch.device('cpu')
file_path = os.path.join(DATASET_DIR, PKG_NAME)
train_set, test_set = get_datasets(file_path, device=device, batch_size=1, split=(0.95, 0.05), size=NEW_SIZE, classes='ours')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

# logging
# TODO: get name from commit id
name = 'test_run'
writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, name))

# network stuff
learning_rate = 5e-4
model = MassCNN(cfg, num_classes=7, conn_drop_prob=0, output_size=NEW_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
semseg_loss = nn.CrossEntropyLoss(reduction='mean')
mask_loss = nn.L1Loss(reduction='mean')
epochs = 1


for ep in range(epochs):
    total_train_m_loss = 0.0
    total_train_s_loss = 0.0
    total_valid_m_loss = 0.0
    total_valid_s_loss = 0.0
    # training
    # model.train()
    # for batch_idx, (_, rgbs, labels, masks, car_transforms) in enumerate(train_loader):
    #     print(f'\repoch: {ep}/{epochs}, training batch: {batch_idx} / {len(train_loader)}', end='')
    #     optimizer.zero_grad()
    #     mask_preds, sseg_preds = model(rgbs[0], car_transforms[0])
    #     m_loss = mask_loss(mask_preds.squeeze(), masks.squeeze())
    #     s_loss = semseg_loss(sseg_preds, labels.squeeze())
    #     (m_loss + s_loss).backward()
    #     optimizer.step()
    #     batch_train_m_loss = m_loss.item()
    #     batch_train_s_loss = s_loss.item()
    #     writer.add_scalar("loss/batch_train_msk", batch_train_m_loss, ep * len(train_loader) + batch_idx)
    #     writer.add_scalar("loss/batch_train_seg", batch_train_s_loss, ep * len(train_loader) + batch_idx)
    #     total_train_m_loss += batch_train_m_loss
    #     total_train_s_loss += batch_train_s_loss
    #     break

    # writer.add_scalar("\nloss/total_train_msk", total_train_m_loss, ep + 1)
    # writer.add_scalar("loss/total_train_seg", total_train_s_loss, ep + 1)
    # print(f'epoch loss: {total_train_m_loss} mask, {total_train_s_loss} segmentation')

    # validation
    model.eval()
    visaulized = False
    with torch.no_grad():
        for batch_idx, (_, rgbs, labels, masks, car_transforms) in enumerate(test_loader):
            print(f'\repoch: {ep}/{epochs}, validation batch: {batch_idx} / {len(test_loader)}', end='')
            mask_preds, sseg_preds = model(rgbs[0], car_transforms[0])
            # m_loss = mask_loss(mask_preds.squeeze(), masks.squeeze())
            # s_loss = semseg_loss(sseg_preds, labels.squeeze())
            batch_valid_m_loss = 0.0 # m_loss.item()
            batch_valid_s_loss = 0.0 # s_loss.item()
            writer.add_scalar("\nloss/batch_valid_msk", batch_valid_m_loss, ep * len(test_loader) + batch_idx)
            writer.add_scalar("loss/batch_valid_seg", batch_valid_s_loss, ep * len(test_loader) + batch_idx)
            total_valid_m_loss += batch_valid_m_loss
            total_valid_s_loss += batch_valid_s_loss
            # visaluize the first agent from the first batch
            if not visaulized:
                # plt.imshow(mask_preds.squeeze()[0].numpy())
                # plt.show()
                # plt.imshow(masks.squeeze()[0].numpy())
                # plt.show()
                _, sseg_pred = torch.max(sseg_preds[0], dim=0)
                # plt.imshow(our_semantics_to_cityscapes_rgb(sseg_pred))
                # plt.show()
                # plt.imshow(our_semantics_to_cityscapes_rgb(labels.squeeze()[0]))
                # plt.show()
                # import pdb; pdb.set_trace()
                writer.add_image("validation/predicted_mask", mask_preds[0], ep + 1)
                writer.add_image("validation/target_mask", masks[0, 0].unsqueeze(0), ep + 1)
                writer.add_image("validation/predicted_segmentation",
                    torch.from_numpy(our_semantics_to_cityscapes_rgb(sseg_pred)).permute(2, 0, 1), ep + 1)
                writer.add_image("validation/target_segmentation", 
                    torch.from_numpy(our_semantics_to_cityscapes_rgb(labels.squeeze()[0])).permute(2, 0, 1), ep + 1)
                visaulized = True

    writer.add_scalar("loss/total_valid_msk", total_valid_m_loss, ep + 1)
    writer.add_scalar("loss/total_valid_seg", total_valid_s_loss, ep + 1)
    print(f'epoch loss: {total_valid_m_loss} mask, {total_valid_s_loss} segmentation')

writer.close()