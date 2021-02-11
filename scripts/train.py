import os
import h5py
import torch
import kornia
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# from data.color_map import semantic_to_cityscapes
from data.dataloader import get_datasets
from data.config import SemanticCloudConfig

from model.mass_cnn import MassCNN


# opening semantic cloud settings file
cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
DATASET_DIR = "/home/hosein"
PKG_NAME = "tp.hdf5"
NEW_SIZE = (256, 205)


# opening hdf5 file for the dataset
file_path = os.path.join(DATASET_DIR, PKG_NAME)
train_set, test_set = get_datasets(file_path, batch_size=1, split=(0.8, 0.2), size=NEW_SIZE, classes='ours')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)


# network stuff
learning_rate = 1e-4
model = MassCNN(cfg, num_classes=10, conn_drop_prob=0, output_size=NEW_SIZE)
loss = nn.CrossEntropyLoss(reduction='mean')
epochs = 1

for ep in range(epochs):
    for batch, (_, rgbs, labels, masks, car_transforms) in enumerate(train_loader):
        mask_pred, sseg_pred = model(rgbs[0], car_transforms[0])
        import pdb; pdb.set_trace()