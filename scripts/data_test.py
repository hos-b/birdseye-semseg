import h5py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from data.color_map import semantic_to_cityscapes
from data.dataloader import get_dataloader

DATASET_DIR = "/home/hosein"
PKG_NAME = "async.hdf5"

print("opening {}".format(PKG_NAME))
file_path = os.path.join(DATASET_DIR, PKG_NAME)
hdf5 = h5py.File(file_path, "r")
dataset = hdf5["dataset_1"]
agent_count = dataset.attrs["agent_count"][0]

print(f"found {(dataset.shape[0] - 1) // agent_count} samples")
print(f"agent_count attribute: {agent_count}")

loader = get_dataloader(file_path, batch_size=1, train=False)
# plot stuff
rows = agent_count
columns = 4
for idx, (ids, rgbs, semsegs, masks, car_transforms) in enumerate(loader):

    print (f"index {idx + 1}/{len(loader)}")
    fig = plt.figure(figsize=(20, 30))

    for i in range(agent_count):
        # print(car_transforms[0, i, :3, 3])
        rgb = rgbs[0, i, :, :, :].permute(1, 2, 0)
        rgb = (rgb + 1) / 2
        mask = ((masks[0, i, :, :] / 255.0).unsqueeze(2)).numpy()

        ax = []
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i * columns + 1))
        ax[-1].set_title(f"rgb_{i}")
        plt.imshow(rgb)

        ax.append(fig.add_subplot(rows, columns, i * columns + 2))
        ax[-1].set_title(f"semseg_{i}")
        semantic_img = semantic_to_cityscapes(semsegs[0, i, :, :])
        plt.imshow(semantic_img)

        ax.append(fig.add_subplot(rows, columns, i * columns + 3))
        ax[-1].set_title(f"mask_{i}")
        # import pdb; pdb.set_trace()
        plt.imshow(mask.squeeze())

        masked_bev = semantic_img.copy()
        masked_bev[(mask == 0).squeeze(), :] = 0
        ax.append(fig.add_subplot(rows, columns, i * columns + 4))
        ax[-1].set_title(f"masked bev_{i}")
        plt.imshow(masked_bev)

    plt.show()

hdf5.close()