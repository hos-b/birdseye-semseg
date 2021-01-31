import h5py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from data.color_map import semantic_to_cityscapes
from data.dataloader import get_dataloader
from data.mask_warp import get_aggregate_mask

DATASET_DIR = "/home/hosein"
PKG_NAME = "tp.hdf5"
PPM = 1000.0 / 25.0

image_resize = True
NEW_SIZE = (205, 256)

# opening hdf5 file for metadata
print("opening {}".format(PKG_NAME))
file_path = os.path.join(DATASET_DIR, PKG_NAME)
hdf5 = h5py.File(file_path, "r")
dataset = hdf5["dataset_1"]
agent_count = dataset.attrs["agent_count"][0]
print(f"found {(dataset.shape[0] - 1) // agent_count} samples")
print(f"agent_count attribute: {agent_count}")

# opening hdf5 file for the dataset
loader = get_dataloader(file_path, batch_size=1, train=False)

# plot stuff
rows = agent_count
columns = 4
for idx, (ids, rgbs, semsegs, masks, car_transforms) in enumerate(loader):

    print (f"index {idx + 1}/{len(loader)}")
    fig = plt.figure(figsize=(20, 30))
    # import pdb; pdb.set_trace()

    for i in range(agent_count):
        # print(car_transforms[0, i, :3, 3])
        rgb = rgbs[0, i, :, :, :].permute(1, 2, 0)
        rgb = (rgb + 1) / 2
        mask = ((masks[0, i, :, :] / 255.0).unsqueeze(2)).numpy()

        # create subplot and append to ax
        ax = []
        ax.append(fig.add_subplot(rows, columns, i * columns + 1))

        # front RGB image
        ax[-1].set_title(f"rgb_{i}")
        plt.imshow(rgb)

        # semantic BEV image
        ax.append(fig.add_subplot(rows, columns, i * columns + 2))
        ax[-1].set_title(f"semseg_{i}")
        semantic_img = semantic_to_cityscapes(semsegs[0, i, :, :])
        plt.imshow(semantic_img)

        # aggregating the masks
        ax.append(fig.add_subplot(rows, columns, i * columns + 3))
        ax[-1].set_title(f"mask_{i}")
        aggregate_mask = get_aggregate_mask(masks.squeeze(), car_transforms.squeeze(), i, PPM, 1000, 800)
        aggregate_mask = aggregate_mask.squeeze().numpy()
        # import pdb; pdb.set_trace()
        if image_resize:
            aggregate_mask = cv2.resize(aggregate_mask, NEW_SIZE, interpolation=cv2.INTER_LINEAR)
        plt.imshow(aggregate_mask)

        # masking the semantic BEV
        if image_resize:
            semantic_img = cv2.resize(semantic_img, NEW_SIZE, interpolation=cv2.INTER_LINEAR)
        semantic_img[(aggregate_mask == 0).squeeze(), :] = 0
        ax.append(fig.add_subplot(rows, columns, i * columns + 4))
        ax[-1].set_title(f"masked bev_{i}")
        plt.imshow(semantic_img)

    plt.show()

hdf5.close()