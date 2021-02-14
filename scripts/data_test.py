import os
import h5py
import matplotlib.pyplot as plt

from data.color_map import carla_semantics_to_cityscapes_rgb, our_semantics_to_cityscapes_rgb
from data.dataloader import get_dataloader
from data.mask_warp import get_single_aggregate_mask
from data.config import SemanticCloudConfig
from data.utils import squeeze_all

DATASET_DIR = "/home/hosein"
PKG_NAME = "tp.hdf5"


# opening semantic cloud settings file
cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')

# image geometry
NEW_SIZE = (256, 205)
CENTER = (cfg.center_x(NEW_SIZE[1]), cfg.center_y(NEW_SIZE[0]))
PPM = cfg.pix_per_m(NEW_SIZE[0], NEW_SIZE[1])

# opening hdf5 file for metadata
print("opening {}".format(PKG_NAME))
file_path = os.path.join(DATASET_DIR, PKG_NAME)
hdf5 = h5py.File(file_path, "r")
dataset = hdf5["dataset_1"]
agent_count = dataset.attrs["agent_count"][0]
print(f"found {(dataset.shape[0] - 1) // agent_count} samples")
print(f"agent_count attribute: {agent_count}")

# opening hdf5 file for the dataset
classes = 'ours'
loader = get_dataloader(file_path, batch_size=1, size=NEW_SIZE, classes=classes)

# plot stuff
rows = agent_count
columns = 6
for idx, (_, rgbs, semsegs, masks, car_transforms) in enumerate(loader):
    rgbs, semsegs, masks, car_transforms = squeeze_all(rgbs, semsegs, masks, car_transforms)
    print (f"index {idx + 1}/{len(loader)}")
    fig = plt.figure(figsize=(20, 30))
    for i in range(agent_count):
        rgb = rgbs[i, ...].permute(1, 2, 0)
        rgb = (rgb + 1) / 2
        mask = ((masks[i, ...] / 255.0).unsqueeze(2)).numpy().squeeze()

        # create subplot and append to ax
        ax = []
        ax.append(fig.add_subplot(rows, columns, i * columns + 1))

        # front RGB image
        ax[-1].set_title(f"rgb_{i}")
        plt.imshow(rgb)

        # semantic BEV image
        ax.append(fig.add_subplot(rows, columns, i * columns + 2))
        ax[-1].set_title(f"semseg_{i}")
        if classes == 'carla':
            semantic_img = carla_semantics_to_cityscapes_rgb(semsegs[i, ...])
        elif classes == 'ours':
            semantic_img = our_semantics_to_cityscapes_rgb(semsegs[i, ...])
        plt.imshow(semantic_img)

        # basic mask
        ax.append(fig.add_subplot(rows, columns, i * columns + 3))
        ax[-1].set_title(f"mask_{i}")
        plt.imshow(mask)
        
        # basic mask x semantic BEV
        ax.append(fig.add_subplot(rows, columns, i * columns + 4))
        ax[-1].set_title(f"masked_bev_{i}")
        semantic_img_cp = semantic_img.copy()
        semantic_img_cp[(mask == 0), :] = 0
        plt.imshow(semantic_img_cp)

        # aggregating the masks
        ax.append(fig.add_subplot(rows, columns, i * columns + 5))
        ax[-1].set_title(f"agg_masked_{i}")
        aggregate_mask = get_single_aggregate_mask(masks.squeeze(), car_transforms.squeeze(), i, \
                                                   PPM, NEW_SIZE[0], NEW_SIZE[1], CENTER[0], CENTER[1])
        aggregate_mask = aggregate_mask.squeeze().numpy()
        plt.imshow(aggregate_mask)

        # aggregated mask x semantic BEV
        semantic_img[(aggregate_mask == 0).squeeze(), :] = 0
        ax.append(fig.add_subplot(rows, columns, i * columns + 6))
        ax[-1].set_title(f"agg_masked_bev_{i}")
        plt.imshow(semantic_img)

    plt.show()

hdf5.close()