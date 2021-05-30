import torch
import matplotlib.pyplot as plt
import random

from data.color_map import covert_semantics_to_rgb
from data.dataset import MassHDF5
from data.mask_warp import get_single_aggregate_mask
from data.config import SemanticCloudConfig, TrainingConfig
from data.utils import squeeze_all

show_random_samples = False
# opening semantic cloud settings file
cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
train_cfg = TrainingConfig('config/training.yml')
# image geometry
NEW_SIZE = (train_cfg.output_h, train_cfg.output_w)
CENTER = (cfg.center_x(NEW_SIZE[1]), cfg.center_y(NEW_SIZE[0]))
PPM = cfg.pix_per_m(NEW_SIZE[0], NEW_SIZE[1])

# opening hdf5 file for the dataset
dset = MassHDF5(dataset=train_cfg.trainset_name, path=train_cfg.dset_dir,
                hdf5name=train_cfg.trainset_file, size=NEW_SIZE,
                classes=train_cfg.classes, jitter=train_cfg.color_jitter)
loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, num_workers=1)
# plot stuff
columns = 6
for idx, (_, rgbs, semsegs, masks, car_transforms, _) in enumerate(loader):
    # randomly skip samples (useful for large datasets)
    if show_random_samples and bool(random.randint(0, 1)):
        continue
    rgbs, semsegs, masks, car_transforms = squeeze_all(rgbs, semsegs, masks, car_transforms)
    print (f"index {idx + 1}/{len(loader)}")
    fig = plt.figure(figsize=(20, 30))
    rows = rgbs.shape[0]
    for i in range(rows):
        rgb = rgbs[i, ...].permute(1, 2, 0)
        rgb = (rgb + 1) / 2
        mask = ((masks[i, ...] / 255.0).unsqueeze(2)).numpy().squeeze()

        # create subplot and append to ax
        ax = []

        # front RGB image
        ax.append(fig.add_subplot(rows, columns, i * columns + 1))
        ax[-1].set_title(f"rgb_{i}")
        plt.imshow(rgb)

        # semantic BEV image
        ax.append(fig.add_subplot(rows, columns, i * columns + 2))
        ax[-1].set_title(f"semseg_{i}")
        semantic_img = covert_semantics_to_rgb(semsegs[i, ...], train_cfg.classes)
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
        aggregate_mask = get_single_aggregate_mask(masks, car_transforms, i, PPM, NEW_SIZE[0], NEW_SIZE[1], CENTER[0], CENTER[1])
        aggregate_mask = aggregate_mask.squeeze().numpy()
        plt.imshow(aggregate_mask)

        # aggregated mask x semantic BEV
        semantic_img[(aggregate_mask == 0).squeeze(), :] = 0
        ax.append(fig.add_subplot(rows, columns, i * columns + 6))
        ax[-1].set_title(f"agg_masked_bev_{i}")
        plt.imshow(semantic_img)

    plt.show()