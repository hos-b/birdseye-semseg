import h5py
import numpy as np
import os

import torch
import torch.utils.data
import torchvision.transforms as transforms

#pylint: disable=E1101
#pylint: disable=not-callable

DATASET_DIR = "/home/hosein"
PKG_NAME = "test.hdf5"

class MassHDF5(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.rgb_transform = kwargs.get("transform")
        self.file_path = kwargs.get("file_path")
        self.size = kwargs.get("size")
        print(f"opening {self.file_path}")
        self.hdf5 = h5py.File(self.file_path, "r")        
        self.dataset = self.hdf5["dataset_1"]
        self.n_samples = self.dataset.shape[0]
        self.agent_count = self.dataset.attrs["agent_count"][0]
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        ids = []
        rgbs = []
        semsegs = []
        masks = []
        car_transforms = []
        for i in range(self.agent_count):    
            ids.append(torch.tensor([self.dataset[idx * self.agent_count + i, "agent_id"]], dtype=torch.long))
            # H, W, C
            rgbs.append(self.rgb_transform(self.dataset[idx * self.agent_count + i, "front_rgb"]
                    .view(dtype=np.uint8).reshape(480, 640, 4)[:, :, [2, 1, 0]])) # BGR to RGB
            # H, W
            semsegs.append(torch.tensor(self.dataset[idx * self.agent_count + i, "top_semseg"]
                    .view(dtype=np.uint8).reshape(1000, 800), dtype=torch.uint8))
            # H, W
            masks.append(self.mask_transform(self.dataset[idx * self.agent_count + i, "top_mask"]
                        .view(dtype=np.uint8).reshape(1000, 800)))
            # 4 x 4
            car_transforms.append(torch.tensor(self.dataset[idx * self.agent_count + i, "transform"]
                    .view(dtype=np.float64).reshape(4, 4), dtype=torch.float64).transpose(0, 1))
        return torch.stack(ids), torch.stack(rgbs), torch.stack(semsegs), \
                                 torch.stack(masks), torch.stack(car_transforms)
        
    def __len__(self):
        return int((self.n_samples - 1) / self.agent_count)


def get_datasets(file_path, batch_size, split=(0.8, 0.2), size=(1000, 800)):
    reader = MassHDF5(file_path=file_path, size=size)
    return torch.utils.data.random_split(reader, [int(split[0] * len(reader)),
                                                  int(split[1] * len(reader))])

def get_dataloader(file_path, batch_size=1):
    transform_list = []
    transform_list.append(transforms.ToTensor())
    # transform_list.append(transforms.ColorJitter(hue=.05, saturation=.05))
    # transform_list.append(transforms.Resize(256))
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)
    reader = MassHDF5(transform=transform, file_path=file_path, size=None)
    data_loader = torch.utils.data.DataLoader(reader, batch_size=batch_size, shuffle=False,
                                              num_workers=1)
    return data_loader