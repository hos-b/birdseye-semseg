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
        self.train = kwargs.get("train", True)
        self.transform = kwargs.get("transform")
        self.file_path = kwargs.get("file_path")
        print(f"opening {self.file_path}")
        self.hdf5 = h5py.File(self.file_path, "r")        
        self.dataset = self.hdf5["dataset_1"]
        self.n_samples = self.dataset.shape[0]
        self.agent_count = self.dataset.attrs["agent_count"][0]

    def __getitem__(self, idx):
        ids = []
        rgbs = []
        semsegs = []
        masks = []
        car_transforms = []
        for i in range(self.agent_count):    
            ids.append(torch.tensor([self.dataset[idx * self.agent_count + i, "agent_id"]], dtype=torch.long))
            # H, W, C
            rgbs.append(self.transform(self.dataset[idx * self.agent_count + i, "front_rgb"]
                    .view(dtype=np.uint8).reshape(480, 640, 4)[:, :, [2, 1, 0]])) # BGR to RGB
            # H, W
            semsegs.append(torch.tensor(self.dataset[idx * self.agent_count + i, "top_semseg"]
                    .view(dtype=np.uint8).reshape(1000, 800), dtype=torch.uint8))
            # H, W
            masks.append(torch.tensor(self.dataset[idx * self.agent_count + i, "top_mask"]
                    .view(dtype=np.uint8).reshape(1000, 800), dtype=torch.float32))
            # 4 x 4
            car_transforms.append(torch.tensor(self.dataset[idx * self.agent_count + i, "transform"]
                    .view(dtype=np.float64).reshape(4, 4), dtype=torch.float64).transpose(0, 1))
        return torch.stack(ids), torch.stack(rgbs), torch.stack(semsegs), \
                                 torch.stack(masks), torch.stack(car_transforms)
        
    def __len__(self):
        return int((self.n_samples - 1) / self.agent_count)

def get_dataloader(file_path, batch_size=1, train=False):
    transform_list = []
    transform_list.append(transforms.ToTensor())
    if train :
        transform_list.append(transforms.ColorJitter(hue=.05, saturation=.05))
    # transform_list.append(transforms.Resize(256))
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)
    reader = MassHDF5(train=train, transform=transform, file_path=file_path)
    data_loader = torch.utils.data.DataLoader(reader, batch_size=batch_size, shuffle=train,
                                              num_workers=4 if train else 1)
    return data_loader