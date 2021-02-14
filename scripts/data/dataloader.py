import h5py
import numpy as np

import cv2
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image as PILImage

from data.color_map import carla_semantics_to_our_semantics

class MassHDF5(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.file_path = kwargs.get('file_path')
        self.size = kwargs.get('size')
        self.device = kwargs.get('device')
        clss = kwargs.get('classes')
        self.use_class_subset = kwargs.get('classes') == 'ours'
        print(f'opening {self.file_path}')
        self.hdf5 = h5py.File(self.file_path, 'r')
        self.dataset = self.hdf5['dataset_1']
        self.n_samples = self.dataset.shape[0]
        self.agent_count = self.dataset.attrs['agent_count'][0]
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size, interpolation=PILImage.NEAREST),
            transforms.ToTensor()
        ])

        self.hdf5.close()
        self.hdf5 = None
        self.dataset = None

    def __getitem__(self, idx):
        if self.dataset == None:
            self.hdf5 = h5py.File(self.file_path, 'r')
            self.dataset = self.hdf5['dataset_1']

        ids = []
        rgbs = []
        semsegs = []
        masks = []
        car_transforms = []
        for i in range(self.agent_count):
            # Agent ID: never used
            ids.append(torch.tensor([self.dataset[idx * self.agent_count + i, "agent_id"]], dtype=torch.long))
            # RGB Image: H, W, C
            rgbs.append(self.rgb_transform(self.dataset[idx * self.agent_count + i, "front_rgb"]
                    .view(dtype=np.uint8).reshape(480, 640, 4)[:, :, [2, 1, 0]])) # BGR to RGB
            # Semantic Label: H, W
            semseg = self.dataset[idx * self.agent_count + i, "top_semseg"] .view(dtype=np.uint8).reshape(1000, 800)
            if self.use_class_subset:
                semseg = carla_semantics_to_our_semantics(semseg)
            # opencv size is (width, height), instead of (rows, cols)
            semseg = cv2.resize(semseg, dsize=self.size[::-1], interpolation=cv2.INTER_NEAREST)
            semsegs.append(torch.tensor(semseg, dtype=torch.long))
            # Masks: H, W
            masks.append(self.mask_transform(self.dataset[idx * self.agent_count + i, "top_mask"]
                        .view(dtype=np.uint8).reshape(1000, 800)).squeeze())
            # Car Transforms: 4 x 4
            car_transforms.append(torch.tensor(self.dataset[idx * self.agent_count + i, "transform"]
                    .view(dtype=np.float64).reshape(4, 4), dtype=torch.float64).transpose(0, 1))
        return torch.stack(ids).to(self.device), torch.stack(rgbs).to(self.device), \
               torch.stack(semsegs).to(self.device), torch.stack(masks).to(self.device), \
               torch.stack(car_transforms).to(self.device)

    def __len__(self):
        return int((self.n_samples - 1) / self.agent_count)


def get_datasets(file_path, device, batch_size, split=(0.8, 0.2), size=(1000, 800), classes='carla'):
    if classes != 'carla' and classes != 'ours':
        print("unknown segmentation class category: {classes}")
        classes = 'carla'
    dset = MassHDF5(file_path=file_path, size=size, classes=classes, device=device)
    return torch.utils.data.random_split(dset, [int(split[0] * len(dset)), int(split[1] * len(dset))])

def get_dataloader(file_path, batch_size=1, size=(1000, 800), classes='carla'):
    dset = MassHDF5(file_path=file_path, size=size, classes=classes)
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=1)