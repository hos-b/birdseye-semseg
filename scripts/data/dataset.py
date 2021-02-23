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
        self.dset_name = kwargs.get('dataset', 'town-01')
        self.use_class_subset = kwargs.get('classes') == 'ours'
        print(f'opening {self.file_path}')
        self.hdf5 = h5py.File(self.file_path, 'r')
        self.dataset = self.hdf5[self.dset_name]
        self.min_agent_count = self.dataset.attrs['min_agent_count'][0]
        self.max_agent_count = self.dataset.attrs['max_agent_count'][0]
        self.n_samples = self.get_dataset_size()
        self.batch_indices, self.batch_sizes = self.parse_batch_indices()
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
            self.dataset = self.hdf5[self.dset_name]

        ids = []
        rgbs = []
        semsegs = []
        masks = []
        car_transforms = []
        b_agent_count = self.batch_sizes[idx]
        b_start_idx = self.batch_indices[idx]
        for i in range(b_agent_count):
            # Agent ID: never used
            ids.append(torch.tensor([self.dataset[b_start_idx + i, "agent_id"]], dtype=torch.long))
            # RGB Image: H, W, C
            rgbs.append(self.rgb_transform(self.dataset[b_start_idx + i, "front_rgb"]
                    .view(dtype=np.uint8).reshape(480, 640, 4)[:, :, [2, 1, 0]])) # BGR to RGB
            # Semantic Label: H, W
            semseg = self.dataset[b_start_idx + i, "top_semseg"] .view(dtype=np.uint8).reshape(500, 400)
            if self.use_class_subset:
                semseg = carla_semantics_to_our_semantics(semseg)
            # opencv size is (width, height), instead of (rows, cols)
            semseg = cv2.resize(semseg, dsize=self.size[::-1], interpolation=cv2.INTER_NEAREST)
            semsegs.append(torch.tensor(semseg, dtype=torch.long))
            # Masks: H, W
            masks.append(self.mask_transform(self.dataset[b_start_idx + i, "top_mask"]
                        .view(dtype=np.uint8).reshape(500, 400)).squeeze())
            # Car Transforms: 4 x 4
            car_transforms.append(torch.tensor(self.dataset[b_start_idx + i, "transform"]
                    .view(dtype=np.float64).reshape(4, 4), dtype=torch.float64).transpose(0, 1))
        return torch.stack(ids).to(self.device), torch.stack(rgbs).to(self.device), \
               torch.stack(semsegs).to(self.device), torch.stack(masks).to(self.device), \
               torch.stack(car_transforms).to(self.device)

    def __len__(self):
        return self.batch_sizes.shape[0]
    
    def get_dataset_size(self):
        """
        a sanity check to make sure that the correct number of 
        frames have been recorded. returns the double checked
        total number of samples.
        """
        total_samples = 0
        # for some reason the attribute is saved in reverse!
        batch_histogram = np.flip(self.dataset.attrs['batch_histogram'])
        for i in range(batch_histogram.shape[0]):
            total_samples += (i + self.min_agent_count) * batch_histogram[i]
        
        assert(total_samples == self.dataset.shape[0] - 1, "unexpected number of samples")
        return total_samples

    def parse_batch_indices(self):
        """
        iterate the dataset to find the start of each batch.
        agent ids go 0 1 ... 0 1 2 3 ... 0 1 ...
        """
        batch_start_indices = []
        batch_sizes = []
        for i in range(self.n_samples):
            if self.dataset[i, 'agent_id'] == 0:
                batch_start_indices.append(i)
                if len(batch_start_indices) > 1:
                    batch_sizes.append(batch_start_indices[-1] - batch_start_indices[-2])
        batch_sizes.append(self.n_samples - batch_start_indices[-1])
        return np.array(batch_start_indices, dtype=np.uint), \
               np.array(batch_sizes, dtype=np.uint)

def get_datasets(file_path, device, split=(0.8, 0.2), size=(500, 400), classes='carla'):
    if classes != 'carla' and classes != 'ours':
        print("unknown segmentation class category: {classes}")
        classes = 'carla'
    dset = MassHDF5(file_path=file_path, size=size, classes=classes, device=device)
    return torch.utils.data.random_split(dset, [int(split[0] * len(dset)), int(split[1] * len(dset))])