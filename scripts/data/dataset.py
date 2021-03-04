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
        self.path = kwargs.get('path')
        self.hdf5name = kwargs.get('hdf5name')
        self.full_path = self.path + '/' + self.hdf5name
        self.size = kwargs.get('size')
        self.device = kwargs.get('device')
        self.dset_name = kwargs.get('dataset', 'town-01')
        self.use_class_subset = kwargs.get('classes') == 'ours'
        print(f'opening {self.full_path}')
        self.hdf5 = h5py.File(self.full_path, 'r')
        self.dataset = self.hdf5[self.dset_name]
        self.min_agent_count = self.dataset.attrs['min_agent_count'][0]
        self.max_agent_count = self.dataset.attrs['max_agent_count'][0]
        self.n_samples = self.get_dataset_size()
        self.batch_indices, self.batch_sizes = self.get_batch_info()
        print(f"found {self.n_samples} samples in {self.batch_sizes.shape[0]} batches")
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
            self.hdf5 = h5py.File(self.full_path, 'r')
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
                        .view(dtype=np.uint8).reshape(500, 400, 1)).squeeze())
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
        
        assert total_samples == self.dataset.shape[0] - 1, "unexpected number of samples"
        return total_samples
    
    def get_batch_info(self, estimate_batch_count=11000):
        """
        tries to read batch start indices and batch sizes from the metadata file.
        if file doesn't exist, it parses the dataset and creates the file.
        """
        md_name = self.hdf5name.replace('hdf5', 'metadata')
        # try to read from file
        try:
            fmeta = open(self.path + '/' + md_name, mode='rb')
        except FileNotFoundError:
            print('could not locate metadata file for the dataset. generating...')
            return self._parse_batch_indices(md_name, estimate_batch_count)
        metadata = np.fromfile(fmeta, dtype = np.uint32)
        half = int(metadata.shape[0] / 2)
        fmeta.close()
        return metadata[:half], metadata[half:]

    def _parse_batch_indices(self, filename: str, estimate_batch_count=11000):
        """
        iterates the dataset to find the start of each batch.
        agent ids go 0 1 ... 0 1 2 3 ... 0 1 ...
        """
        # np arrays & indices to avoid heavy append() calls
        batch_start_indices = np.zeros(shape=(estimate_batch_count), dtype=np.uint32)
        batch_start_indices_c = 0
        batch_sizes = np.zeros(shape=(estimate_batch_count), dtype=np.uint32)
        batch_sizes_c = 0
        # manually adding the first batch to avoid an extra if in the loop
        batch_start_indices[0] = 0
        batch_start_indices_c += 1
        # loop the rest
        for i in range(1, self.n_samples):
            if self.dataset[i, 'agent_id'] == 0:
                batch_sizes[batch_sizes_c] = i - batch_start_indices[batch_start_indices_c - 1]
                batch_sizes_c += 1
                batch_start_indices[batch_start_indices_c] = i
                batch_start_indices_c += 1
        batch_sizes[batch_sizes_c] = self.n_samples - batch_start_indices[batch_start_indices_c - 1]
        batch_sizes_c += 1
        # resize arrays
        batch_sizes = batch_sizes[:batch_sizes_c]
        batch_start_indices = batch_start_indices[:batch_start_indices_c]
        # write the result to a file
        metadata = np.append(batch_start_indices, batch_sizes)
        try:
            fmeta = open(self.path + '/' + filename, mode='wb')
            metadata.tofile(fmeta)
            fmeta.close()
            print(f'successfully wrote metadata to file {self.path}/{filename}')
        except:
            print(f'could not write metadata to file {self.path}/{filename}')
        return batch_start_indices, batch_sizes

def get_datasets(dataset, path, hdf5name, device, split=(0.8, 0.2), size=(500, 400), classes='carla'):
    if classes != 'carla' and classes != 'ours':
        print("unknown segmentation class category: {classes}, using 'carla'")
        classes = 'carla'
    dset = MassHDF5(dataset=dataset, path=path, hdf5name=hdf5name,
                    size=size, classes=classes, device=device)
    return torch.utils.data.random_split(dset, [int(split[0] * len(dset)), int(split[1] * len(dset))])