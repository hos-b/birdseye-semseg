import h5py
import numpy as np

import cv2
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image as PILImage
from data.utils import separate_masks
from data.color_map import convert_semantic_classes


class MassHDF5(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.path = kwargs.get('path')
        self.hdf5name = kwargs.get('hdf5name')
        self.full_path = self.path + '/' + self.hdf5name
        self.size = kwargs.get('size')
        self.jitter = kwargs.get('jitter')
        self.dset_name = kwargs.get('dataset', 'town-01')
        self.classes = kwargs.get('classes')
        print(f'dataset file: {self.full_path}')
        try:
            self.hdf5 = h5py.File(self.full_path, 'r')
        except:
            print('could not open hdf5 file for reading')
            exit()
        self.dataset = self.hdf5[self.dset_name]
        self.min_agent_count = self.dataset.attrs['min_agent_count'][0]
        self.max_agent_count = self.dataset.attrs['max_agent_count'][0]
        self.n_samples = self.get_dataset_size()
        self.batch_indices, self.batch_sizes = self.get_batch_info(50100)
        print(f'found {self.n_samples} samples in {self.batch_sizes.shape[0]} batches')
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=self.jitter[0], contrast=self.jitter[1],
                                   hue=self.jitter[2], saturation=self.jitter[3]),
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

        rgbs = []
        semsegs = []
        masks = []
        car_transforms = []
        b_agent_count = self.batch_sizes[idx]
        b_start_idx = self.batch_indices[idx]
        for i in range(b_agent_count):
            # RGB Image: H, W, C
            rgbs.append(self.rgb_transform(self.dataset[b_start_idx + i, 'front_rgb']
                    .view(dtype=np.uint8).reshape(480, 640, 4)[:, :, [2, 1, 0]])) # BGR to RGB
            # Masks: H, W
            masks.append(self.mask_transform(self.dataset[b_start_idx + i, 'top_mask']
                        .view(dtype=np.uint8).reshape(500, 400, 1)).squeeze())
            # Semantic Label: H, W
            semseg = self.dataset[b_start_idx + i, 'top_semseg'] .view(dtype=np.uint8).reshape(500, 400)
            # change semantic labels to a subset
            semseg = convert_semantic_classes(semseg, self.classes, masks[-1])
            # opencv size is (width, height), instead of (rows, cols)
            semseg = cv2.resize(semseg, dsize=self.size[::-1], interpolation=cv2.INTER_NEAREST)
            semsegs.append(torch.tensor(semseg, dtype=torch.long))
            # Car Transforms: 4 x 4
            car_transforms.append(torch.tensor(self.dataset[b_start_idx + i, 'transform']
                    .view(dtype=np.float64).reshape(4, 4), dtype=torch.float64).transpose(0, 1))
        # cut the mask into two separate tensors
        veh_masks, fov_masks = separate_masks(torch.stack(masks))
        return torch.stack(rgbs), torch.stack(semsegs), veh_masks, fov_masks, \
               torch.stack(car_transforms), torch.LongTensor([idx])

    def __len__(self):
        return self.batch_sizes.shape[0]
    
    def get_dataset_size(self):
        """
        a sanity check to make sure that the correct number of 
        frames have been recorded. returns the double checked
        total number of samples.
        """
        total_samples = 0
        batch_histogram = self.dataset.attrs['batch_histogram']
        for i in range(batch_histogram.shape[0]):
            total_samples += (i + self.min_agent_count) * batch_histogram[i]
        
        assert total_samples == self.dataset.shape[0] - 1, 'unexpected number of samples'
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