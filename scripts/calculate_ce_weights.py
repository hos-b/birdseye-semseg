import torch

from data.config import SemanticCloudConfig, TrainingConfig
from data.dataset import MassHDF5
from data.utils import to_device


def main():
    # parsing config file
    geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    train_cfg = TrainingConfig('config/training.yml')
    # gpu selection ------------------------------------------------------------------
    device_str = train_cfg.device
    if train_cfg.device == 'cuda':
        torch.cuda.set_device(0)
        device_str += f':{0}'
    device = torch.device(device_str)
    torch.manual_seed(train_cfg.torch_seed)
    # dataset ------------------------------------------------------------------------
    new_size = (train_cfg.output_h, train_cfg.output_w)
    train_set = MassHDF5(dataset=train_cfg.trainset_name, path=train_cfg.dset_dir,
                         hdf5name=train_cfg.trainset_file, size=new_size,
                         classes=train_cfg.classes, jitter=train_cfg.color_jitter)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=train_cfg.shuffle_data,
                                               num_workers=train_cfg.loader_workers)
    # start counting -----------------------------------------------------------------
    class_counts = torch.Tensor(size=(train_cfg.num_classes,)).long().to(device)
    loader_length = len(train_loader)
    for batch_idx, (rgbs, labels, car_masks, fov_masks, tfs, _) in enumerate(train_loader):
        masks = car_masks + fov_masks
        print(f'\rprocessing batch {batch_idx}/{loader_length}', end='', flush=True)
        batch_size = rgbs.shape[1]
        _, labels, masks, _ = to_device(rgbs, labels, masks, tfs, device)
        for i in range(batch_size):
            semantics = labels[0, i]
            mask = masks[0, i]
            uniq_indices, uniq_counts = semantics[mask == 1].unique(sorted=True, return_counts=True)
            for uidx, ucount in zip(uniq_indices, uniq_counts):
                class_counts[uidx] += ucount

    print('done')
    print(f'counts: \n{class_counts}')
    print(f'max weights: \n{class_counts.max() / class_counts.double()}')
    print(f'median weights: \n{class_counts.median() / class_counts.double()}')


if __name__ == '__main__':
    main()