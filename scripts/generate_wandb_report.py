import os
import torch
import numpy as np
import wandb
from agent.agent_pool import CurriculumPool

from model.large_mcnn import TransposedMCNN, MaxoutMCNNT
from model.noisy_mcnn import NoisyMCNN
from data.config import SemanticCloudConfig, ReportConfig
from data.dataset import MassHDF5
from data.utils import to_device
from evaluate import plot_batch

def get_noisy_transforms(transforms: torch.Tensor, dx_std, dy_std, th_std) -> torch.Tensor:
    """
    return a noisy version of the transforms given the noise parameters
    """
    batch_size = transforms.shape[0]
    se2_noise = torch.zeros_like(transforms)
    if th_std != 0.0:
        rand_t = (torch.normal(mean=0.0, std=th_std, size=(1,)) * (np.pi / 180.0)).repeat(batch_size)
        se2_noise[:, 0, 0] = torch.cos(rand_t)
        se2_noise[:, 0, 1] = -torch.sin(rand_t)
        se2_noise[:, 1, 0] = torch.sin(rand_t)
        se2_noise[:, 1, 1] = torch.cos(rand_t)
    else:
        se2_noise[:, 0, 0] = 1
        se2_noise[:, 1, 1] = 1
    if dx_std != 0.0:
        se2_noise[:, 0, 3] = torch.normal(mean=0.0, std=dx_std, size=(batch_size,))
    if dy_std != 0.0:
        se2_noise[:, 1, 3] = torch.normal(mean=0.0, std=dy_std, size=(batch_size,))
    se2_noise[:, 2, 2] = 1
    se2_noise[:, 3, 3] = 1
    return transforms @ se2_noise, rand_t[0], se2_noise[0, 0, 3], se2_noise[0, 1, 3]

def visalized_hard_batches(models: list, dataset: MassHDF5, cfg: ReportConfig,
                           device: torch.device, NEW_SIZE, CENTER, PPM):
    
    agent_pool = CurriculumPool(cfg.difficulty, cfg.max_agent_count,
                                cfg.max_agent_count, device)
    count = 1
    for b_idx, (hard_batch_index) in enumerate(cfg.hard_batch_indices):
        hard_batch_dict = {}
        hard_batch_dict['step'] = count
        count += 1
        # no need to squeeze with __getitem__
        (hrgbs, hlabels, hcar_masks, hfov_masks, htransforms, _) = \
                            dataset.__getitem__(hard_batch_index)
        hmasks = hcar_masks + hfov_masks
        hrgbs, hlabels, hmasks, htransforms = to_device(device, hrgbs, hlabels, 
                                                        hmasks, htransforms)
        agent_pool.generate_connection_strategy(hmasks, htransforms,
                                                PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                CENTER[0], CENTER[1])
        # add noise to batch and visualize it
        if cfg.se2_noise_enable:
            htransforms, x_noise, y_noise, th_noise = get_noisy_transforms(htransforms,
                                                                           cfg.se2_noise_dx_std,
                                                                           cfg.se2_noise_dy_std,
                                                                           cfg.se2_noise_th_std)
            hard_batch_dict['x_noise'] = x_noise
            hard_batch_dict['y_noise'] = y_noise
            hard_batch_dict['yaw_noise'] = th_noise

        # visualize model outpus
        for model_idx, (model) in enumerate(models):
            print(f'visualizing {cfg.hard_batch_labels[b_idx]} on '
                  f'{cfg.model_names[model_idx]}{" " * 20}', end='\r', flush=True)
            with torch.no_grad():
                hsseg_preds, hmask_preds = \
                    model(hrgbs, htransforms, agent_pool.adjacency_matrix)
            hard_batch_img = plot_batch(hrgbs, hlabels, hsseg_preds, hmask_preds,
                                        hmasks, agent_pool, plot_dest='image',
                                        semantic_classes=cfg.classes,
                                        title=f'Run: {cfg.runs[model_idx]}, B#: {hard_batch_index}')
            hard_batch_dict[f'{cfg.hard_batch_labels[b_idx]}/{cfg.model_names[model_idx]}'] = \
                wandb.Image(hard_batch_img, caption=f'{hard_batch_index}')
        wandb.log(hard_batch_dict)
    print('\ndone')
    wandb.finish()

def main():
    rep_cfg = ReportConfig('config/report.yml')
    sem_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    if len(rep_cfg.hard_batch_indices) == 0:
        print('training.yml: hard batches cannot be empty.')
        exit(-1)
    device = torch.device(rep_cfg.device)
    torch.manual_seed(rep_cfg.torch_seed)
    models = []
    # image geometry
    NEW_SIZE = (rep_cfg.output_h, rep_cfg.output_w)
    CENTER = (sem_cfg.center_x(NEW_SIZE[1]), sem_cfg.center_y(NEW_SIZE[0]))
    PPM = sem_cfg.pix_per_m(NEW_SIZE[0], NEW_SIZE[1])
    # dataloader stuff
    test_set = MassHDF5(dataset=rep_cfg.dset_name, path=rep_cfg.dset_dir,
                        hdf5name=rep_cfg.dset_file, size=NEW_SIZE,
                        classes=rep_cfg.classes, jitter=[0, 0, 0, 0])
    # network stuff
    for i in range(len(rep_cfg.runs)):
        snapshot_dir = rep_cfg.snapshot_dir.format(rep_cfg.runs[i])
        snapshot_path = f'{rep_cfg.model_versions[i]}_model.pth'
        snapshot_path = snapshot_dir + '/' + snapshot_path
        if not os.path.exists(snapshot_path):
            print(f'{snapshot_path} does not exist')
            exit()
        if rep_cfg.model_names[i] == 'mcnnT':
            model = TransposedMCNN(rep_cfg.num_classes, NEW_SIZE,
                        sem_cfg, rep_cfg.aggregation_types[i]).to(device)
        elif rep_cfg.model_names[i] == 'mcnnTMax':
            model = MaxoutMCNNT(rep_cfg.num_classes, NEW_SIZE,
                        sem_cfg, rep_cfg.aggregation_types[i]).to(device)
        elif rep_cfg.model_names[i] == 'mcnnNoisy':
            model = NoisyMCNN(rep_cfg.num_classes, NEW_SIZE,
                        sem_cfg, rep_cfg.aggregation_types[i]).to(device)
        else:
            print(f'unknown network architecture {rep_cfg.model_names[i]}')
            exit()
        model.load_state_dict(torch.load(snapshot_path))
        models.append(model)
        print(f'loading {snapshot_path}')
    # start wandb
    wandb.init(
        project='birdseye-semseg',
        entity='ais-birdseye',
        group='reports',
        name=rep_cfg.report_name,
        dir=rep_cfg.log_dir,
        config={
            'runs': ','.join(rep_cfg.runs),
            'model_names': ','.join(rep_cfg.model_names),
            'model_versions': ','.join(rep_cfg.model_versions),
            'agg_types': ','.join(rep_cfg.aggregation_types)
        }
    )
    visalized_hard_batches(models, test_set, rep_cfg, device,
                           NEW_SIZE, CENTER, PPM)

if __name__ == '__main__':
    main()