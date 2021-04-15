from hashlib import pbkdf2_hmac
import os
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data.color_map import our_semantics_to_cityscapes_rgb
from data.dataset import get_datasets
from data.config import SemanticCloudConfig, TrainingConfig
from data.utils import squeeze_all, to_device
from model.mcnn import MCNN, MCNN4
from model.large_mcnn import LMCNN
from agent.agent_pool import CurriculumPool

def plot_batch(rgbs: torch.Tensor, labels: torch.Tensor, sseg_preds: torch.Tensor, 
               mask_preds: torch.Tensor, gt_masks: torch.Tensor, agent_pool: CurriculumPool,
               plot_dest: str, filename = 'test.png'):
    agent_count = rgbs.shape[0]
    columns = 7
    fig = plt.figure(figsize=(22, agent_count * 4))
    for i in range(agent_count):
        rgb = rgbs[i, ...].permute(1, 2, 0)
        rgb = (rgb + 1) / 2
        single_gt_mask = gt_masks[i].cpu()
        combined_gt_mask = agent_pool.combined_masks[i].cpu()
        ss_gt_img = our_semantics_to_cityscapes_rgb(labels[i].cpu())
        # create subplot and append to ax
        ax = []

        # front RGB image
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 1))
        ax[-1].set_title(f"rgb {i}")
        plt.imshow(rgb.cpu())

        # basic mask
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 2))
        ax[-1].set_title(f"target mask {i}")
        plt.imshow(single_gt_mask)

        # predicted mask
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 3))
        ax[-1].set_title(f"predicted mask {i}")
        plt.imshow(mask_preds[i].squeeze().cpu())

        # omniscient semantic BEV image
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 4))
        ax[-1].set_title(f"omniscient BEV {i}")
        plt.imshow(ss_gt_img)

        # target semantic BEV image
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 5))
        ax[-1].set_title(f"target BEV {i}")
        ss_gt_img[combined_gt_mask == 0] = 0
        plt.imshow(ss_gt_img)

        # predicted semseg
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 6))
        ax[-1].set_title(f"predicted BEV {i}")
        ss_pred = torch.max(sseg_preds[i], dim=0)[1]
        ss_pred_img = our_semantics_to_cityscapes_rgb(ss_pred.cpu())
        plt.imshow(ss_pred_img)

        # masked predicted semseg
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 7))
        ax[-1].set_title(f"masked prediction {i}")
        ss_pred_img[combined_gt_mask == 0] = 0
        plt.imshow(ss_pred_img)
    
    if plot_dest == 'disk':
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = int(width), int(height)
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        fig.clear()
        plt.close(fig)
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elif plot_dest == 'image':
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = int(width), int(height)
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        fig.clear()
        plt.close(fig)
        return image
    elif plot_dest == 'show':
        plt.show()

def evaluate(**kwargs):
    train_cfg: TrainingConfig = kwargs.get('train_cfg')
    model = kwargs.get('model')
    loader = kwargs.get('loader')
    device = kwargs.get('device')
    agent_pool: CurriculumPool = kwargs.get('agent_pool')
    model.eval()
    NEW_SIZE, CENTER, PPM = kwargs.get('geom_properties')
    sample_plot_prob = 1.0 / train_cfg.eval_plot_count
    probs = [1 - sample_plot_prob, sample_plot_prob] 
    # plot stuff
    columns = 6
    for idx, (ids, rgbs, labels, masks, car_transforms) in enumerate(loader):
        # randomly skip samples (useful for large datasets)
        if train_cfg.eval_random_samples and np.random.choice([True, False], 1, p=probs):
            continue
        rgbs, labels, masks, car_transforms = to_device(rgbs, labels,
                                                        masks, car_transforms,
                                                        device, train_cfg.pin_memory)
        rgbs, labels, masks, car_transforms = squeeze_all(rgbs, labels, masks, car_transforms)
        agent_pool.generate_connection_strategy(ids, masks, car_transforms,
                                                PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                CENTER[0], CENTER[1])
        print(f"index {idx + 1}/{len(loader)}")
        agent_count = rgbs.shape[0]
        
        # network output
        with torch.no_grad():
            sseg_preds, mask_preds = model(rgbs, car_transforms, agent_pool.adjacency_matrix)
        plot_batch(rgbs, labels, sseg_preds, mask_preds, masks, agent_pool, train_cfg.eval_plot,
                   f'{train_cfg.eval_plot_dir}/{train_cfg.eval_run}_batch{idx + 1}.png')

        
        train_cfg.eval_plot_count -= 1
        if train_cfg.eval_plot_count == 0:
            print('\ndone!')
            break

def main():
    # configuration
    train_cfg = TrainingConfig('config/training.yml')
    geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    new_size = (train_cfg.output_h, train_cfg.output_w)
    center = (geom_cfg.center_x(new_size[1]), geom_cfg.center_y(new_size[0]))
    ppm = geom_cfg.pix_per_m(new_size[0], new_size[1])
    # torch device
    device_str = train_cfg.device
    if train_cfg.device == 'cuda':
        torch.cuda.set_device(0)
        device_str += f':{0}'
    device = torch.device(device_str)
    # seed to insure the same train/test split
    torch.manual_seed(train_cfg.torch_seed)
    # plot stuff 
    train_cfg.eval_plot_dir = train_cfg.eval_plot_dir.format(train_cfg.eval_run)
    train_cfg.eval_plot_dir += '_' + train_cfg.eval_plot_tag
    if not os.path.exists(train_cfg.eval_plot_dir):
        os.makedirs(train_cfg.eval_plot_dir)
    if train_cfg.eval_plot == 'disk':
        print('saving plots to disk')
        matplotlib.use('Agg')
    elif train_cfg.eval_plot == 'show':
        print('showing plots')
    else:
        print("valid plot options are 'show' and 'disk'")
        exit()
    # network stuff
    if train_cfg.eval_model_version != 'best' and train_cfg.eval_model_version != 'last':
        print("valid model options are 'best' and 'last'")
        exit()
    train_cfg.snapshot_dir = train_cfg.snapshot_dir.format(train_cfg.eval_run)
    snapshot_path = f'{train_cfg.eval_model_version}_model.pth'
    snapshot_path = train_cfg.snapshot_dir + '/' + snapshot_path
    if not os.path.exists(snapshot_path):
        print(f'{snapshot_path} does not exist')
        exit()
    if train_cfg.model_name == 'mcnn':
        model = MCNN(train_cfg.num_classes, new_size,
                     geom_cfg).cuda(0)
    elif train_cfg.model_name == 'mcnn4':
        model = MCNN4(train_cfg.num_classes, new_size,
                      geom_cfg).cuda(0)
    elif train_cfg.model_name == 'mcnnL':
        model = LMCNN(train_cfg.num_classes, new_size,
                      geom_cfg).cuda(0)
    else:
        print('unknown network architecture {train_cfg.eval_model_name}')
    model.load_state_dict(torch.load(snapshot_path))
    agent_pool = CurriculumPool(train_cfg.eval_difficulty,
                                train_cfg.eval_difficulty,
                                train_cfg.max_agent_count, train_cfg.strategy,
                                train_cfg.strategy_parameter, device)
    # dataloader stuff
    train_set, test_set = get_datasets(train_cfg.dset_name, train_cfg.dset_dir,
                                       train_cfg.dset_file, (0.8, 0.2),
                                       new_size, train_cfg.classes)
    if train_cfg.eval_dataset == 'train':
        eval_set = train_set
    elif train_cfg.eval_dataset == 'test':
        eval_set = test_set
    else:
        print(f'uknown dataset split {train_cfg.eval_dataset}')
        exit()
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1,
                                              shuffle=train_cfg.shuffle_data,
                                              pin_memory=train_cfg.pin_memory,
                                              num_workers=train_cfg.loader_workers)
    print(f'evaluating run {train_cfg.eval_run} with {train_cfg.eval_model_version} '
          f'snapshot of {train_cfg.eval_model_name}')
    print(f'gathering at most {train_cfg.eval_plot_count} from the {train_cfg.eval_dataset} '
          f'set randomly? {train_cfg.eval_random_samples}, w/ difficulty = {train_cfg.eval_difficulty}')
    evaluate(train_cfg=train_cfg, model=model, agent_pool=agent_pool, loader=eval_loader,
             geom_properties=(new_size, center, ppm), device=device)

if __name__ == '__main__':
    main()