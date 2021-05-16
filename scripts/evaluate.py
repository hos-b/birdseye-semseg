import os
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from agent.agent_pool import CurriculumPool
from data.color_map import our_semantics_to_cityscapes_rgb
from data.dataset import get_datasets
from data.config import EvaluationConfig, SemanticCloudConfig, TrainingConfig
from data.utils import squeeze_all, to_device
from data.utils import font_dict, newline_dict
from model.mcnn import MCNN, MCNN4
from model.large_mcnn import LMCNN, LWMCNN


def plot_batch(rgbs: torch.Tensor, labels: torch.Tensor, sseg_preds: torch.Tensor, 
               mask_preds: torch.Tensor, gt_masks: torch.Tensor, agent_pool: CurriculumPool,
               plot_dest: str, filename = 'test.png', title=''):
    agent_count = rgbs.shape[0]
    columns = 7
    fig = plt.figure(figsize=(30, agent_count * 4))
    fig.suptitle(f'{newline_dict[agent_count]}{title}', fontsize=font_dict[agent_count])
    # plt.axis('off')
    for i in range(agent_count):
        rgb = rgbs[i, ...].permute(1, 2, 0)
        rgb = (rgb + 1) / 2
        single_gt_mask = gt_masks[i].cpu()
        combined_gt_mask = agent_pool.combined_masks[i].cpu()
        ss_gt_img = our_semantics_to_cityscapes_rgb(labels[i].cpu())
        # create subplot and append to ax
        ax = []

        # front RGB image
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 1, xticks=[], yticks=[]))
        ax[-1].set_title(f"rgb {i}")
        plt.imshow(rgb.cpu())

        # basic mask
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 2, xticks=[], yticks=[]))
        ax[-1].set_title(f"target mask {i}")
        plt.imshow(single_gt_mask)

        # predicted mask
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 3, xticks=[], yticks=[]))
        ax[-1].set_title(f"predicted mask {i}")
        plt.imshow(mask_preds[i].squeeze().cpu())

        # omniscient semantic BEV image
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 4, xticks=[], yticks=[]))
        ax[-1].set_title(f"omniscient BEV {i}")
        plt.imshow(ss_gt_img)

        # target semantic BEV image
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 5, xticks=[], yticks=[]))
        ax[-1].set_title(f"target BEV {i}")
        ss_gt_img[combined_gt_mask == 0] = 0
        plt.imshow(ss_gt_img)

        # predicted semseg
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 6, xticks=[], yticks=[]))
        ax[-1].set_title(f"predicted BEV {i}")
        ss_pred = torch.max(sseg_preds[i], dim=0)[1]
        ss_pred_img = our_semantics_to_cityscapes_rgb(ss_pred.cpu())
        plt.imshow(ss_pred_img)

        # masked predicted semseg
        ax.append(fig.add_subplot(agent_count, columns, i * columns + 7, xticks=[], yticks=[]))
        ax[-1].set_title(f"masked prediction {i}")
        ss_pred_img[combined_gt_mask == 0] = 0
        plt.imshow(ss_pred_img)
    
    if plot_dest == 'disk':
        matplotlib.use('Agg')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = int(width), int(height)
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        fig.clear()
        plt.close(fig)
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elif plot_dest == 'image':
        matplotlib.use('Agg')
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
    eval_cfg: EvaluationConfig = kwargs.get('eval_cfg')
    model = kwargs.get('model')
    loader = kwargs.get('loader')
    device = kwargs.get('device')
    agent_pool: CurriculumPool = kwargs.get('agent_pool')
    model.eval()
    NEW_SIZE, CENTER, PPM = kwargs.get('geom_properties')
    sample_plot_prob = 1.0 / eval_cfg.plot_count
    probs = [1 - sample_plot_prob, sample_plot_prob] 
    # plot stuff
    columns = 6
    for idx, (ids, rgbs, labels, masks, car_transforms, batch_no) in enumerate(loader):
        rgbs, labels, masks, car_transforms = to_device(rgbs, labels,
                                                        masks, car_transforms,
                                                        device, False)
        rgbs, labels, masks, car_transforms = squeeze_all(rgbs, labels, masks, car_transforms)
        agent_pool.generate_connection_strategy(ids, masks, car_transforms,
                                                PPM, NEW_SIZE[0], NEW_SIZE[1],
                                                CENTER[0], CENTER[1])
        print(f"index {idx + 1}/{len(loader)}")
        agent_count = rgbs.shape[0]
        
        # network output
        with torch.no_grad():
            sseg_preds, mask_preds = model(rgbs, car_transforms, agent_pool.adjacency_matrix)
        plot_batch(rgbs, labels, sseg_preds, mask_preds, masks, agent_pool, eval_cfg.plot_type,
                   f'{eval_cfg.plot_dir}/{eval_cfg.run}_batch{idx + 1}.png', 
                   f'Batch #{batch_no.item()}')
        
        eval_cfg.plot_count -= 1
        if eval_cfg.plot_count == 0:
            print('\ndone!')
            break

def main():
    # configuration
    eval_cfg = EvaluationConfig('config/evaluation.yml')
    geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    new_size = (eval_cfg.output_h, eval_cfg.output_w)
    center = (geom_cfg.center_x(new_size[1]), geom_cfg.center_y(new_size[0]))
    ppm = geom_cfg.pix_per_m(new_size[0], new_size[1])
    # torch device
    device_str = eval_cfg.device
    if eval_cfg.device == 'cuda':
        torch.cuda.set_device(0)
        device_str += f':{0}'
    device = torch.device(device_str)
    # seed to insure the same train/test split
    torch.manual_seed(eval_cfg.torch_seed)
    # plot stuff 
    eval_cfg.plot_dir = eval_cfg.plot_dir.format(eval_cfg.run)
    eval_cfg.plot_dir += '_' + eval_cfg.plot_tag
    if not os.path.exists(eval_cfg.plot_dir):
        os.makedirs(eval_cfg.plot_dir)
    if eval_cfg.plot_type == 'disk':
        print('saving plots to disk')
    elif eval_cfg.plot_type == 'show':
        print('showing plots')
    else:
        print("valid plot types are 'show' and 'disk'")
        exit()
    # network stuff
    if eval_cfg.model_version != 'best' and eval_cfg.model_version != 'last':
        print("valid model version are 'best' and 'last'")
        exit()
    eval_cfg.snapshot_dir = eval_cfg.snapshot_dir.format(eval_cfg.run)
    snapshot_path = f'{eval_cfg.model_version}_model.pth'
    snapshot_path = eval_cfg.snapshot_dir + '/' + snapshot_path
    if not os.path.exists(snapshot_path):
        print(f'{snapshot_path} does not exist')
        exit()
    if eval_cfg.model_name == 'mcnn':
        model = MCNN(eval_cfg.num_classes, new_size,
                     geom_cfg, eval_cfg.aggregation_type).cuda(0)
    elif eval_cfg.model_name == 'mcnn4':
        model = MCNN4(eval_cfg.num_classes, new_size,
                      geom_cfg, eval_cfg.aggregation_type).cuda(0)
    elif eval_cfg.model_name == 'mcnnL':
        model = LMCNN(eval_cfg.num_classes, new_size,
                      geom_cfg, eval_cfg.aggregation_type).cuda(0)
    elif eval_cfg.model_name == 'mcnnLW':
        model = LWMCNN(eval_cfg.num_classes, new_size,
                       geom_cfg, eval_cfg.aggregation_type).cuda(0)
    else:
        print('unknown network architecture {eval_cfg.model_name}')
        exit()
    model.load_state_dict(torch.load(snapshot_path))
    agent_pool = CurriculumPool(eval_cfg.difficulty, eval_cfg.difficulty,
                                eval_cfg.max_agent_count, device)
    # dataloader stuff
    train_set, test_set = get_datasets(eval_cfg.dset_name, eval_cfg.dset_dir,
                                       eval_cfg.dset_file, (0.8, 0.2),
                                       new_size, eval_cfg.classes)
    if eval_cfg.data_split == 'train':
        eval_set = train_set
    elif eval_cfg.data_split == 'test':
        eval_set = test_set
    else:
        print(f'uknown dataset split {eval_cfg.data_split}')
        exit()
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1,
                                              shuffle=eval_cfg.random_samples,
                                              num_workers=1)
    print(f'evaluating run {eval_cfg.run} with {eval_cfg.model_version} '
          f'snapshot of {eval_cfg.model_name}')
    print(f'gathering at most {eval_cfg.plot_count} from the {eval_cfg.data_split} '
          f'set randomly? {eval_cfg.random_samples}, w/ difficulty = {eval_cfg.difficulty}')
    evaluate(model=model, agent_pool=agent_pool, loader=eval_loader, eval_cfg=eval_cfg,
             geom_properties=(new_size, center, ppm), device=device)

if __name__ == '__main__':
    main()