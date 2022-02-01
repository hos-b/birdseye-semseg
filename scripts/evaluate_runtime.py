from operator import concat
import os
import cv2
import torch
import numpy as np

import data.color_map as color_map
from data.dataset import MassHDF5
from data.config import SemanticCloudConfig, EvaluationConfig
from data.color_map import convert_semantics_to_rgb
from data.mask_warp import get_single_adjacent_aggregate_mask
from data.utils import squeeze_all, to_device
from data.utils import get_noisy_transforms, get_relative_noise
from data.utils import write_xycentered_text, write_xcentered_text
from model.factory import get_model


def main():
    sem_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    eval_cfg = EvaluationConfig('config/evaluation.yml')
    device = torch.device(eval_cfg.device)
    torch.manual_seed(eval_cfg.torch_seed)
    if len(eval_cfg.runtime_network_labels) != len(eval_cfg.runs):
        print(f'sanity-check-error: runtime labels list size must match network count.')
        exit()
    # image geometry ---------------------------------------------------------------------------------------
    NEW_SIZE = (eval_cfg.output_h, eval_cfg.output_w)
    CENTER = (sem_cfg.center_x(NEW_SIZE[1]), sem_cfg.center_y(NEW_SIZE[0]))
    PPM = sem_cfg.pix_per_m(NEW_SIZE[0], NEW_SIZE[1])
    # semantic classes
    if eval_cfg.classes == 'carla':
        semantic_classes = color_map.__carla_classes
    elif eval_cfg.classes == 'ours':
        semantic_classes = color_map.__our_classes
    elif eval_cfg.classes == 'ours+mask':
        semantic_classes = color_map.__our_classes_plus_mask
    elif eval_cfg.classes == 'diminished':
        semantic_classes = color_map.__diminished_classes
    elif eval_cfg.classes == 'diminished+mask':
        semantic_classes = color_map.__diminished_classes_plus_mask
    else:
        raise ValueError('unknown class set')
    # dataloader stuff -------------------------------------------------------------------------------------
    test_set = MassHDF5(dataset=eval_cfg.dset_name, path=eval_cfg.dset_dir,
                        hdf5name=eval_cfg.dset_file, size=NEW_SIZE,
                        classes=eval_cfg.classes, jitter=[0, 0, 0, 0],
                        mask_gaussian_sigma=eval_cfg.gaussian_mask_std,
                        guassian_kernel_size=eval_cfg.gaussian_kernel_size)
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    if test_set.batch_histogram.sum() != test_set.batch_histogram[test_set.max_agent_count - 1]:
        print('sanity-check-error: batch sizes histogram has multiple non-empty bins.')
        exit()
    eval_cfg.runtime_cache_dir += f'/{eval_cfg.dset_file.split(".")[0]}/{eval_cfg.runtime_title}'
    os.makedirs(eval_cfg.runtime_cache_dir, exist_ok=True)
    # other network stuff ----------------------------------------------------------------------------------
    networks = {}
    for i in range(len(eval_cfg.runs)):
        snapshot_dir = eval_cfg.snapshot_dir.format(eval_cfg.runs[i])
        snapshot_path = f'{eval_cfg.model_versions[i]}_model.pth'
        snapshot_path = snapshot_dir + '/' + snapshot_path
        if not os.path.exists(snapshot_path):
            print(f'{snapshot_path} does not exist')
            exit(-1)
        model = get_model(
            eval_cfg.model_names[i], eval_cfg.num_classes, NEW_SIZE,
            sem_cfg, eval_cfg.aggregation_types[i]
        ).to(device)
        print(f'loading {snapshot_path}')
        try:
            model.load_state_dict(torch.load(snapshot_path))
        except:
            print(f'{eval_cfg.model_names[i]} implementation is incompatible with {eval_cfg.runs[i]}')
            exit()
        networks[eval_cfg.runtime_network_labels[i]] = (model, eval_cfg.model_gnn_flags[i])
    # evaluate the added networks --------------------------------------------------------------------------
    dset_length = len(loader)
    rgb_h = eval_cfg.output_h
    rgb_w = int(640 / (480 / eval_cfg.output_w))
    border_size = eval_cfg.runtime_border_size
    label_w = 200
    label_h = 65
    font_scale = 1
    font_thickness = 2
    for idx, (rgbs, labels, car_masks, fov_masks, gt_transforms, _) in enumerate(loader):
        print(f'\r{idx + 1}/{dset_length}', end='')
        rgbs, labels, car_masks, fov_masks, gt_transforms = to_device(
            device, rgbs, labels, car_masks, fov_masks, gt_transforms
        )
        rgbs, labels, car_masks, fov_masks, gt_transforms = squeeze_all(
            rgbs, labels, car_masks, fov_masks, gt_transforms
        )
        agent_count = rgbs.shape[0]
        adjacency_matrix = torch.ones((agent_count, agent_count))
        frame = []
        h_border = None
        for agent_index, i in enumerate(eval_cfg.runtime_agents):
            agent_sample = []
            v_border = np.zeros((eval_cfg.output_h, border_size, 3), dtype=np.uint8)
            v_border[:, :] = eval_cfg.runtime_bkg_color
            agent_sample.append(v_border)
            # create agent label
            agent_label = np.zeros((eval_cfg.output_h, label_w, 3), dtype=np.uint8)
            agent_label[:, :] = eval_cfg.runtime_bkg_color
            write_xycentered_text(
                agent_label, f'agent {agent_index}', eval_cfg.runtime_text_color,
                cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness
            )
            agent_sample.append(agent_label)
            agent_sample.append(v_border)
            # get resized rgb
            rgb = rgbs[i, ...].permute(1, 2, 0)
            rgb = ((rgb + 1) * 255 / 2).cpu().numpy().astype(np.uint8)
            rgb = cv2.resize(rgb, dsize=(rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
            agent_sample.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            agent_sample.append(v_border)
            # get gt semantics and mask
            gt_ss_img = convert_semantics_to_rgb(labels[i].cpu(), eval_cfg.classes)
            gt_aggr_mask = get_single_adjacent_aggregate_mask(
                car_masks + fov_masks, gt_transforms, i, PPM,
                eval_cfg.output_h, eval_cfg.output_w, CENTER[0], CENTER[1],
                adjacency_matrix, True
            ).cpu().numpy()
            if eval_cfg.transparent_masks:
                gt_ss_img[gt_aggr_mask == 0, :] = gt_ss_img[gt_aggr_mask == 0, :] / 1.5
            else:
                gt_ss_img[gt_aggr_mask == 0, :] = 0
            agent_sample.append(cv2.cvtColor(gt_ss_img, cv2.COLOR_BGR2RGB))
            agent_sample.append(v_border)
            # add localization noise (if std > 0)
            car_transforms = get_noisy_transforms(
                gt_transforms,
                eval_cfg.se2_noise_dx_std,
                eval_cfg.se2_noise_dy_std,
                eval_cfg.se2_noise_th_std
            )
            # get network outuput
            for (network, graph_flag) in networks.values():
                _, _, pred_aggr_sseg, pred_aggr_mask = network.get_eval_output(
                    semantic_classes, graph_flag, rgbs, car_masks, fov_masks,
                    car_transforms, adjacency_matrix, PPM, eval_cfg.output_h, eval_cfg.output_w,
                    CENTER[0], CENTER[1], i, True,
                    torch.device('cpu'), True
                )
                pred_aggr_ss_img = convert_semantics_to_rgb(pred_aggr_sseg.argmax(dim=0), eval_cfg.classes)
                # thresholding aggr. mask
                pred_aggr_mask[pred_aggr_mask < eval_cfg.mask_thresh] = 0
                pred_aggr_mask[pred_aggr_mask >= eval_cfg.mask_thresh] = 1
                if eval_cfg.transparent_masks and graph_flag:
                    pred_aggr_ss_img[pred_aggr_mask == 0, :] = pred_aggr_ss_img[pred_aggr_mask == 0, :] / 1.5
                else:
                    pred_aggr_ss_img[pred_aggr_mask == 0, :] = 0
                agent_sample.append(cv2.cvtColor(pred_aggr_ss_img, cv2.COLOR_BGR2RGB))
                agent_sample.append(v_border)

            # add concatenated agent sample to final frame
            frame.append(np.concatenate(agent_sample, axis=1))
            if h_border is None:
                h_border = np.zeros((border_size, frame[0].shape[1], 3), dtype=np.uint8)
                h_border[:, :] = eval_cfg.runtime_bkg_color
            frame.append(h_border)

        # add labels to the top
        label_row = np.zeros((label_h, frame[0].shape[1], 3), dtype=np.uint8)
        label_row[:, :] = eval_cfg.runtime_bkg_color
        offset = label_w + (2 * border_size)
        write_xycentered_text(
            label_row[:, offset:offset + rgb_w], 'Input',
            eval_cfg.runtime_text_color, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness
        )
        offset += rgb_w + border_size
        write_xycentered_text(
            label_row[:, offset:offset + eval_cfg.output_w], 'GT',
            eval_cfg.runtime_text_color, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness
        )
        for net_label in eval_cfg.runtime_network_labels:
            offset += eval_cfg.output_w + border_size
            write_xycentered_text(
                label_row[:, offset:offset + eval_cfg.output_w], net_label,
                eval_cfg.runtime_text_color, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness
            )
        frame.insert(0, label_row)
        cv2.imwrite(
            f'{eval_cfg.runtime_cache_dir}/output_{idx:04d}.png',
            np.concatenate(frame, axis=0)
        )

    print(f'\ndone.\nuse ffmpeg -r 10 -i {eval_cfg.runtime_cache_dir}/output_%04d.png '
          f'-vcodec mpeg4 -y -vb 40M {eval_cfg.runtime_cache_dir}/../{eval_cfg.runtime_title}.mp4')

if __name__ == '__main__':
    main()