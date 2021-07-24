import os
from typing import Dict
import cv2
import torch
import kornia
import tkinter
import numpy as np
import PIL.ImageTk
import PIL.Image as PILImage

import data.color_map as color_map
from data.dataset import MassHDF5
from data.config import SemanticCloudConfig, EvaluationConfig
from data.color_map import convert_semantics_to_rgb
from data.mask_warp import get_single_relative_img_transform, get_all_aggregate_masks
from data.utils import squeeze_all, to_device
from data.utils import get_noisy_transforms
from metrics.iou import iou_per_class
from model.large_mcnn import TransposedMCNN
from model.noisy_mcnn import NoisyMCNN

class SampleWindow:
    def __init__(self, eval_cfg: EvaluationConfig, device: torch.device, new_size, center, ppm):
        # network stuff
        self.networks = {}
        self.output_h = new_size[0]
        self.output_w = new_size[1]
        self.center_x = center[0]
        self.center_y = center[1]
        self.ppm = ppm
        self.eval_cfg = eval_cfg
        self.semantic_classes = eval_cfg.classes
        self.class_count = eval_cfg.num_classes
        self.device = device
        self.current_data = None
        self.adjacency_matrix = None
        self.self_masking_en = False
        self.agent_index = 0
        self.agent_count = 8
        self.window = tkinter.Tk()
        # image panels captions
        self.rgb_panel_caption         = tkinter.Label(self.window, text='front rgb')
        self.masked_pred_panel_caption = tkinter.Label(self.window, text='masked pred')
        self.full_pred_panel_caption   = tkinter.Label(self.window, text='full pred')
        self.target_panel_caption      = tkinter.Label(self.window, text='target')
        self.rgb_panel_caption.         grid(column=0, row=0, columnspan=5)
        self.masked_pred_panel_caption. grid(column=6, row=0, columnspan=5)
        self.full_pred_panel_caption.   grid(column=11, row=0, columnspan=5)
        self.target_panel_caption.      grid(column=16, row=0, columnspan=5)
        # agent selection buttons
        self.sep_1 = tkinter.Label(self.window, text='  ')
        self.sep_1.grid(row=1, rowspan=10, column=21)
        buttons_per_row = 4
        for i in range(8):
            exec(f"self.abutton_{i} = tkinter.Button(self.window, text={i + 1})")
            exec(f"self.abutton_{i}.configure(command=lambda: self.agent_clicked({i}))", locals(), locals())
            exec(f"self.abutton_{i}.grid(column={22 + (i % buttons_per_row)}, row={1 + (i // buttons_per_row)})")
        # misc. buttons
        self.agent_label  = tkinter.Label(self.window, text=f'agent {self.agent_index}/{self.agent_count}')
        self.next_sample  = tkinter.Button(self.window, command=self.change_sample, text='next sample')
        self.smask_button = tkinter.Button(self.window, command=self.toggle_self_masking, text=f'self mask: {int(self.self_masking_en)}')
        self.agent_label. grid(column=22, row=0, columnspan=4)
        self.smask_button.grid(column=22, row=3, columnspan=4)
        self.next_sample. grid(column=22, row=4, columnspan=4)
        # noise parameters
        self.noise_label     = tkinter.Label(self.window, text=f'noise parameters')
        self.noise_label.    grid(column=22, row=5, columnspan=4)
        self.dx_noise_label  = tkinter.Label(self.window, text=f'std-x:')
        self.dx_noise_label. grid(column=22, row=6, columnspan=2)
        self.dx_noise_text   = tkinter.StringVar(value=f'{self.eval_cfg.se2_noise_dx_std}')
        self.dx_noise_entry  = tkinter.Entry(self.window, width=5, textvariable=self.dx_noise_text)
        self.dx_noise_entry. grid(column=24, row=6, columnspan=2)
        self.dy_noise_label  = tkinter.Label(self.window, text=f'std-y:')
        self.dy_noise_label. grid(column=22, row=7, columnspan=2)
        self.dy_noise_text   = tkinter.StringVar(value=f'{self.eval_cfg.se2_noise_dy_std}')
        self.dy_noise_entry  = tkinter.Entry(self.window, width=5, textvariable=self.dy_noise_text)
        self.dy_noise_entry. grid(column=24, row=7, columnspan=2)
        self.th_noise_label  = tkinter.Label(self.window, text=f'std-yaw:')
        self.th_noise_label. grid(column=22, row=8, columnspan=2)
        self.th_noise_text   = tkinter.StringVar(value=f'{self.eval_cfg.se2_noise_th_std}')
        self.th_noise_entry  = tkinter.Entry(self.window, width=5, textvariable=self.th_noise_text)
        self.th_noise_entry. grid(column=24, row=8, columnspan=2)
        self.apply_noise     = tkinter.Button(self.window, command=self.apply_noise_params, text='undertaker')
        self.apply_noise.    grid(column=22, row=9, columnspan=4)

        # adjacency matrix buttons
        self.sep_2 = tkinter.Label(self.window, text='  ')
        self.sep_2.grid(row=0, rowspan=10, column=26)
        for j in range(8):
            for i in range(8):
                if i == 0:
                    exec(f"self.mlabel_{j}{i} = tkinter.Label(self.window, text={j + 1})")
                    exec(f"self.mlabel_{j}{i}.grid(column={j + 28}, row={i})")
                if j == 0:
                    exec(f"self.mlabel_{j}{i} = tkinter.Label(self.window, text={i + 1})")
                    exec(f"self.mlabel_{j}{i}.grid(column={j + 27}, row={i + 1})")
                exec(f"self.mbutton_{j}{i} = tkinter.Button(self.window, text='1' if {i} == {j} else '0')")
                exec(f"self.mbutton_{j}{i}.configure(command=lambda: self.matrix_clicked({i}, {j}))", locals(), locals())
                exec(f"self.mbutton_{j}{i}.grid(column={j + 28}, row={i + 1})")
        # ious
        self.mskd_ious = {}
        self.full_ious = {}

    def add_network(self, network: torch.nn.Module, label: str):
        id = len(self.networks)
        net_row = 2 + len(self.networks) * 12
        network.eval()
        self.networks[label] = network
        self.mskd_ious[label] = torch.zeros((self.class_count, 1), dtype=torch.float64).to(self.device)
        self.full_ious[label] = torch.zeros((self.class_count, 1), dtype=torch.float64).to(self.device)
        exec(f"self.network_label_{id}        = tkinter.Label(self.window, text='[{label}]')")
        exec(f"self.rgb_panel_{id}            = tkinter.Label(self.window, text='placeholder')")
        exec(f"self.masked_pred_panel_{id}    = tkinter.Label(self.window, text='placeholder')")
        exec(f"self.full_pred_panel_{id}      = tkinter.Label(self.window, text='placeholder')")
        exec(f"self.target_panel_{id}         = tkinter.Label(self.window, text='placeholder')")
        exec(f"self.masked_iou_label_{id}     = tkinter.Label(self.window, text='placeholder')")
        exec(f"self.full_iou_label_{id}       = tkinter.Label(self.window, text='placeholder')")
        exec(f"self.seperator_{id}            = tkinter.Label(self.window, text='{'-' * 105}')")
        exec(f"self.network_label_{id}.         grid(column=0, row={net_row - 1}, columnspan=1)")
        exec(f"self.rgb_panel_{id}.             grid(column=0, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.masked_pred_panel_{id}.     grid(column=6, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.full_pred_panel_{id}.       grid(column=11, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.target_panel_{id}.          grid(column=16, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.masked_iou_label_{id}.      grid(column=0, row={net_row + 8}, columnspan=20)")
        exec(f"self.full_iou_label_{id}.        grid(column=0, row={net_row + 9}, columnspan=20)")
        exec(f"self.seperator_{id}.             grid(column=0, row={net_row + 10}, columnspan=20)")

    def assign_dataset_iterator(self, dset_iterator):
        self.dset_iterator = dset_iterator

    def set_baseline(self, net: torch.nn.Module):
        self.baseline = net
        self.baseline.eval()

    def start(self):
        if 'baseline' not in self.networks:
            print("missing 'baseline' network")
            exit()
        self.change_sample()
        self.window.mainloop()

    def matrix_clicked(self, i: int, j: int):
        if self.adjacency_matrix[j, i] == 1:
            self.adjacency_matrix[j, i] = 0
            self.adjacency_matrix[i, j] = 0
            exec(f"self.mbutton_{j}{i}.configure(text='0')")
            exec(f"self.mbutton_{i}{j}.configure(text='0')")
        else:
            self.adjacency_matrix[j, i] = 1
            self.adjacency_matrix[i, j] = 1
            exec(f"self.mbutton_{j}{i}.configure(text='1')")
            exec(f"self.mbutton_{i}{j}.configure(text='1')")
        self.update_prediction()

    def agent_clicked(self, agent_id: int):
        if agent_id < self.agent_count:
            self.agent_index = agent_id
            self.agent_label.configure(text=f'agent {self.agent_index + 1}/{self.agent_count}')
            self.update_prediction()

    def toggle_self_masking(self):
        self.self_masking_en = not self.self_masking_en
        self.smask_button.configure(text=f'self mask: {int(self.self_masking_en)}')
        self.update_prediction()
    
    def apply_noise_params(self):
        th_std = self.eval_cfg.se2_noise_th_std
        dy_std = self.eval_cfg.se2_noise_dx_std
        dx_std = self.eval_cfg.se2_noise_dy_std
        try:
            th_std = float(self.th_noise_text.get())
        except ValueError:
            print(f'invalid noise parameter {self.th_noise_text.get()}')
            self.th_noise_text.set(f'{self.eval_cfg.se2_noise_th_std}')
        try:
            dx_std = float(self.dx_noise_text.get())
        except ValueError:
            print(f'invalid noise parameter {self.dx_noise_text.get()}')
            self.dx_noise_text.set(f'{self.eval_cfg.se2_noise_dx_std}')
        try:
            dy_std = float(self.dy_noise_text.get())
        except ValueError:
            print(f'invalid noise parameter {self.dy_noise_text.get()}')
            self.dy_noise_text.set(f'{self.eval_cfg.se2_noise_dy_std}')
        self.eval_cfg.se2_noise_th_std = th_std
        self.eval_cfg.se2_noise_dx_std = dx_std
        self.eval_cfg.se2_noise_dy_std = dy_std
        self.update_prediction()

    def calculate_ious(self, dataset: MassHDF5, segmentation_classes: dict):
        if not self.eval_cfg.evaluate_at_start:
            print('full dataset evaluation disabled in yaml file')
            for i, (network) in enumerate(self.networks.keys()):
                exec(f"self.full_iou_label_{i}.configure(text='network not evaluated')")
                exec(f"self.masked_iou_label_{i}.configure(text='network not evaluated')")
            return

        sample_count = 0
        dloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        total_length = len(dloader)
        print('calculating IoUs...')
        for idx, (rgbs, labels, car_masks, fov_masks, car_transforms, _) in enumerate(dloader):
            masks = car_masks + fov_masks
            print(f'\r{idx + 1}/{total_length}', end='')
            rgbs, labels, masks, car_transforms = to_device(self.device, rgbs, labels,
                                                            masks, car_transforms)
            rgbs, labels, masks, car_transforms = squeeze_all(rgbs, labels, masks, car_transforms)
            gt_aggregate_masks = get_all_aggregate_masks(masks, car_transforms, self.ppm,
                                                      self.output_h, self.output_w,
                                                      self.center_x, self.center_y)
            if self.eval_cfg.se2_noise_enable:
                car_transforms = get_noisy_transforms(car_transforms,
                                                      self.eval_cfg.se2_noise_dx_std,
                                                      self.eval_cfg.se2_noise_dy_std,
                                                      self.eval_cfg.se2_noise_th_std)
            agent_count = rgbs.shape[0]
            sample_count += agent_count
            for name, network in self.networks.items():
                # manual aggregation for baseline network
                if name == 'baseline':
                    with torch.no_grad():
                        single_ss_preds, single_mask_preds = network(rgbs, car_transforms, torch.eye(agent_count))
                    # masking of regions outside FoV [for a fair comparison]
                    single_ss_preds *= single_mask_preds
                    ss_preds = torch.zeros_like(single_ss_preds)
                    for i in range(agent_count):
                        relative_tfs = get_single_relative_img_transform(car_transforms, i, self.ppm,
                                                                         self.output_h, self.output_w,
                                                                         self.center_x, self.center_y).to(self.device)
                        relative_semantics = kornia.warp_affine(single_ss_preds, relative_tfs,
                                                                dsize=(self.output_h, self.output_w),
                                                                flags='nearest')
                        ss_preds[i] = relative_semantics.sum(dim=0)
                # latent aggregation for others
                else:
                    with torch.no_grad():
                        ss_preds, _ = network(rgbs, car_transforms, torch.ones((agent_count,
                                                                                agent_count)))
                self.mskd_ious[name] += iou_per_class(ss_preds, labels, gt_aggregate_masks).to(self.device)
                self.full_ious[name] += iou_per_class(ss_preds, labels, torch.ones_like(gt_aggregate_masks)).to(self.device)

        for i, (network) in enumerate(self.networks.keys()):
            full_iou_txt = 'full IoU  '
            mskd_iou_txt = 'mskd IoU  '
            for semantic_idx, semantic_class in segmentation_classes.items():
                full_iou_txt += f'{semantic_class.lower()}: {(self.full_ious[network][semantic_idx] / sample_count).item():.02f} '
                mskd_iou_txt += f'{semantic_class.lower()}: {(self.mskd_ious[network][semantic_idx] / sample_count).item():.02f} '
            exec(f"self.full_iou_label_{i}.configure(text='{full_iou_txt}')")
            exec(f"self.masked_iou_label_{i}.configure(text='{mskd_iou_txt}')")

    def change_sample(self):
        (rgbs, labels, car_masks, fov_masks, car_transforms, batch_index) = next(self.dset_iterator)
        masks = car_masks + fov_masks
        rgbs, labels, masks, car_transforms = to_device(self.device, rgbs, labels,
                                                        masks, car_transforms)
        rgbs, labels, masks, car_transforms = squeeze_all(rgbs, labels, masks, car_transforms)
        self.current_data = (rgbs, labels, masks, car_transforms)
        self.agent_count = rgbs.shape[0]
        self.adjacency_matrix = torch.eye(self.agent_count)
        self.window.title(f'batch #{batch_index.squeeze().item()}')
        self.agent_index = 0
        self.agent_label.configure(text=f'agent {self.agent_index + 1}/{self.agent_count}')
        # enable/disable adjacency matrix buttons, reset their labels
        for j in range(8):
            for i in range(8):
                exec(f"self.mbutton_{j}{i}.configure(text='1' if {i} == {j} else '0')")
                if i < self.agent_count and j < self.agent_count:
                    exec(f"self.mbutton_{j}{i}['state'] = 'normal'")
                else:
                    exec(f"self.mbutton_{j}{i}['state'] = 'disabled'")
        # enable/disable agent buttons
        for i in range(8):
            if i < self.agent_count:
                exec(f"self.abutton_{i}['state'] = 'normal'")
            else:
                exec(f"self.abutton_{i}['state'] = 'disabled'")
        self.update_prediction()

    def update_prediction(self):
        (rgbs, labels, _, car_transforms) = self.current_data
        
        car_transforms = get_noisy_transforms(car_transforms,
                                                self.eval_cfg.se2_noise_dx_std,
                                                self.eval_cfg.se2_noise_dy_std,
                                                self.eval_cfg.se2_noise_th_std)
        # front RGB image
        rgb = rgbs[self.agent_index, ...].permute(1, 2, 0)
        rgb = ((rgb + 1) * 255 / 2).cpu().numpy().astype(np.uint8)
        rgb = cv2.resize(rgb, (342, 256), cv2.INTER_LINEAR)
        rgb_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(rgb), 'RGB')

        # target image
        ss_gt_img = convert_semantics_to_rgb(labels[self.agent_index].cpu(), self.semantic_classes)
        target_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(ss_gt_img), 'RGB')

        for i, (name, network) in enumerate(self.networks.items()):
            # >>> front RGB image
            exec(f"self.rgb_panel_{i}.configure(image=rgb_tk)")
            exec(f"self.rgb_panel_{i}.image = rgb_tk")

            # >>> target image
            exec(f"self.target_panel_{i}.configure(image=target_tk)")
            exec(f"self.target_panel_{i}.image = target_tk")


            if name == 'baseline':
                # >>> masked predicted semseg w/o external influence
                with torch.no_grad():
                    all_ss_preds, all_mask_preds = network(rgbs, car_transforms, torch.eye(self.agent_count))
                current_ss_pred = all_ss_preds[self.agent_index].argmax(dim=0)
                current_ss_pred_img = convert_semantics_to_rgb(current_ss_pred.cpu(), self.semantic_classes)
                current_mask_pred = all_mask_preds[self.agent_index].squeeze().cpu()
                current_ss_pred_img[current_mask_pred == 0] = 0
                masked_pred_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(current_ss_pred_img), 'RGB')
                exec(f"self.masked_pred_panel_{i}.configure(image=masked_pred_tk)")
                exec(f"self.masked_pred_panel_{i}.image = masked_pred_tk")
                # >>> full predicted semseg w/ influence from adjacency matrix
                # using self masking (only for baseline since others do latent masking)
                if self.self_masking_en:
                    all_ss_preds *= all_mask_preds
                not_selected = torch.where(self.adjacency_matrix[self.agent_index] == 0)[0]
                relative_tfs = get_single_relative_img_transform(car_transforms, self.agent_index,
                                                                 self.ppm, self.output_h, self.output_w,
                                                                 self.center_x, self.center_y).to(self.device)
                current_warped_semantics = kornia.warp_affine(all_ss_preds, relative_tfs,
                                                              dsize=(self.output_h, self.output_w),
                                                              flags='nearest')
                current_warped_semantics[not_selected] = 0
                current_warped_semantics = current_warped_semantics.sum(dim=0).argmax(dim=0)
                current_aggregated_semantics = convert_semantics_to_rgb(current_warped_semantics.cpu(),
                                                                        self.semantic_classes)
            else:
                # >>> masked predicted semseg w/o external influence
                with torch.no_grad():
                    all_ss_eye, all_mask_eye = network(rgbs, car_transforms, torch.eye(self.agent_count))
                current_ss_pred = all_ss_eye[self.agent_index].argmax(dim=0)
                current_ss_pred_img = convert_semantics_to_rgb(current_ss_pred.cpu(), self.semantic_classes)
                current_mask_pred = all_mask_eye[self.agent_index].squeeze().cpu()
                current_ss_pred_img[current_mask_pred == 0] = 0
                masked_pred_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(current_ss_pred_img), 'RGB')
                exec(f"self.masked_pred_panel_{i}.configure(image=masked_pred_tk)")
                exec(f"self.masked_pred_panel_{i}.image = masked_pred_tk")
                # >>> full predicted semseg w/ influence from adjacency matrix
                with torch.no_grad():
                    all_ss_preds, all_mask_preds = network(rgbs, car_transforms, self.adjacency_matrix)
                current_aggregated_semantics = \
                    convert_semantics_to_rgb(all_ss_preds[self.agent_index].argmax(dim=0).cpu(),
                                             self.semantic_classes)

            full_pred_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(current_aggregated_semantics), 'RGB')
            exec(f"self.full_pred_panel_{i}.configure(image=full_pred_tk)")
            exec(f"self.full_pred_panel_{i}.image = full_pred_tk")

def main():
    sem_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    eval_cfg = EvaluationConfig('config/evaluation.yml')
    device = torch.device(eval_cfg.device)
    torch.manual_seed(eval_cfg.torch_seed)
    # image geometry
    NEW_SIZE = (eval_cfg.output_h, eval_cfg.output_w)
    CENTER = (sem_cfg.center_x(NEW_SIZE[1]), sem_cfg.center_y(NEW_SIZE[0]))
    PPM = sem_cfg.pix_per_m(NEW_SIZE[0], NEW_SIZE[1])
    # gui object
    gui = SampleWindow(eval_cfg, device, NEW_SIZE, CENTER, PPM)
    # dataloader stuff
    test_set = MassHDF5(dataset=eval_cfg.dset_name, path=eval_cfg.dset_dir,
                        hdf5name=eval_cfg.dset_file, size=NEW_SIZE,
                        classes=eval_cfg.classes, jitter=[0, 0, 0, 0])
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    gui.assign_dataset_iterator(iter(loader))
    # baseline stuff
    baseline_dir = eval_cfg.snapshot_dir.format(eval_cfg.baseline_run)
    baseline_path = baseline_dir + f'/{eval_cfg.baseline_model_version}_model.pth'
    if eval_cfg.baseline_model_name == 'mcnnT':
        baseline_model = TransposedMCNN(eval_cfg.num_classes, NEW_SIZE,
                    sem_cfg, eval_cfg.aggregation_types[0]).to(device)
    elif eval_cfg.baseline_model_name == 'mcnnNoisy':
        baseline_model = NoisyMCNN(eval_cfg.num_classes, NEW_SIZE,
                    sem_cfg, eval_cfg.aggregation_types[0]).cuda(0)
    else:
        print(f'unknown baseline network architecture {eval_cfg.baseline_model_name}')
        exit()
    baseline_model.load_state_dict(torch.load(baseline_path))
    gui.add_network(baseline_model, 'baseline')
    # other network stuff
    for i in range(len(eval_cfg.runs)):
        snapshot_dir = eval_cfg.snapshot_dir.format(eval_cfg.runs[i])
        snapshot_path = f'{eval_cfg.model_versions[i]}_model.pth'
        snapshot_path = snapshot_dir + '/' + snapshot_path
        if not os.path.exists(snapshot_path):
            print(f'{snapshot_path} does not exist')
            exit()
        if eval_cfg.model_names[i] == 'mcnnT':
            model = TransposedMCNN(eval_cfg.num_classes, NEW_SIZE,
                        sem_cfg, eval_cfg.aggregation_types[i]).to(device)
        elif eval_cfg.model_names[i] == 'mcnnNoisy':
            model = NoisyMCNN(eval_cfg.num_classes, NEW_SIZE,
                        sem_cfg, eval_cfg.aggregation_types[i]).to(device)
        else:
            print(f'unknown network architecture {eval_cfg.model_names[i]}')
            exit()
        model.load_state_dict(torch.load(snapshot_path))
        print(f'loading {snapshot_path}')
        gui.add_network(model, eval_cfg.runs[i])
    # evaluate the added networks
    if eval_cfg.classes == 'carla':
        segmentation_classes = color_map.__carla_classes
    elif eval_cfg.classes == 'ours':
        segmentation_classes = color_map.__our_classes
    elif eval_cfg.classes == 'ours+mask':
        segmentation_classes = color_map.__our_classes_plus_mask
    elif eval_cfg.classes == 'diminished':
        segmentation_classes = color_map.__diminished_classes
    gui.calculate_ious(test_set, segmentation_classes)
    # start the gui
    gui.start()

if __name__ == '__main__':
    main()