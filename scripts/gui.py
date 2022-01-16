import os
from typing import List
import cv2
import torch
import tkinter
import tkinter.font as tkFont
import numpy as np
import PIL.ImageTk
import PIL.Image as PILImage

import data.color_map as color_map
from data.dataset import MassHDF5
from data.config import SemanticCloudConfig, EvaluationConfig
from data.color_map import convert_semantics_to_rgb
from data.mask_warp import get_single_adjacent_aggregate_mask
from data.utils import squeeze_all, to_device
from data.utils import get_noisy_transforms, get_relative_noise
from model.factory import get_model
from metrics.iou import NetworkMetrics
from metrics.inference_time import InferenceMetrics
from metrics.noise import NoiseMetrics

class SampleWindow:
    def __init__(self, eval_cfg: EvaluationConfig, classes_dict, device: torch.device, new_size, center, ppm):
        # network stuff
        self.networks = {}
        self.graph_flags = {}
        self.output_h = new_size[0]
        self.output_w = new_size[1]
        self.center_x = center[0]
        self.center_y = center[1]
        self.ppm = ppm
        self.eval_cfg = eval_cfg
        self.semantic_classes = eval_cfg.classes
        self.segclass_dict = classes_dict
        self.device = device
        self.current_data = None
        self.adjacency_matrix = None
        self.show_masks = False
        self.baseline_masking_en = False
        self.noise_correction_en = True
        self.noisy_transforms = None
        self.agent_index = 0
        self.agent_count = 8
        self.batch_index = 0
        self.visualized_data = {}
        # root window
        self.root = tkinter.Tk()
        # visualization window
        self.viz_frame = tkinter.Frame(self.root)
        self.viz_frame.pack(side='left', fill='both', expand=True)
        self.canvas = tkinter.Canvas(self.viz_frame)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar = tkinter.Scrollbar(self.viz_frame, orient='vertical', command=self.canvas.yview)
        self.scrollbar.pack(side='right', fill='y')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<MouseWheel>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.viz_window = tkinter.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.viz_window, anchor='nw')
        # control window
        self.control_window = tkinter.Frame(self.root)
        self.control_window.pack(side='right', fill='y', expand=False)
        # image panels captions
        self.rgb_panel_caption         = tkinter.Label(self.viz_window, text='front rgb')
        self.solo_pred_panel_caption   = tkinter.Label(self.viz_window, text='solo pred')
        self.aggr_pred_panel_caption   = tkinter.Label(self.viz_window, text='aggr pred')
        self.aggr_trgt_panel_caption   = tkinter.Label(self.viz_window, text='target')
        self.rgb_panel_caption.        grid(column=0, row=0, columnspan=5)
        self.solo_pred_panel_caption.  grid(column=6, row=0, columnspan=5)
        self.aggr_pred_panel_caption.  grid(column=11, row=0, columnspan=5)
        self.aggr_trgt_panel_caption.  grid(column=16, row=0, columnspan=5)
        # space separator for top side of control window
        self.sep_1 = tkinter.Label(self.control_window, text='    ')
        self.sep_1.grid(row=0, column=0, columnspan=16)
        # space separator for left side of control window
        self.sep_1 = tkinter.Label(self.control_window, text='    ')
        self.sep_1.grid(row=1, column=0, rowspan=10, columnspan=2)
        # agent selection buttons
        buttons_per_row = 4
        for i in range(8):
            exec(f"self.abutton_{i} = tkinter.Button(self.control_window, text='{i + 1}')")
            exec(f"self.abutton_{i}.configure(command=lambda: self.agent_clicked({i}))", locals(), locals())
            exec(f"self.abutton_{i}.grid(column={2 + (i % buttons_per_row)}, row={1 + i // buttons_per_row})")
        # misc. buttons
        self.smask_button     = tkinter.Button(self.control_window, command=self.toggle_baseline_self_masking, text=f'filter baseline: {int(self.baseline_masking_en)}')
        self.viz_masks_button = tkinter.Button(self.control_window, command=self.toggle_mask_visualization, text=f'visualize masks: {int(self.show_masks)}')
        self.anc_toggle       = tkinter.Button(self.control_window, command=self.toggle_active_noise_cancellation, text=f'ANC: {int(self.noise_correction_en)}')
        self.next_sample      = tkinter.Button(self.control_window, command=self.change_sample, text='next')
        self.save_sample      = tkinter.Button(self.control_window, command=self.write_sample, text='save')
        self.smask_button.      grid(column=2, row=3, columnspan=4)
        self.viz_masks_button.  grid(column=2, row=4, columnspan=4)
        self.anc_toggle.        grid(column=2, row=11, columnspan=4)
        self.next_sample.       grid(column=2, row=5, columnspan=2)
        self.save_sample.       grid(column=4, row=5, columnspan=2)
        # noise parameters
        self.noise_label     = tkinter.Label(self.control_window, text=f'noise parameters')
        self.noise_label.    grid(column=2, row=6, columnspan=4)
        self.dx_noise_label  = tkinter.Label(self.control_window, text=f'std-x:')
        self.dx_noise_label. grid(column=2, row=7, columnspan=2)
        self.dx_noise_text   = tkinter.StringVar(value=f'{self.eval_cfg.se2_noise_dx_std}')
        self.dx_noise_entry  = tkinter.Entry(self.control_window, width=5, textvariable=self.dx_noise_text)
        self.dx_noise_entry. grid(column=4, row=7, columnspan=2)
        self.dy_noise_label  = tkinter.Label(self.control_window, text=f'std-y:')
        self.dy_noise_label. grid(column=2, row=8, columnspan=2)
        self.dy_noise_text   = tkinter.StringVar(value=f'{self.eval_cfg.se2_noise_dy_std}')
        self.dy_noise_entry  = tkinter.Entry(self.control_window, width=5, textvariable=self.dy_noise_text)
        self.dy_noise_entry. grid(column=4, row=8, columnspan=2)
        self.th_noise_label  = tkinter.Label(self.control_window, text=f'std-yaw:')
        self.th_noise_label. grid(column=2, row=9, columnspan=2)
        self.th_noise_text   = tkinter.StringVar(value=f'{self.eval_cfg.se2_noise_th_std}')
        self.th_noise_entry  = tkinter.Entry(self.control_window, width=5, textvariable=self.th_noise_text)
        self.th_noise_entry. grid(column=4, row=9, columnspan=2)
        self.apply_noise     = tkinter.Button(self.control_window, command=self.apply_noise_params, text='undertaker')
        self.apply_noise.    grid(column=2, row=10, columnspan=4)
        # space separator for left side of control window
        self.sep_2 = tkinter.Label(self.control_window, text='    ')
        self.sep_2.grid(row=1, rowspan=10, column=16, columnspan=2)
        # adjacency matrix buttons
        for j in range(8):
            for i in range(8):
                if i == 0:
                    exec(f"self.mlabel_{j}{i} = tkinter.Label(self.control_window, text={j + 1})")
                    exec(f"self.mlabel_{j}{i}.grid(column={j + 8}, row={i + 1})")
                if j == 0:
                    exec(f"self.mlabel_{j}{i} = tkinter.Label(self.control_window, text={i + 1})")
                    exec(f"self.mlabel_{j}{i}.grid(column={j + 7}, row={i + 2})")
                if i == j or eval_cfg.adjacency_init == 'ones':
                    button_text = '1'
                else:
                    button_text = '0'
                exec(f"self.mbutton_{j}{i} = tkinter.Button(self.control_window, text='{button_text}')")
                exec(f"self.mbutton_{j}{i}.configure(command=lambda: self.matrix_clicked({i}, {j}))", locals(), locals())
                exec(f"self.mbutton_{j}{i}.grid(column={j + 8}, row={i + 2})")
        # relative injected noise table
        self.relative_noise_table         = tkinter.Message(self.control_window, text='placeholder', anchor='w', width=600)
        self.relative_noise_table.        grid(column=7, row=12, columnspan=8, rowspan=9)
        # set default font
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=12)
        default_font.configure(family='Helvetica')
        self.root.option_add("*Font", default_font)

    def _refresh_agent_buttons(self):
        for i in range(8):
            if i == self.agent_index:
                exec(f"self.abutton_{i}.configure(font=('Helvetica', 10, 'bold', 'underline'))")
            else:
                exec(f"self.abutton_{i}.configure(font=('Helvetica', 10))")

    def _update_relative_noise_table(self, noise):
        noise_str = 'relative injected noise\n\n'
        noise_str += f'# |    x    |    y    | theta \n'
        noise_str += f'--+---------+---------+-------\n'
        for i in range(self.agent_count):
            theta = (noise[i, 2] * 180 / np.pi).item()
            if theta <= -180: theta += 360
            elif theta >= 180: theta -= 360
            noise_str += f'{i} | ' + \
                         f'{noise[i, 0].item():.3f}'.ljust(6) + '  | ' + \
                         f'{noise[i, 1].item():.3f}'.ljust(6) + '  | ' + \
                         f'{theta:.2f}'.ljust(6) + '\n'
        self.relative_noise_table.configure(text=noise_str)

    def write_sample(self):
        if len(self.visualized_data) == 0:
            return
        root_dir = os.path.join(self.eval_cfg.sample_save_dir, 'batch_' + str(self.batch_index.item()))
        agent_dir = os.path.join(root_dir, f'agent_{self.agent_index}')
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(agent_dir, exist_ok=True)
        for key, value in self.visualized_data.items():
            if isinstance(value, dict):
                network_dir = os.path.join(agent_dir, key)
                os.makedirs(network_dir, exist_ok=True)
                for k, v in value.items():
                    if k.endswith('_mask'):
                        cv2.imwrite(os.path.join(network_dir, f'{k}.png'), v * 255)
                    else:
                        v = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(network_dir, f'{k}.png'), v)
            else:
                if key.endswith('_mask'):
                    cv2.imwrite(os.path.join(agent_dir, f'{key}.png'), value * 255)
                else:
                    value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(agent_dir, f'{key}.png'), value)
        self.visualized_data.clear()

    def add_network(self, network: torch.nn.Module, label: str, graph_net: bool):
        id = len(self.networks)
        if label in self.networks:
            count = 2
            while True:
                if label + f'_v{count}' not in self.networks:
                    label = label + f'_v{count}'
                    break
                count += 1
        net_row = 2 + len(self.networks) * 13
        network.eval()
        self.networks[label] = network
        self.graph_flags[label] = graph_net
        exec(f"self.network_label_{id}        = tkinter.Label(self.viz_window, text='[{label}]')")
        exec(f"self.rgb_panel_{id}            = tkinter.Label(self.viz_window, text='placeholder')")
        exec(f"self.solo_pred_panel_{id}      = tkinter.Label(self.viz_window, text='placeholder')")
        exec(f"self.aggr_pred_panel_{id}      = tkinter.Label(self.viz_window, text='placeholder')")
        exec(f"self.aggr_trgt_panel_{id}      = tkinter.Label(self.viz_window, text='placeholder')")
        exec(f"self.zero_noiz_iou_label_{id}  = tkinter.Label(self.viz_window, text='network not evaluated')")
        exec(f"self.pass_noiz_iou_label_{id}  = tkinter.Label(self.viz_window, text='network not evaluated')")
        exec(f"self.actv_noiz_iou_label_{id}  = tkinter.Label(self.viz_window, text='network not evaluated')")
        exec(f"self.seperator_{id}            = tkinter.Label(self.viz_window, text='{'-' * 105}')")
        exec(f"self.network_label_{id}.         grid(column=0, row={net_row - 1}, columnspan=1)")
        exec(f"self.rgb_panel_{id}.             grid(column=0, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.solo_pred_panel_{id}.       grid(column=6, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.aggr_pred_panel_{id}.       grid(column=11, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.aggr_trgt_panel_{id}.       grid(column=16, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.zero_noiz_iou_label_{id}.   grid(column=0, row={net_row + 8}, columnspan=20)")
        exec(f"self.pass_noiz_iou_label_{id}.   grid(column=0, row={net_row + 9}, columnspan=20)")
        exec(f"self.actv_noiz_iou_label_{id}.   grid(column=0, row={net_row + 10}, columnspan=20)")
        exec(f"self.seperator_{id}.             grid(column=0, row={net_row + 11}, columnspan=20)")

    def assign_dataset_iterator(self, dset_iterator):
        self.dset_iterator = dset_iterator

    def start(self):
        if False not in self.graph_flags.values():
            print("warning: no baseline network has been added")
        self.change_sample()
        self.viz_window.mainloop()

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
        self.update_prediction(False)

    def agent_clicked(self, agent_id: int):
        if agent_id < self.agent_count:
            self.agent_index = agent_id
            self._refresh_agent_buttons()
            self.update_prediction(False)

    def toggle_mask_visualization(self):
        self.show_masks = not self.show_masks
        self.viz_masks_button.configure(text=f'visualize masks: {int(self.show_masks)}')
        self.update_prediction(False)

    def toggle_baseline_self_masking(self):
        self.baseline_masking_en = not self.baseline_masking_en
        self.smask_button.configure(text=f'filter baseline: {int(self.baseline_masking_en)}')
        self.update_prediction(False)

    def toggle_active_noise_cancellation(self):
        self.noise_correction_en =  not self.noise_correction_en
        self.anc_toggle.configure(text=f'ANC: {int(self.noise_correction_en)}')
        self.update_prediction(False)

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
        self.update_prediction(True)

    def calculate_ious(self, dataset: MassHDF5):
        dloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        total_length = len(dloader)
        print('calculating IoUs...')
        metrics = NetworkMetrics(self.networks, self.segclass_dict, self.device)

        for idx, (rgbs, labels, car_masks, fov_masks, car_transforms, _) in enumerate(dloader):
            print(f'\r{idx + 1}/{total_length}', end='')
            rgbs, labels, car_masks, fov_masks, car_transforms = to_device(
                self.device, rgbs, labels, car_masks, fov_masks, car_transforms
            )
            rgbs, labels, car_masks, fov_masks, car_transforms = squeeze_all(
                rgbs, labels, car_masks, fov_masks, car_transforms
            )

            for name, network in self.networks.items():
                metrics.update_network(
                    name, network, self.graph_flags[name], rgbs,
                    car_masks, fov_masks, car_transforms, labels, self.ppm,
                    self.output_h, self.output_w, self.center_x, self.center_y,
                    self.eval_cfg.mask_thresh, self.eval_cfg.se2_noise_dx_std,
                    self.eval_cfg.se2_noise_dy_std, self.eval_cfg.se2_noise_th_std
                )

        metrics.finish()

        for i, (network) in enumerate(self.networks.keys()):
            lines = []
            for (noise_type, ious) in metrics.metrics[network].items():
                noise_type = noise_type.replace('_', ' ')
                line = f'{noise_type}: '
                for semantic_idx, semantic_cls in self.segclass_dict.items():
                    line += f'{semantic_cls.lower()}:{ious["mskd"][semantic_idx].item():.2f} '
                if 'Mask' not in self.segclass_dict.values():
                    line += f'mask: {ious["mask_iou"]:.2f}'
                lines.append(line)
            exec(f"self.zero_noiz_iou_label_{i}.configure(text='{lines[0]}')")
            exec(f"self.pass_noiz_iou_label_{i}.configure(text='{lines[1]}')")
            exec(f"self.actv_noiz_iou_label_{i}.configure(text='{lines[2]}')")

        metrics.write_to_file('metrics.txt')
        print('\ndone')

    def calculate_inference_time(self, dataset: MassHDF5):
        dloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        total_length = len(dloader)
        print('calculating network inference times...')
        metrics = InferenceMetrics(self.networks, self.eval_cfg.max_agent_count)

        for idx, (rgbs, labels, car_masks, fov_masks, car_transforms, _) in enumerate(dloader):
            print(f'\r{idx + 1}/{total_length}', end='')
            rgbs, labels, car_masks, fov_masks, car_transforms = to_device(self.device, rgbs, labels,
                                                                           car_masks, fov_masks,
                                                                           car_transforms)
            rgbs, labels, car_masks, fov_masks, car_transforms = squeeze_all(rgbs, labels, car_masks,
                                                                             fov_masks, car_transforms)
            for name, network in self.networks.items():
                metrics.update_network(name, network, rgbs, car_masks, car_transforms)

        metrics.finish()
        metrics.write_to_file('./inference.txt')
        print('\ndone')

    def calculate_noise_cancellation(self, dataset: MassHDF5):
        dloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        total_length = len(dloader)
        print('evaluating noise cancellation...')
        metrics: List[NoiseMetrics] = []
        for network_label, network in self.networks.items():
            if hasattr(network, 'feat_matching_net'):
                metrics.append(
                    NoiseMetrics(
                        network_label, network, self.device, self.eval_cfg.max_agent_count
                    )
                )
            else:
                print(f'{network_label} does not have a noise estimating subnet. skipping...')

        for idx, (rgbs, labels, car_masks, fov_masks, car_transforms, _) in enumerate(dloader):
            print(f'\r{idx + 1}/{total_length}', end='')
            rgbs, labels, car_masks, fov_masks, car_transforms = to_device(
                self.device, rgbs, labels, car_masks, fov_masks, car_transforms
            )
            rgbs, labels, car_masks, fov_masks, car_transforms = squeeze_all(
                rgbs, labels, car_masks, fov_masks, car_transforms
            )
            noisy_transforms = get_noisy_transforms(
                car_transforms,
                self.eval_cfg.se2_noise_dx_std,
                self.eval_cfg.se2_noise_dy_std,
                self.eval_cfg.se2_noise_th_std
            )
            for noise_metric in metrics:
                noise_metric.update_network(
                    rgbs, car_masks, fov_masks,
                    car_transforms, noisy_transforms,
                    self.eval_cfg.se2_noise_dx_std, 
                    self.eval_cfg.se2_noise_dy_std,
                    self.eval_cfg.se2_noise_th_std,
                    self.output_h, self.output_w, self.ppm,
                    self.center_x, self.center_y
                )

        if len(metrics) > 1:
            for noise_metric in metrics:
                noise_metric.finish()
                noise_metric.write_to_file(f'{noise_metric.label}'[:20] + '_noizeval.txt')
        else:
            metrics[0].finish()
            metrics[0].write_to_file('noise.txt')

        print('\ndone')

    def change_sample(self):
        try:
            (rgbs, labels, car_masks, fov_masks, car_transforms, batch_index) = next(self.dset_iterator)
        except StopIteration:
            self.next_sample['state'] = 'disabled'
            self.root.title(f'end of dataset')
            return
        rgbs, labels, car_masks, fov_masks, car_transforms = to_device(self.device, rgbs, labels,
                                                                       car_masks, fov_masks,
                                                                       car_transforms)
        rgbs, labels, car_masks, fov_masks, car_transforms = squeeze_all(rgbs, labels, car_masks,
                                                                         fov_masks, car_transforms)
        self.current_data = (rgbs, labels, car_masks, fov_masks, car_transforms)
        self.agent_count = rgbs.shape[0]
        self.batch_index = batch_index
        if self.eval_cfg.adjacency_init == 'eye':
            self.adjacency_matrix = torch.eye(self.agent_count)
        else:
            self.adjacency_matrix = torch.ones((self.agent_count, self.agent_count))
        self.root.title(f'batch #{batch_index.squeeze().item()}')
        self.agent_index = 0
        self._refresh_agent_buttons()
        # enable/disable adjacency matrix buttons, reset their labels
        for j in range(8):
            for i in range(8):
                if i == j or self.eval_cfg.adjacency_init == 'ones':
                    button_text = '1'
                else:
                    button_text = '0'
                exec(f"self.mbutton_{j}{i}.configure(text='{button_text}')")
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
        self.update_prediction(True)

    def update_prediction(self, resample_noise):
        (rgbs, labels, car_masks, fov_masks, car_transforms) = self.current_data
        # keep track of visualization state for saving the images
        self.visualized_data.clear()
        # front RGB image
        rgb = rgbs[self.agent_index, ...].permute(1, 2, 0)
        rgb = ((rgb + 1) * 255 / 2).cpu().numpy().astype(np.uint8)
        self.visualized_data['rgb'] = rgb.copy()
        rgb = cv2.resize(rgb, (342, 256), cv2.INTER_LINEAR)
        rgb_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(rgb), 'RGB')
        # target image and mask
        ss_gt_img = convert_semantics_to_rgb(labels[self.agent_index].cpu(), self.semantic_classes)
        self.visualized_data['target_semantics'] = ss_gt_img.copy()
        if self.show_masks:
            aggr_gt_mask = get_single_adjacent_aggregate_mask(
                car_masks + fov_masks, car_transforms, self.agent_index, self.ppm,
                self.output_h, self.output_w, self.center_x, self.center_y,
                self.adjacency_matrix, True
            ).cpu().numpy()
            self.visualized_data['target_aggregated_mask'] = aggr_gt_mask
            if self.eval_cfg.transparent_masks:
                ss_gt_img[aggr_gt_mask == 0, :] = ss_gt_img[aggr_gt_mask == 0, :] / 1.5
            else:
                ss_gt_img[aggr_gt_mask == 0, :] = 0
            self.visualized_data['masked_target_semantics'] = ss_gt_img
        target_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(ss_gt_img), 'RGB')
        # add noise (important to do after the gt aggr. mask is calculated)
        if resample_noise:
            self.noisy_transforms = get_noisy_transforms(car_transforms,
                                                         self.eval_cfg.se2_noise_dx_std,
                                                         self.eval_cfg.se2_noise_dy_std,
                                                         self.eval_cfg.se2_noise_th_std)
        injected_noise_params = get_relative_noise(car_transforms, self.noisy_transforms, self.agent_index)
        self._update_relative_noise_table(injected_noise_params)
        # network outputs
        for i, (name, network) in enumerate(self.networks.items()):
            self.visualized_data[name] = {}
            # >>> front RGB image
            exec(f"self.rgb_panel_{i}.configure(image=rgb_tk)")
            exec(f"self.rgb_panel_{i}.image = rgb_tk")

            # >>> target image
            exec(f"self.aggr_trgt_panel_{i}.configure(image=target_tk)")
            exec(f"self.aggr_trgt_panel_{i}.image = target_tk")

            solo_sseg_pred, solo_mask_pred, \
            aggr_sseg_pred, aggr_mask_pred = network.get_eval_output(
                self.segclass_dict, self.graph_flags[name], rgbs, car_masks, fov_masks,
                self.noisy_transforms, self.adjacency_matrix, self.ppm, self.output_h, self.output_w,
                self.center_x, self.center_y, self.agent_index, self.baseline_masking_en,
                torch.device('cpu'), self.noise_correction_en
            )
            # thresholding masks
            solo_mask_pred[solo_mask_pred < self.eval_cfg.mask_thresh] = 0
            solo_mask_pred[solo_mask_pred >= self.eval_cfg.mask_thresh] = 1
            aggr_mask_pred[aggr_mask_pred < self.eval_cfg.mask_thresh] = 0
            aggr_mask_pred[aggr_mask_pred >= self.eval_cfg.mask_thresh] = 1
            # log estimated noise
            if self.eval_cfg.log_noise_estimate and hasattr(network, 'feat_matching_net'):
                estimated_noise_tf = network.feat_matching_net.estimated_noise[self.agent_index]
                estimated_noise_params = torch.zeros((self.agent_count, 3), dtype=car_transforms.dtype,
                                                     device=car_transforms.device)
                estimated_noise_params[:, :2] = estimated_noise_tf[:, :2, 3]
                estimated_noise_params[:,  2] = torch.atan2(estimated_noise_tf[:, 1, 0],
                                                            estimated_noise_tf[:, 0, 0])
                print(f'agent {self.agent_index} noise parameters in {name} ------- ')
                for a in range(self.agent_count):
                    print(f'w.r.t. agent {a}')
                    print(f'xx-noise: {injected_noise_params[a, 0].item():.4f} estimated {estimated_noise_params[a, 0].item():.4f}')
                    print(f'yy-noise: {injected_noise_params[a, 1].item():.4f} estimated {estimated_noise_params[a, 1].item():.4f}')
                    print(f'th-noise: {(injected_noise_params[a, 2] * 180 / np.pi).item():.2f}   '
                          f'estimated {(estimated_noise_params[a, 2] * 180 / np.pi).item():.2f}')
            solo_sseg_pred_img = convert_semantics_to_rgb(solo_sseg_pred.argmax(dim=0), self.semantic_classes)
            self.visualized_data[name]['solo_semantics'] = solo_sseg_pred_img.copy()
            self.visualized_data[name]['solo_mask'] = solo_mask_pred.numpy()
            self.visualized_data[name]['aggregated_mask'] = aggr_mask_pred.numpy()
            if self.show_masks:
                if self.eval_cfg.transparent_masks:
                    solo_sseg_pred_img[solo_mask_pred == 0, :] = solo_sseg_pred_img[solo_mask_pred == 0, :] / 1.5
                else:
                    solo_sseg_pred_img[solo_mask_pred == 0, :] = 0
                self.visualized_data[name]['masked_solo_semantics'] = solo_sseg_pred_img
            solo_sseg_pred_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(solo_sseg_pred_img), 'RGB')
            # >>> masked predicted semseg w/o external influence
            exec(f"self.solo_pred_panel_{i}.configure(image=solo_sseg_pred_tk)")
            exec(f"self.solo_pred_panel_{i}.image = solo_sseg_pred_tk")
            # >>> full predicted semseg w/ influence from adjacency matrix
            aggr_sseg_pred_img = convert_semantics_to_rgb(aggr_sseg_pred.argmax(dim=0),
                                                          self.semantic_classes)
            self.visualized_data[name]['aggregated_semantics'] = aggr_sseg_pred_img.copy()
            if self.show_masks:
                if self.eval_cfg.transparent_masks and self.graph_flags[name]:
                    aggr_sseg_pred_img[aggr_mask_pred == 0, :] = aggr_sseg_pred_img[aggr_mask_pred == 0, :] / 1.5
                else:
                    aggr_sseg_pred_img[aggr_mask_pred == 0, :] = 0
                self.visualized_data[name]['masked_aggregated_semantics'] = aggr_sseg_pred_img
            aggr_sseg_pred_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(aggr_sseg_pred_img), 'RGB')
            exec(f"self.aggr_pred_panel_{i}.configure(image=aggr_sseg_pred_tk)")
            exec(f"self.aggr_pred_panel_{i}.image = aggr_sseg_pred_tk")

def main():
    sem_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    eval_cfg = EvaluationConfig('config/evaluation.yml')
    device = torch.device(eval_cfg.device)
    torch.manual_seed(eval_cfg.torch_seed)
    # image geometry ---------------------------------------------------------------------------------------
    NEW_SIZE = (eval_cfg.output_h, eval_cfg.output_w)
    CENTER = (sem_cfg.center_x(NEW_SIZE[1]), sem_cfg.center_y(NEW_SIZE[0]))
    PPM = sem_cfg.pix_per_m(NEW_SIZE[0], NEW_SIZE[1])
    # gui object
    # evaluate the added networks
    if eval_cfg.classes == 'carla':
        segmentation_classes = color_map.__carla_classes
    elif eval_cfg.classes == 'ours':
        segmentation_classes = color_map.__our_classes
    elif eval_cfg.classes == 'ours+mask':
        segmentation_classes = color_map.__our_classes_plus_mask
    elif eval_cfg.classes == 'diminished':
        segmentation_classes = color_map.__diminished_classes
    elif eval_cfg.classes == 'diminished+mask':
        segmentation_classes = color_map.__diminished_classes_plus_mask
    else:
        raise ValueError('Unknown class set')
    gui = SampleWindow(eval_cfg, segmentation_classes, device, NEW_SIZE, CENTER, PPM)
    # dataloader stuff -------------------------------------------------------------------------------------
    test_set = MassHDF5(dataset=eval_cfg.dset_name, path=eval_cfg.dset_dir,
                        hdf5name=eval_cfg.dset_file, size=NEW_SIZE,
                        classes=eval_cfg.classes, jitter=[0, 0, 0, 0],
                        mask_gaussian_sigma=eval_cfg.gaussian_mask_std,
                        guassian_kernel_size=eval_cfg.gaussian_kernel_size)
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    gui.assign_dataset_iterator(iter(loader))
    # other network stuff ----------------------------------------------------------------------------------
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
        gui.add_network(model, eval_cfg.runs[i], eval_cfg.model_gnn_flags[i])
    # evaluate the added networks --------------------------------------------------------------------------
    if eval_cfg.evaluate_ious_at_start:
        gui.calculate_ious(test_set)
    else:
        print('iou calculation disabled.')
    if eval_cfg.profile_at_start:
        gui.calculate_inference_time(test_set)
    else:
        print('inference time profiling disabled.')
    if eval_cfg.evaluate_noise_at_start:
        gui.calculate_noise_cancellation(test_set)
    else:
        print('noise evaluation disabled.')
    # start the gui ----------------------------------------------------------------------------------------
    print('starting gui...')
    gui.start()

if __name__ == '__main__':
    main()