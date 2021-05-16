import os
from tkinter.constants import HORIZONTAL
import cv2
import torch
import kornia
import tkinter
import numpy as np
import PIL.ImageTk
import PIL.Image as PILImage

from data.dataset import get_datasets
from data.config import SemanticCloudConfig, EvaluationConfig
from data.color_map import our_semantics_to_cityscapes_rgb
from data.mask_warp import get_single_relative_img_transform
from data.utils import squeeze_all, to_device
from model.large_mcnn import LMCNN, LWMCNN
from model.mcnn import MCNN, MCNN4

class SampleWindow:
    def __init__(self, class_count: int, device: torch.device, new_size, center, ppm):
        # network stuff
        self.networks = {}
        self.output_h = new_size[0]
        self.output_w = new_size[1]
        self.center_x = center[0]
        self.center_y = center[1]
        self.ppm = ppm
        self.class_count = class_count
        self.device = device
        self.current_data = None
        self.adjacency_matrix = None
        self.self_masking_en = False
        self.agent_index = 0
        self.agent_count = 8
        self.window = tkinter.Tk()
        # # image panels
        # self.rgb_panel         = tkinter.Label(self.window, text='placeholder')
        # self.masked_pred_panel = tkinter.Label(self.window, text='placeholder')
        # self.full_pred_panel   = tkinter.Label(self.window, text='placeholder')
        # self.target_panel      = tkinter.Label(self.window, text='placeholder')
        # self.rgb_panel.         grid(column=0, row=2, columnspan=5, rowspan=8)
        # self.masked_pred_panel. grid(column=6, row=2, columnspan=5, rowspan=8)
        # self.full_pred_panel.   grid(column=11, row=2, columnspan=5, rowspan=8)
        # self.target_panel.      grid(column=16, row=2, columnspan=5, rowspan=8)
        self.rgb_panel_caption         = tkinter.Label(self.window, text='front rgb')
        self.masked_pred_panel_caption = tkinter.Label(self.window, text='masked pred')
        self.full_pred_panel_caption   = tkinter.Label(self.window, text='full pred')
        self.target_panel_caption      = tkinter.Label(self.window, text='target')
        self.rgb_panel_caption.         grid(column=0, row=0, columnspan=5)
        self.masked_pred_panel_caption. grid(column=6, row=0, columnspan=5)
        self.full_pred_panel_caption.   grid(column=11, row=0, columnspan=5)
        self.target_panel_caption.      grid(column=16, row=0, columnspan=5)
        # self.baseline_label            = tkinter.Label(self.window, text='[baseline]')
        # self.baseline_label.            grid(column=0, row=0, columnspan=1)
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

    def add_network(self, network: torch.nn.Module, label: str):
        id = len(self.networks)
        net_row = 2 + len(self.networks) * 9
        network.eval()
        self.networks[label] = network
        exec(f"self.network_label_{id}     = tkinter.Label(self.window, text='[{label}]')")
        exec(f"self.rgb_panel_{id}         = tkinter.Label(self.window, text='placeholder')")
        exec(f"self.masked_pred_panel_{id} = tkinter.Label(self.window, text='placeholder')")
        exec(f"self.full_pred_panel_{id}   = tkinter.Label(self.window, text='placeholder')")
        exec(f"self.target_panel_{id}      = tkinter.Label(self.window, text='placeholder')")
        exec(f"self.network_label_{id}.     grid(column=0, row={net_row - 1}, columnspan=1)")
        exec(f"self.rgb_panel_{id}.         grid(column=0, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.masked_pred_panel_{id}. grid(column=6, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.full_pred_panel_{id}.   grid(column=11, row={net_row}, columnspan=5, rowspan=8)")
        exec(f"self.target_panel_{id}.      grid(column=16, row={net_row}, columnspan=5, rowspan=8)")

    def assign_dataset(self, dset_iterator):
        self.dset_iterator = dset_iterator

    def set_baseline(self, net: torch.nn.Module):
        self.baseline = net
        self.baseline.eval()

    def start(self):
        baseline_available = False
        for name in self.networks.keys():
            if name == 'baseline':
                baseline_available = True
                break
        if not baseline_available:
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

    def change_sample(self):
        (_, rgbs, labels, masks, car_transforms, batch_index) = next(self.dset_iterator)
        rgbs, labels, masks, car_transforms = to_device(rgbs, labels,
                                                        masks, car_transforms,
                                                        self.device, False)
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
        # front RGB image
        rgb = rgbs[self.agent_index, ...].permute(1, 2, 0)
        rgb = ((rgb + 1) * 255 / 2).cpu().numpy().astype(np.uint8)
        rgb = cv2.resize(rgb, (342, 256), cv2.INTER_LINEAR)
        rgb_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(rgb), 'RGB')

        # target image
        ss_gt_img = our_semantics_to_cityscapes_rgb(labels[self.agent_index].cpu())
        target_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(ss_gt_img), 'RGB')

        for i, (name, network) in enumerate(self.networks.items()):
            # combined prediction
            with torch.no_grad():
                if name == 'baseline':
                    all_ss_preds, all_mask_preds = network(rgbs, car_transforms, torch.eye(self.agent_count))
                else:
                    all_ss_preds, all_mask_preds = network(rgbs, car_transforms, self.adjacency_matrix)

            # >>> front RGB image
            exec(f"self.rgb_panel_{i}.configure(image=rgb_tk)")
            exec(f"self.rgb_panel_{i}.image = rgb_tk")

            # >>> target image
            exec(f"self.target_panel_{i}.configure(image=target_tk)")
            exec(f"self.target_panel_{i}.image = target_tk")

            # >>> masked predicted semseg w/o external influence
            if name == 'baseline':
                current_ss_pred = all_ss_preds[self.agent_index].argmax(dim=0)
                current_ss_pred_img = our_semantics_to_cityscapes_rgb(current_ss_pred.cpu())
                current_mask_pred = all_mask_preds[self.agent_index].squeeze().cpu()
                current_ss_pred_img[current_mask_pred == 0] = 0
                masked_pred_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(current_ss_pred_img), 'RGB')
                exec(f"self.masked_pred_panel_{i}.configure(image=masked_pred_tk)")
                exec(f"self.masked_pred_panel_{i}.image = masked_pred_tk")
            else:
                all_ss_eye, all_mask_eye = network(rgbs, car_transforms, torch.eye(self.agent_count))
                current_ss_pred = all_ss_eye[self.agent_index].argmax(dim=0)
                current_ss_pred_img = our_semantics_to_cityscapes_rgb(current_ss_pred.cpu())
                current_mask_pred = all_mask_eye[self.agent_index].squeeze().cpu()
                current_ss_pred_img[current_mask_pred == 0] = 0
                masked_pred_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(current_ss_pred_img), 'RGB')
                exec(f"self.masked_pred_panel_{i}.configure(image=masked_pred_tk)")
                exec(f"self.masked_pred_panel_{i}.image = masked_pred_tk")

            # >>> full predicted semseg w/ influence from adjacency matrix

            # using self masking (only for baseline since others do latent masking)
            if self.self_masking_en and name == 'baseline':
                all_ss_preds *= all_mask_preds

            # applying adjacency matrix (others already do in forward pass)
            if name == 'baseline':
                not_selected = torch.where(self.adjacency_matrix[self.agent_index] == 0)[0]
                relative_tfs = get_single_relative_img_transform(car_transforms, self.agent_index,
                                                                 self.ppm, self.output_h, self.output_w,
                                                                 self.center_x, self.center_y).to(self.device)
                current_warped_semantics = kornia.warp_affine(all_ss_preds, relative_tfs,
                                                              dsize=(self.output_h, self.output_w),
                                                              flags='nearest')
                current_warped_semantics[not_selected] = 0
                current_warped_semantics = current_warped_semantics.sum(dim=0).argmax(dim=0)
                current_aggregated_semantics = our_semantics_to_cityscapes_rgb(current_warped_semantics.cpu())
            else:
                current_aggregated_semantics = \
                    our_semantics_to_cityscapes_rgb(all_ss_preds[self.agent_index].argmax(dim=0).cpu())

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
    gui = SampleWindow(eval_cfg.num_classes, device, NEW_SIZE, CENTER, PPM)
    # dataloader stuff
    _, test_set = get_datasets(eval_cfg.dset_name, eval_cfg.dset_dir,
                               eval_cfg.dset_file, (0.8, 0.2),
                               NEW_SIZE, eval_cfg.classes)
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    gui.assign_dataset(iter(loader))
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
        model = MCNN(eval_cfg.num_classes, NEW_SIZE,
                     sem_cfg, eval_cfg.aggregation_type).to(device)
    elif eval_cfg.model_name == 'mcnn4':
        model = MCNN4(eval_cfg.num_classes, NEW_SIZE,
                      sem_cfg, eval_cfg.aggregation_type).to(device)
    elif eval_cfg.model_name == 'mcnnL':
        model = LMCNN(eval_cfg.num_classes, NEW_SIZE,
                      sem_cfg, eval_cfg.aggregation_type).to(device)
    elif eval_cfg.model_name == 'mcnnLW':
        model = LWMCNN(eval_cfg.num_classes, NEW_SIZE,
                       sem_cfg, eval_cfg.aggregation_type).to(device)
    else:
        print('unknown network architecture {eval_cfg.model_name}')
        exit()
    model.load_state_dict(torch.load(snapshot_path))
    gui.add_network(model, 'baseline')
    gui.add_network(model, 'baseline as a normal net')
    # start the gui
    gui.start()

if __name__ == '__main__':
    main()