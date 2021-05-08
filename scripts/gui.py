import os
import cv2
import torch
import kornia
import tkinter
import numpy as np
import PIL.ImageTk
import PIL.Image as PILImage

from data.dataset import get_datasets
from data.config import SemanticCloudConfig, TrainingConfig, EvaluationConfig
from data.color_map import our_semantics_to_cityscapes_rgb
from data.mask_warp import get_single_relative_img_transform
from data.utils import squeeze_all, to_device
from model.large_mcnn import LMCNN, LWMCNN
from model.mcnn import MCNN, MCNN4

class SampleWindow:
    def __init__(self, class_count: int, device: torch.device,
                 tcfg: TrainingConfig, new_size, center, ppm):
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
        self.window = tkinter.Tk()
        self.agent_index = 0
        self.agent_count = 8
        self.agent_label  = tkinter.Label(self.window, text=f'agent {self.agent_index}/{self.agent_count}')
        self.next_agent   = tkinter.Button(self.window, command=lambda: self.next_prev_clicked(True), text='next agent')
        self.prev_agent   = tkinter.Button(self.window, command=lambda: self.next_prev_clicked(False), text='prev agent')
        self.next_sample  = tkinter.Button(self.window, command=self.change_sample, text='next sample')
        self.smask_button = tkinter.Button(self.window, command=self.toggle_self_masking, text=f'self mask: {int(self.self_masking_en)}')
        self.agent_label. grid(column=21, row=0)
        self.next_agent.  grid(column=21, row=1)
        self.prev_agent.  grid(column=21, row=2)
        self.next_sample. grid(column=21, row=3)
        self.smask_button.grid(column=21, row=4)
        self.rgb_panel         = tkinter.Label(self.window, text='placeholder')
        self.masked_pred_panel = tkinter.Label(self.window, text='placeholder')
        self.full_pred_panel   = tkinter.Label(self.window, text='placeholder')
        self.target_panel      = tkinter.Label(self.window, text='placeholder')
        self.rgb_panel.         grid(column=0, row=1, columnspan=5, rowspan=8)
        self.masked_pred_panel. grid(column=6, row=1, columnspan=5, rowspan=8)
        self.full_pred_panel.   grid(column=11, row=1, columnspan=5, rowspan=8)
        self.target_panel.      grid(column=16, row=1, columnspan=5, rowspan=8)
        self.rgb_panel_caption         = tkinter.Label(self.window, text='front rgb')
        self.masked_pred_panel_caption = tkinter.Label(self.window, text='masked pred')
        self.full_pred_panel_caption   = tkinter.Label(self.window, text='full pred')
        self.target_panel_caption      = tkinter.Label(self.window, text='target')
        self.rgb_panel_caption.         grid(column=0, row=0, columnspan=5)
        self.masked_pred_panel_caption. grid(column=6, row=0, columnspan=5)
        self.full_pred_panel_caption.   grid(column=11, row=0, columnspan=5)
        self.target_panel_caption.      grid(column=16, row=0, columnspan=5)

    def assign_dataset(self, dset_iterator):
        self.dset_iterator = dset_iterator

    def assign_network(self, net: torch.nn.Module):
        self.network = net

    def start(self):
        for j in range(8):
            for i in range(8):
                exec(f"self.button_{j}{i} = tkinter.Button(self.window, text='1' if {i} == {j} else '0')")
                exec(f"self.button_{j}{i}.configure(command=lambda: self.matrix_clicked({i}, {j}))", locals(), locals())
                exec(f"self.button_{j}{i}.grid(column={j + 22}, row={i})")

        self.window.title('MASS GUI')
        self.change_sample()
        self.window.mainloop()

    def matrix_clicked(self, i: int, j: int):
        if self.adjacency_matrix[j, i] == 1:
            self.adjacency_matrix[j, i] = 0
            self.adjacency_matrix[i, j] = 0
            exec(f"self.button_{j}{i}.configure(text='0')")
            exec(f"self.button_{i}{j}.configure(text='0')")
        else:
            self.adjacency_matrix[j, i] = 1
            self.adjacency_matrix[i, j] = 1
            exec(f"self.button_{j}{i}.configure(text='1')")
            exec(f"self.button_{i}{j}.configure(text='1')")
        self.update_prediction()

    def next_prev_clicked(self, next: bool):
        if next:
            self.agent_index = min(self.agent_index + 1, self.agent_count - 1)
        else:
            self.agent_index = max(self.agent_index - 1, 0)
        self.agent_label.configure(text=f'agent {self.agent_index + 1}/{self.agent_count}')
        self.update_prediction()
    
    def toggle_self_masking(self):
        self.self_masking_en = not self.self_masking_en
        self.smask_button.configure(text=f'self mask: {int(self.self_masking_en)}')
        self.update_prediction()

    def change_sample(self):
        (_, rgbs, labels, masks, car_transforms, _) = next(self.dset_iterator)
        rgbs, labels, masks, car_transforms = to_device(rgbs, labels,
                                                        masks, car_transforms,
                                                        self.device, False)
        rgbs, labels, masks, car_transforms = squeeze_all(rgbs, labels, masks, car_transforms)
        self.current_data = (rgbs, labels, masks, car_transforms)
        self.agent_count = rgbs.shape[0]
        self.adjacency_matrix = torch.eye(self.agent_count)
        self.agent_index = 0
        self.agent_label.configure(text=f'agent {self.agent_index + 1}/{self.agent_count}')
        # enable/disable matrix
        for j in range(8):
            for i in range(8):
                if i < self.agent_count and j < self.agent_count:
                    exec(f"self.button_{j}{i}['state'] = 'normal'")
                else:
                    exec(f"self.button_{j}{i}['state'] = 'disabled'")
        self.update_prediction()

    def update_prediction(self):
        (rgbs, labels, _, car_transforms) = self.current_data
        # prediction for combined prediction
        with torch.no_grad():
            sseg_preds, mask_preds = self.network(rgbs, car_transforms, torch.eye(self.agent_count))
        
        # front RGB image
        rgb = rgbs[self.agent_index, ...].permute(1, 2, 0)
        rgb = ((rgb + 1) * 255 / 2).cpu().numpy().astype(np.uint8)
        rgb = cv2.resize(rgb, (342, 256), cv2.INTER_LINEAR)
        rgb_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(rgb), 'RGB')
        self.rgb_panel.configure(image=rgb_tk)
        self.rgb_panel.image = rgb_tk

        # target image
        ss_gt_img = our_semantics_to_cityscapes_rgb(labels[self.agent_index].cpu())
        target_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(ss_gt_img), 'RGB')
        self.target_panel.configure(image=target_tk)
        self.target_panel.image = target_tk

        # masked predicted semseg
        ss_pred = sseg_preds[self.agent_index].argmax(dim=0)
        ss_pred_img = our_semantics_to_cityscapes_rgb(ss_pred.cpu())
        mask_pred = mask_preds[self.agent_index].squeeze().cpu()
        ss_pred_img[mask_pred == 0] = 0
        masked_pred_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(ss_pred_img), 'RGB')
        self.masked_pred_panel.configure(image=masked_pred_tk)
        self.masked_pred_panel.image = masked_pred_tk

        # full predicted semseg
        if self.self_masking_en:
            sseg_preds *= mask_preds
        not_selected = torch.where(self.adjacency_matrix[self.agent_index] == 0)[0]
        relative_tfs = get_single_relative_img_transform(car_transforms, self.agent_index,
                                                         self.ppm, self.output_h, self.output_w,
                                                         self.center_x, self.center_y).to(self.device)
        warped_semantics = kornia.warp_affine(sseg_preds, relative_tfs, dsize=(self.output_h, self.output_w),
                                              flags='nearest')
        # applying adjacency matrix
        warped_semantics[not_selected] = 0
        warped_semantics = warped_semantics.sum(dim=0).argmax(dim=0)
        aggregated_semantics = our_semantics_to_cityscapes_rgb(warped_semantics.cpu())
        full_pred_tk = PIL.ImageTk.PhotoImage(PILImage.fromarray(aggregated_semantics), 'RGB')
        self.full_pred_panel.configure(image=full_pred_tk)
        self.full_pred_panel.image = full_pred_tk

def main():
    train_cfg = TrainingConfig('config/training.yml')
    sem_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
    eval_cfg = EvaluationConfig('config/evaluation.yml')
    device = torch.device(eval_cfg.device)
    torch.manual_seed(train_cfg.torch_seed)
    # image geometry
    NEW_SIZE = (train_cfg.output_h, train_cfg.output_w)
    CENTER = (sem_cfg.center_x(NEW_SIZE[1]), sem_cfg.center_y(NEW_SIZE[0]))
    PPM = sem_cfg.pix_per_m(NEW_SIZE[0], NEW_SIZE[1])
    # gui object
    gui = SampleWindow(train_cfg.num_classes, device, train_cfg, NEW_SIZE, CENTER, PPM)
    # dataloader stuff
    _, test_set = get_datasets(train_cfg.dset_name, train_cfg.dset_dir,
                               train_cfg.dset_file, (0.8, 0.2),
                               NEW_SIZE, train_cfg.classes)
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    gui.assign_dataset(iter(loader))
    # network stuff
    if eval_cfg.model_version != 'best' and eval_cfg.model_version != 'last':
        print("valid model version are 'best' and 'last'")
        exit()
    train_cfg.snapshot_dir = train_cfg.snapshot_dir.format(eval_cfg.run)
    snapshot_path = f'{eval_cfg.model_version}_model.pth'
    snapshot_path = train_cfg.snapshot_dir + '/' + snapshot_path
    if not os.path.exists(snapshot_path):
        print(f'{snapshot_path} does not exist')
        exit()
    if eval_cfg.model_name == 'mcnn':
        model = MCNN(train_cfg.num_classes, NEW_SIZE,
                     sem_cfg, eval_cfg.aggregation_type).to(device)
    elif eval_cfg.model_name == 'mcnn4':
        model = MCNN4(train_cfg.num_classes, NEW_SIZE,
                      sem_cfg, eval_cfg.aggregation_type).to(device)
    elif eval_cfg.model_name == 'mcnnL':
        model = LMCNN(train_cfg.num_classes, NEW_SIZE,
                      sem_cfg, eval_cfg.aggregation_type,
                      eval_cfg.aggregation_activation_limit,
                      eval_cfg.average_aggregation).to(device)
    elif eval_cfg.model_name == 'mcnnLW':
        model = LWMCNN(train_cfg.num_classes, NEW_SIZE,
                       sem_cfg, eval_cfg.aggregation_type,
                       eval_cfg.aggregation_activation_limit,
                       eval_cfg.average_aggregation).to(device)
    else:
        print('unknown network architecture {eval_cfg.model_name}')
        exit()
    model.load_state_dict(torch.load(snapshot_path))
    gui.assign_network(model)
    # start the gui
    gui.start()

if __name__ == '__main__':
    main()