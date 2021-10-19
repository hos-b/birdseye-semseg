import torch
from data.utils import get_noisy_transforms

def get_iou_per_class(predictions: torch.Tensor, labels: torch.Tensor, target_sseg_mask: torch.Tensor, num_classes=7) -> torch.Tensor:
    """
    returns a [num_classes x 1] tensor containing the sum iou of each class for all images.
    in the end, the aggregated ious should be devided by the number of images, skipped here
    because the batch sizes differ.
    """
    assert len(predictions.shape) == 4, f"expected [B x num_classes x H x W], got {predictions.shape}"
    assert len(labels.shape) == 3, f"expected [B x H x W], got {labels.shape}"
    assert predictions.shape[1] == num_classes, f"expected second dim to be {num_classes}, got {predictions.shape[1]}"
    bool_mask = target_sseg_mask == 1
    pred_argmax = torch.argmax(predictions, dim=1)
    ious = torch.zeros((num_classes, 1), dtype=torch.float64)
    for i in range(num_classes):
        pred_class_i = pred_argmax == i
        labl_class_i = labels == i
        # image lvl iou for each class
        intersection = (pred_class_i & labl_class_i & bool_mask).sum(dim=(1, 2))
        union = ((pred_class_i | labl_class_i) & bool_mask).sum(dim=(1, 2))
        iou = intersection / union
        # set NaNs to zero
        iou[iou != iou] = 0
        ious[i] = iou.sum()
    return ious

def get_mask_iou(predictions: torch.Tensor, gt_masks: torch.Tensor, detection_tresh):
    assert len(predictions.shape) == len(gt_masks.shape), \
           f'dimensions of predictions {predictions.shape} != ground truth {gt_masks.shape}'
    preds = predictions.clone()
    preds[preds >= detection_tresh] = 1.0
    preds[preds < detection_tresh] = 0.0
    preds = preds == 1.0
    labels = gt_masks == 1.0
    # image lvl iou for mask
    intersection = (preds & labels).sum(dim=(1, 2))
    union = (preds | labels).sum(dim=(1, 2))
    iou = intersection / union
    # set NaNs to zero
    iou[iou != iou] = 0
    return iou.sum()

class NetworkMetrics:
    def __init__(self, networks: dict, class_dict: dict, device: torch.device):
        self.metrics = {}
        self.class_dict = class_dict
        class_count = len(class_dict)
        for key in networks.keys():
            self.metrics[key] = {
                'no_noise': {
                    'mskd': torch.zeros((class_count, 1), dtype=torch.float64).to(device),
                    'full': torch.zeros((class_count, 1), dtype=torch.float64).to(device),
                    'mask_iou': 0.0
                },
                'pa_noise': {
                    'mskd': torch.zeros((class_count, 1), dtype=torch.float64).to(device),
                    'full': torch.zeros((class_count, 1), dtype=torch.float64).to(device),
                    'mask_iou': 0.0
                },
                'ac_noise': {
                    'mskd': torch.zeros((class_count, 1), dtype=torch.float64).to(device),
                    'full': torch.zeros((class_count, 1), dtype=torch.float64).to(device),
                    'mask_iou': 0.0
                }
            }
        self.sample_count = 0

    def update_network(self, network_label: str, network: torch.nn.Module,
                       graph_flag, rgbs, car_masks, fov_masks, gt_transforms,
                       labels, ppm, output_h, output_w, center_x, center_y,
                       mask_thresh, noise_std_x, noise_std_y, noise_std_theta):
        
        self.sample_count += rgbs.shape[0]
        noisy_transforms = get_noisy_transforms(gt_transforms,
                                                noise_std_x,
                                                noise_std_y,
                                                noise_std_theta)
        # no noise, no correction
        batch_mskd_ious, batch_full_ious, mask_iou = network.get_batch_ious(
            self.class_dict, graph_flag, rgbs, car_masks, fov_masks,
            gt_transforms, gt_transforms, labels, ppm, output_h, output_w,
            center_x, center_y, mask_thresh, False
        )
        self.metrics[network_label]['no_noise']['mask_iou'] += mask_iou
        self.metrics[network_label]['no_noise']['mskd'] += batch_mskd_ious
        self.metrics[network_label]['no_noise']['full'] += batch_full_ious
        # with noise, no correction
        batch_mskd_ious, batch_full_ious, mask_iou = network.get_batch_ious(
            self.class_dict, graph_flag, rgbs, car_masks, fov_masks,
            gt_transforms, noisy_transforms, labels, ppm, output_h, output_w,
            center_x, center_y, mask_thresh, False
        )
        self.metrics[network_label]['pa_noise']['mask_iou'] += mask_iou
        self.metrics[network_label]['pa_noise']['mskd'] += batch_mskd_ious
        self.metrics[network_label]['pa_noise']['full'] += batch_full_ious
        # with noise, with correction
        batch_mskd_ious, batch_full_ious, mask_iou = network.get_batch_ious(
            self.class_dict, graph_flag, rgbs, car_masks, fov_masks,
            gt_transforms, noisy_transforms, labels, ppm, output_h, output_w,
            center_x, center_y, mask_thresh, True
        )
        self.metrics[network_label]['ac_noise']['mask_iou'] += mask_iou
        self.metrics[network_label]['ac_noise']['mskd'] += batch_mskd_ious
        self.metrics[network_label]['ac_noise']['full'] += batch_full_ious
    
    def finish(self):
        for network in self.metrics.keys():
            for key in self.metrics[network].keys():
                self.metrics[network][key]['mskd'] /= (self.sample_count / len(self.metrics.keys()))
                self.metrics[network][key]['full'] /= (self.sample_count / len(self.metrics.keys()))
                self.metrics[network][key]['mask_iou'] /= (self.sample_count / len(self.metrics.keys()))
                self.metrics[network][key]['mskd'] *= 100.0
                self.metrics[network][key]['full'] *= 100.0
                self.metrics[network][key]['mask_iou'] *= 100.0
        self.sample_count = 0

    def write_to_file(self, file_path: str):
        headers = [cls.lower()[:4] for cls in self.class_dict.values()]
        if 'mask' not in headers:
            headers.append('mask')
        file = open(file_path, 'w')
        file.write(f'network\t\t\t{" || ".join(headers)}\n')
        for (net_label, net_dict) in self.metrics.items():
            lines = [net_label + ' ' + '=' * (76 - len(net_label)) + '\n']
            for (noise_type, ious) in net_dict.items():
                noise_type = noise_type.replace('_', ' ')
                for mask_type in ['full', 'mskd']:
                    line = f'{noise_type}:{mask_type}\t'
                    for semantic_idx in self.class_dict:
                        line += f'{ious[mask_type][semantic_idx].item():.2f}\t'
                    if 'Mask' not in self.class_dict.values():
                        line += f'{ious["mask_iou"]:.2f}'
                    lines.append(line + '\n')
            file.writelines(lines)
        file.close()
