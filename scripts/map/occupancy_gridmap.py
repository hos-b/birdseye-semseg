import torch
import kornia

from data.config import SemanticCloudConfig
from data.mask_warp import get_single_relative_img_transform
""" 
this class implements an occupancy grid map. each pixel keeps
track of all the classes using their corresponding log odds.
the image can be requested by taking the argmax
"""
class GridMap:
    def __init__(self, class_count: int, cfg: SemanticCloudConfig, device):
        self.class_count = class_count
        self.device = device
        self.ppm = cfg.pix_per_m()
        self.center_x = cfg.center_x()
        self.center_y = cfg.center_y()
        self.map = torch.zeros((class_count, cfg.image_rows, cfg.image_cols),
                               dtype=torch.float64).to(device)

    def calculate_logodds(self, i: int, sem_block: torch.Tensor,
                          transforms: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        takes a predicted semantic block & produces the
        calculated logodds based on the given adjacency matrix
        """
        outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
        # B x 7 x 256 x 205
        relative_tfs = get_single_relative_img_transform(transforms, i, self.ppm, self.cf_h, self.cf_w,
                                                         self.center_x, self.center_y).to(self.device)
        warped_semantics = kornia.warp_affine(sem_block, relative_tfs, dsize=(self.cf_h, self.cf_w),
                                              mode='nearest')
        # applying the adjacency matrix
        warped_semantics[outside_fov] = 0
        # 7 x 256 x 205, probably not even necessary
        probs = torch.nn.Softmax(warped_semantics, dim=0)
        logodds = torch.log(probs / (1 - probs))
        # 256 x 205
        return torch.argmax(logodds, dim=0)
