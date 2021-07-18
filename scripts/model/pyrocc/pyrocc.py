import torch
import kornia
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.pyrocc.fpn import FPN50
from model.pyrocc.pyramid import TransformerPyramid
from model.pyrocc.topdown import TopdownNetwork
from model.pyrocc.classifier import LinearClassifier, BayesianClassifier
from data.utils import get_vehicle_masks
from data.config import SemanticCloudConfig
from data.mask_warp import get_single_relative_img_transform
from operator import mul
from functools import reduce

# map_resolution: spacing between adjacent grid cells in the map (m)
# tfm_channels: number of channels used in the dense transformer layers
# focal length: obvious
# topdown_channels, strides, layers: ...

class PyramidOccupancyNetwork(nn.Module):
    def __init__(self,
                 # our args
                 num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str,
                 # pyrocc args
                 tfm_channels = 32, bayesian_classifer = False,
                 map_extents = [-10.0, 3.1, 10.0, 20.0], ymin = -2, ymax = 4,
                 topdown_channels = 64, topdown_strides = [1, 2], topdown_layers = [4, 4],
                 topdown_blocktype = 'bottleneck', prior = None):
        super().__init__()

        # calculating camera geometry etc
        rgb_w, rgb_h, fov = 640, 480, 60.0
        focal_length = rgb_w / (2 * np.tan(fov * np.pi / 360))
        self.calib = torch.tensor([[focal_length, 0,            rgb_w / 2.0],
                                   [0,            focal_length, rgb_h / 2.0],
                                   [0,            0,            1           ]],
                                   dtype=torch.float32).cuda()
        map_resolution = 1 / sem_cfg.pix_per_m(output_size[0], output_size[1])
        # Build frontend
        self.frontend = FPN50()
        # Build transformer pyramid
        tfm_resolution = map_resolution * reduce(mul, topdown_strides)
        self.transformer = TransformerPyramid(256, tfm_channels, tfm_resolution,
                                              map_extents, ymin, ymax, focal_length)
        # Build topdown network
        self.topdown = TopdownNetwork(tfm_channels, topdown_channels, output_size,
                                      topdown_layers, [1, 1], topdown_blocktype)
        # Build classifiers
        if bayesian_classifer:
            self.classifier = BayesianClassifier(self.topdown.out_channels, num_classes)
        else:
            self.classifier = LinearClassifier(self.topdown.out_channels, num_classes)

        if prior:
            self.classifier.initialise(prior)

        self.output_size = output_size
        # aggregation parameters
        self.aggregation_type = aggr_type
        self.cf_h, self.cf_w = 134, 103
        self.ppm = sem_cfg.pix_per_m(self.cf_h, self.cf_w) # 5.255
        self.center_x = sem_cfg.center_x(self.cf_w) # 51
        self.center_y = sem_cfg.center_y(self.cf_h) # 107

    def forward(self, image, transforms, adjacency_matrix, car_masks):
        # image: [B, 3, 640, 480]
        # Extract multiscale feature maps
        # 0: [B, 256, 60, 80]
        # 1: [B, 256, 30, 40]
        # 2: [B, 256, 15, 20]
        # 3: [B, 256, 8 , 10] [currently disabled]
        # 4: [B, 256, 4 ,  5] [currently disabled]
        feature_maps = self.frontend(image)
        # transform image features to birds-eye-view
        # [B, 32, 134, 103] from:
        # -- [B, 32, 34, 103] <- inside FoV
        # -- [B, 32, 35, 103] <- inside FoV
        # -- [B, 32, 17, 103] <- inside FoV
        # -- [B, 32, 48, 103] <- outside FoV [currently zeroed out]
        bev_feats = self.transformer(feature_maps, self.calib, 48)
        # add ego car masks
        bev_feats += F.interpolate(car_masks.unsqueeze(1), size=(self.cf_h, self.cf_w),
                                   mode='bilinear')
        # [B, 32, 134, 103]
        bev_feats = self.aggregate_features(bev_feats, transforms, adjacency_matrix)
        # apply topdown network
        # [B, 128, 256, 205]
        td_feats = self.topdown(bev_feats)
        # predict individual class log-probabilities
        # [B, class_count, 256, 205]
        return self.classifier(td_feats)

    def aggregate_features(self, x, transforms, adjacency_matrix) -> torch.Tensor:
        """
        aggregate features from all agents in the batch
        """
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, self.ppm, self.cf_h, self.cf_w,
                                                             self.center_x, self.center_y).to(transforms.device)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(self.cf_h, self.cf_w),
                                                 flags=self.aggregation_type)
            # applying the adjacency matrix (difficulty)
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)
        return aggregated_features

    def parameter_count(self):
        """
        returns the number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)