import torch
import torch.nn as nn
import numpy as np

from model.pyrocc.fpn import FPN50
from model.pyrocc.pyramid import TransformerPyramid
from model.pyrocc.topdown import TopdownNetwork
from model.pyrocc.classifier import LinearClassifier, BayesianClassifier
from data.config import SemanticCloudConfig

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
                 # original pyrocc args
                 tfm_channels = 64, bayesian_classifer = False,
                 map_extents = [-10.0, 1.0, 10.0, 25.0], ymin = -2, ymax = 4,
                 topdown_channels = 128, topdown_strides = [1, 2], topdown_layers = [4, 4],
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
        # 0.5 = 0.25 * mul([1, 2])
        tfm_resolution = map_resolution * reduce(mul, topdown_strides)
        self.transformer = TransformerPyramid(256, tfm_channels, tfm_resolution,
                                              map_extents, ymin,
                                              ymax, focal_length)

        # Build topdown network
        self.topdown = TopdownNetwork(tfm_channels, topdown_channels,
                                topdown_layers, topdown_strides,
                                topdown_blocktype)
        
        # Build classifier
        if bayesian_classifer:
            self.classifier = BayesianClassifier(self.topdown.out_channels, num_classes)
        else:
            self.classifier = LinearClassifier(self.topdown.out_channels, num_classes)

        if prior:
            self.classifier.initialise(prior)
    

    def forward(self, image, transforms, adjacency_matrix):
        # image: [B, 3, 640, 480]
        # calib: [B, 3, 3]

        # Extract multiscale feature maps
        # 0: [B, 256, 60, 80]
        # 1: [B, 256, 30, 40]
        # 2: [B, 256, 15, 20]
        # 3: [B, 256, 8 , 10]
        # 4: [B, 256, 4 ,  5]
        feature_maps = self.frontend(image)
        # Transform image features to birds-eye-view
        # [B, 64, 124, 103] from:
        #   -- [2, 64, 60, 103]
        #   -- [2, 64, 35, 103]
        #   -- [2, 64, 17, 103]
        #   -- [2, 64, 9 , 103]
        #   -- [2, 64, 3 , 103]
        bev_feats = self.transformer(feature_maps, self.calib)

        # Apply topdown network
        # [B, 256, 248, 206]
        td_feats = self.topdown(bev_feats)

        # Predict individual class log-probabilities
        # [B, class_count, 248, 206]
        logits = self.classifier(td_feats)
        return logits
    
    def parameter_count(self):
        """
        returns the number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)