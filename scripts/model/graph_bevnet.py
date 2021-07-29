import math
import torch
import kornia
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from data.mask_warp import get_single_relative_img_transform
from data.config import SemanticCloudConfig
from model.large_mcnn import LearningToDownsampleWide, GlobalFeatureExtractor
from model.large_mcnn import FeatureFusionModule, TransposedClassifier
from model.base import DoubleSemantic

class GraphBEVNet(DoubleSemantic):
    """
    FastSCNN slightly modified to output full masks
    with aggregation from other agents.
    """
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type):
        super().__init__()
        # aggregation parameters
        self.sem_cfg = sem_cfg
        self.output_size = output_size
        self.aggregation_type = aggr_type
        self.cf_h, self.cf_w = 80, 108
        self.ppm = self.sem_cfg.pix_per_m(self.cf_h, self.cf_w)
        self.center_x = self.sem_cfg.center_x(self.cf_w)
        self.center_y = self.sem_cfg.center_y(self.cf_h)
        # modules
        self.learning_to_downsample = LearningToDownsampleWide(dw_channels1=32,
                                                               dw_channels2=48,
                                                               out_channels=64)
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels=64,
                                                               block_channels=(64, 128, 256),
                                                               t=8,
                                                               num_blocks=(4, 4, 4),
                                                               pool_sizes=(4, 6, 8, 10))
        self.feature_fusion = FeatureFusionModule(highres_in_channels=64,
                                                 lowres_in_channels=256,
                                                 out_channels=256,
                                                 scale_factor=4)
        self.classifier = TransposedClassifier(256, num_classes)
        # calibration parameters
        rgb_w, rgb_h, fov = 640, 480, 60.0
        focal_length = rgb_w / (2 * np.tan(fov * np.pi / 360))
        img_offset = rgb_h // 2
        # scale calibration parameters to the feature size
        focal_length *= (self.cf_w / rgb_w)
        img_offset *= (self.cf_h / rgb_h)
        self.bev_feature_extractor = BEVConvolution(in_channels=64,
                                                    out_channels=64,
                                                    out_depth=self.cf_h,
                                                    kernel_size=3,
                                                    zmax=20.0,
                                                    ymin=-2,
                                                    ymax=4,
                                                    focal_length=focal_length,
                                                    cy=img_offset)
        # model specification
        self.output_count = 2
        self.model_type = 'semantic-only'
        self.notes = 'very large & slow'

    def forward(self, x, transforms, adjacency_matrix, car_masks):
        # B, 3, 480, 640: input size
        # B, 64, 80, 108
        shared = self.learning_to_downsample(x)
        shared = self.bev_feature_extractor(shared)
        # B, 256, 15, 20
        x = self.global_feature_extractor(shared)
        # B, 256, 80, 108
        x = self.feature_fusion(shared, x)
        # add ego car masks
        x = x + F.interpolate(car_masks.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear', align_corners=True)
        # B, 256, 80, 108
        aggr_x = self.aggregate_features(x, transforms, adjacency_matrix)
        # B, 7, 256, 205
        solo_x = F.interpolate(self.classifier(x), self.output_size, mode='bilinear', align_corners=True)
        aggr_x = F.interpolate(self.classifier(aggr_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_x, aggr_x

    def aggregate_features(self, x, transforms, adjacency_matrix):
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(
                transforms, i, self.ppm,
                self.cf_h, self.cf_w,
                self.center_x, self.center_y
            ).to(transforms.device)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(self.cf_h, self.cf_w),
                                                 flags=self.aggregation_type)
            warped_features[outside_fov] = 0
            aggregated_features[i, ...] = warped_features.sum(dim=0)
        return aggregated_features

    def parameter_count(self):
        """
        returns the number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BEVConvolution(nn.Module):
    """
    Performs 1D convolution over the columns of extracted global
    features to convert them to bird's eye view features.
    """
    def __init__(self, in_channels, out_channels, out_depth, kernel_size,
                 zmax, ymin, ymax, focal_length, cy):
        super().__init__()
        # cropped height
        self.target_height = math.ceil(focal_length * (ymax - ymin) / zmax)
        self.fc = nn.Conv1d(in_channels * self.target_height, out_channels * out_depth,
                            kernel_size=kernel_size, stride=1, padding=1)
        self.reul = nn.ReLU(True)
        self.iconv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.out_channels = out_channels
        # more cropping calculations
        self.y_mid = (ymin + ymax) / 2
        self.z_max = zmax
        self.focal_length = focal_length
        self.c_y = cy
        vmid = self.y_mid * self.focal_length / self.z_max + self.c_y
        self.vmin = math.floor(vmid - self.target_height / 2)
        self.vmax = math.floor(vmid + self.target_height / 2)
    
    def _crop_feature_map(self, fmap):
        """
        crops the feature map to only include the desired y range
        """
        # pad or crop input tensor to match dimensions
        return F.pad(fmap, [0, 0, -self.vmin, self.vmax - fmap.shape[-2]])

    def forward(self, features):
        """
        formward pass similar to pyrocc
        """
        features = self._crop_feature_map(features)
        B, _, _, W = features.shape
        flat_feats = features.flatten(1, 2)
        new_feats = self.fc(flat_feats).view(B, self.out_channels, -1, W)
        return self.iconv(self.reul(new_feats))
