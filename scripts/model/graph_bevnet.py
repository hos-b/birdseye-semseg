import torch
import kornia
import torch.nn as nn
from torch.nn import functional as F
from data.mask_warp import get_single_relative_img_transform
from data.config import SemanticCloudConfig
from model.large_mcnn import LearningToDownsampleWide, GlobalFeatureExtractor, FeatureFusionModule, TransposedClassifier

class GraphBEVNet(torch.nn.Module):
    """
    FastSCNN slightly modified to output full masks
    with aggregation from other agents.
    """
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type):
        super().__init__()
        self.learning_to_downsample = LearningToDownsampleWide(dw_channels1=64,
                                                               dw_channels2=96,
                                                               out_channels=128)
        self.bev_feature_extractor = BEVConvolution(in_channels=128,
                                                    out_channels=256,
                                                    kernel_size=3)
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels=128,
                                                               block_channels=(128, 192, 256),
                                                               t=8,
                                                               num_blocks=(5, 6, 6),
                                                               pool_sizes=(4, 6, 8, 10))
        self.feature_fusion = FeatureFusionModule(highres_in_channels=128,
                                                 lowres_in_channels=256,
                                                 out_channels=256,
                                                 scale_factor=4)
        self.classifier = TransposedClassifier(256, num_classes)
        self.output_size = output_size
        self.sem_cfg = sem_cfg
        # set aggregation parameters
        self.aggregation_type = aggr_type
        self.cf_h, self.cf_w = 80, 108
        self.ppm = self.sem_cfg.pix_per_m(self.cf_h, self.cf_w)
        self.center_x = self.sem_cfg.center_x(self.cf_w)
        self.center_y = self.sem_cfg.center_y(self.cf_h)

    def forward(self, x, transforms, adjacency_matrix, car_masks):
        # B, 3, 480, 640: input size
        # B, 64, 80, 108
        shared = self.learning_to_downsample(x)
        # B, 128, 15, 20
        x = self.global_feature_extractor(shared)
        # B, 128, 80, 108
        x = self.feature_fusion(shared, x)
        # add ego car masks
        x = x + F.interpolate(car_masks.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear')
        # B, 128, 80, 108
        aggr_x = self.aggregate_features(x, transforms, adjacency_matrix)
        aggr_x = torch.sigmoid(self.mask_prediction(aggr_x))
        # B, 1, 80, 108
        solo_x = torch.sigmoid(self.mask_prediction(x))
        # B, 1, 480, 640
        solo_x = F.interpolate(solo_x, self.output_size, mode='bilinear', align_corners=True)
        aggr_x = F.interpolate(aggr_x, self.output_size, mode='bilinear', align_corners=True)
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
    def __init__(self, in_channels, in_height, out_channels, kernel_size):
        super().__init__()
        self.fc = nn.Conv1d(in_channels * in_height, out_channels, kernel_size=kernel_size, stride=1)
        self.reul = nn.ReLU(True)
        self.iconv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.out_channels = out_channels

    def forward(self, features):
        """
        formward pass similar to pyrocc
        """
        B, _, _, W = features.shape
        flat_feats = features.flatten(1, 2)
        new_feats = self.fc(flat_feats).view(B, self.out_channels, -1, W)
        return self.iconv(self.reul(new_feats))