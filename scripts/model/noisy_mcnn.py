import math
import torch
import kornia
from torch import nn
from typing import Tuple
import torch.nn.functional as F
from data.mask_warp import get_single_relative_img_transform
from data.config import SemanticCloudConfig
from model.dual_mcnn import DualTransposedMCNN3x
from model.modules.lie_so3.lie_so3_m import LieSO3

class NoisyMCNNT3x(DualTransposedMCNN3x):
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str):
        super(NoisyMCNNT3x, self).__init__(num_classes, output_size, sem_cfg, aggr_type)
        self.feat_matching_net = LatentFeatureMatcher(128, 128, 64, 32, 5 * 9)
        # semantic aggregation parameters
        self.sem_cf_h, self.sem_cf_w = 80, 108
        self.sem_ppm = self.sem_cfg.pix_per_m(self.sem_cf_h, self.sem_cf_w)
        self.sem_center_x = self.sem_cfg.center_x(self.sem_cf_w)
        self.sem_center_y = self.sem_cfg.center_y(self.sem_cf_h)
        # mask aggregation parameters
        self.msk_cf_h, self.msk_cf_w = 256, 205
        self.msk_ppm = self.sem_cfg.pix_per_m(self.msk_cf_h, self.msk_cf_w)
        self.msk_center_x = self.sem_cfg.center_x(self.msk_cf_w)
        self.msk_center_y = self.sem_cfg.center_y(self.msk_cf_h)
        # model specification
        self.output_count = 4
        self.model_type = 'semantic+mask'
        self.notes = 'using lie-so3 to counter noise'

    def forward(self, rgbs, transforms, adjacency_matrix, car_masks):
        # B, 3, 480, 640: input size
        # B, 64, 80, 108
        shared = self.sseg_mcnn.learning_to_downsample(rgbs)
        # B, 128, 80, 108
        sseg_x = self.sseg_mcnn.global_feature_extractor(shared)
        mask_x = self.mask_feature_extractor(shared)
        # B, 128, 80, 108
        sseg_x = self.sseg_mcnn.feature_fusion(shared, sseg_x)
        mask_x = self.mask_feature_fusion(shared, mask_x)
        # add ego car masks
        sseg_x = sseg_x + F.interpolate(car_masks.unsqueeze(1), size=(self.sem_cf_h, self.sem_cf_w), mode='bilinear', align_corners=True)
        mask_x = mask_x + F.interpolate(car_masks.unsqueeze(1), size=(self.sem_cf_h, self.sem_cf_w), mode='bilinear', align_corners=True)
        # tf noise
        agent_count = transforms.shape[0]
        relative_tf_noise = torch.zeros(size=(agent_count, agent_count, 3, 3),
                                        dtype=torch.float32,
                                        device=transforms.device)
        # B, 128, 80, 108
        # 2 stage message passing for semantics
        aggr_sseg_x = self.aggregate_features(mask_x * sseg_x, transforms, relative_tf_noise, False,
                                              adjacency_matrix, self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y)
        aggr_sseg_x = self.graph_aggr_conv1(aggr_sseg_x)
        aggr_sseg_x = self.aggregate_features(aggr_sseg_x, transforms, relative_tf_noise, True,
                                              adjacency_matrix, self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y)
        aggr_sseg_x = self.graph_aggr_conv2(aggr_sseg_x)
        # solo mask estimation
        # B, 1, 80, 108
        solo_mask_x = torch.sigmoid(self.mask_classifier(mask_x))
        # B, 1, 256, 205
        solo_mask_x = F.interpolate(solo_mask_x, self.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size
        # B, 1, 256, 205
        aggr_mask_x = self.aggregate_features(solo_mask_x.detach(), transforms, relative_tf_noise, True,
                                              adjacency_matrix, self.msk_ppm, self.msk_cf_h, self.msk_cf_w,
                                              self.msk_center_x, self.msk_center_y)
        aggr_mask_x = torch.sigmoid(self.mask_aggr_conv(aggr_mask_x))
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x

    def aggregate_features(self, x, transforms, relative_noise, matched,
                           adjacency_matrix, ppm, cf_h, cf_w, center_x, center_y) -> torch.Tensor:
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, ppm, cf_h, cf_w,
                                                             center_x, center_y).to(transforms.device)
            # if features are already matched in an earlier aggregation step
            if matched:
                relative_tfs = relative_tfs @ relative_noise[i]
                warped_features = kornia.warp_affine(x, relative_tfs, dsize=(cf_h, cf_w),
                                                    flags=self.aggregation_type)
            # otherwise use feature matcher to estimate relative noise
            else:
                warped_features = kornia.warp_affine(x, relative_tfs, dsize=(cf_h, cf_w),
                                                     flags=self.aggregation_type)
                relative_noise[i] = self.feat_matching_net(warped_features[i], warped_features, ppm)
                warped_features = kornia.warp_affine(warped_features, relative_noise[i, :, :2],
                                                     dsize=(cf_h, cf_w),
                                                     flags=self.aggregation_type)
            # applying the adjacency matrix (difficulty)
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)
        return aggregated_features

class LatentFeatureMatcher(nn.Module):
    """
    latent feature matcher takes two feature maps and
    returns their relative transform.
    """
    def __init__(self, c_input = 128, c2 = 128, c_3 = 64, lin_size = 32, lin_dims = 5 * 9):
        super(LatentFeatureMatcher, self).__init__()
        # feature matching network
        self.feature_matcher = nn.Sequential(
            nn.Conv2d(c_input * 2, c2, kernel_size=10, stride=2, groups=c_input, bias=False),
            nn.InstanceNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c_3, kernel_size=10, stride=2, bias=False),
            nn.InstanceNorm2d(c_3),
            nn.ReLU(True),
            nn.Conv2d(c_3, c_3 // 2, kernel_size=5, stride=2, bias=False),
            nn.InstanceNorm2d(c_3 // 2),
            nn.ReLU(True),
        )
        self.linear = nn.Sequential(
            nn.Linear((c_3 // 2) * lin_dims, lin_size),
            nn.ReLU(),
            nn.Linear(lin_size, lin_size),
            nn.ReLU(),
            nn.Linear(lin_size, 5),
        )
        self.lie_so3 = LieSO3()

    def forward(self, feat_x, feat_y, ppm):
        # feat_x: C x 80 x 108
        # feat_y: B x C x 80 x 108
        # interleaved: B x 2C x 80 x 108
        batch_size, channels, feat_h, feat_w = feat_y.shape
        rep_feat_x = feat_x.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x = torch.stack((rep_feat_x, feat_y), dim=2).view(batch_size, channels * 2, feat_h, feat_w)
        x = self.feature_matcher(x)
        x = self.linear(x.view(batch_size, -1))
        rot_matrices = self.lie_so3(x[:, 2:].view(batch_size, 1, 1, 3)).view(batch_size, 3, 3)
        rot_matrices[:, 0, 2] = x[:, 0] * ppm
        rot_matrices[:, 1, 2] = x[:, 1] * ppm
        # zero out other rotations
        rot_matrices[:, 2, :] = 0
        rot_matrices[:, 2, 2] = 1
        return rot_matrices
