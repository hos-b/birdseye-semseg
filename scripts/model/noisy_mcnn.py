import torch
import kornia
from torch import nn
from torch.nn import functional as F
from data.mask_warp import get_single_relative_img_transform
from data.utils import get_noisy_transforms
from data.config import SemanticCloudConfig
from model.large_mcnn import TransposedMCNN


class NoisyMCNN(TransposedMCNN):
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str,
                 noise_dx_std, noise_dy_std, noise_th_std):
        super(NoisyMCNN, self).__init__(num_classes, output_size, sem_cfg, aggr_type)
        self.cf_h, self.cf_w = 80, 108
        self.ppm = self.sem_cfg.pix_per_m(self.cf_h, self.cf_w)
        self.center_x = self.sem_cfg.center_x(self.cf_w)
        self.center_y = self.sem_cfg.center_y(self.cf_h)

        self.se3_dx_std = noise_dx_std
        self.se3_dy_std = noise_dy_std
        self.se3_th_std = noise_th_std
        self.matching_net = LatentFeatureMatcher(self.ppm)

    def aggregate_features(self, x, transforms, adjacency_matrix) -> torch.Tensor:
        # add se3 noise to the transforms
        transforms = get_noisy_transforms(transforms, self.se3_dx_std, self.se3_dy_std, self.se3_th_std)
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, self.ppm, self.cf_h, self.cf_w,
                                                             self.center_x, self.center_y).to(transforms.device)
            relative_tfs = relative_tfs @ self.matching_net(x[i], x)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(self.cf_h, self.cf_w),
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
    def __init__(self, pixels_per_meter):
        super(LatentFeatureMatcher, self).__init__()
        self.pixels_per_meter = pixels_per_meter
        # feature matching network
        self.feature_matcher = nn.Sequential(
            nn.Conv2d(128 * 2, 128, kernel_size=10, stride=2, groups=128, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=10, stride=2, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=5, stride=2, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
        )
        self.linear = nn.Sequential(
            nn.Linear(32 * 5 * 9, 256),
            nn.Sigmoid(),
            nn.Linear(256, 32),
            nn.Sigmoid(),
            nn.Linear(32, 4),
        )

    def forward(self, feat_x, feat_y):
        # feat_x: 128 x 80 x 108
        # feat_y: B x 128 x 80 x 108
        # interleaved: B x 256 x 80 x 108
        batch_size = feat_y.shape[0]
        rep_feat_x = feat_x.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x = torch.stack((rep_feat_x, feat_y), dim=2).view(batch_size, 256, 80, 108)
        x = self.feature_matcher(x)
        x = self.linear(x.view(batch_size, -1))
        angle = torch.atan2(x[:, 2], x[:, 3])
        rot_matrices = torch.zeros((batch_size, 3, 3),
                                    dtype=feat_x.dtype,
                                    device=feat_x.device)
        rot_matrices[:, 0, 0] = torch.cos(angle)
        rot_matrices[:, 0, 1] = -torch.sin(angle)
        rot_matrices[:, 1, 0] = torch.sin(angle)
        rot_matrices[:, 1, 1] = torch.cos(angle)
        rot_matrices[:, 0, 2] = x[:, 0] * self.pixels_per_meter
        rot_matrices[:, 1, 2] = x[:, 1] * self.pixels_per_meter
        rot_matrices[:, 2, 2] = 1
        return rot_matrices
