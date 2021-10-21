import math
import torch
import kornia
from torch import nn
import torch.nn.functional as F
from data.mask_warp import get_single_relative_img_transform, get_rectified_single_relative_img_transform
from data.config import SemanticCloudConfig
from model.dual_mcnn import DualTransposedMCNN3x
from model.modules.lie_so3.lie_so3_m import LieSE3

class NoisyMCNNT3x(DualTransposedMCNN3x):
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str):
        super(NoisyMCNNT3x, self).__init__(num_classes, output_size, sem_cfg, aggr_type)
        self.feat_matching_net = LatentFeatureMatcher(128, 80, 108, 0.01)
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

    def forward(self, rgbs, transforms, adjacency_matrix, car_masks, **kwargs):
        noise_correction_en = kwargs.get('noise_correction_en', True)
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
        if noise_correction_en:
            self.feat_matching_net.refresh(transforms.shape[0], transforms.device)
        # B, 128, 80, 108
        # 2 stage message passing for semantics
        aggr_sseg_x = self.aggregate_features(mask_x * sseg_x, transforms, adjacency_matrix,
                                              self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y, noise_correction_en)
        aggr_sseg_x = self.graph_aggr_conv1(aggr_sseg_x)
        aggr_sseg_x = self.aggregate_features(aggr_sseg_x, transforms, adjacency_matrix,
                                              self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y, noise_correction_en)
        aggr_sseg_x = self.graph_aggr_conv2(aggr_sseg_x)
        # solo mask estimation
        # B, 1, 80, 108
        solo_mask_x = torch.sigmoid(self.mask_classifier(mask_x))
        # B, 1, 256, 205
        solo_mask_x = F.interpolate(solo_mask_x, self.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size
        # B, 1, 256, 205
        aggr_mask_x = self.aggregate_features(solo_mask_x.detach(), transforms, adjacency_matrix,
                                              self.msk_ppm, self.msk_cf_h, self.msk_cf_w,
                                              self.msk_center_x, self.msk_center_y, noise_correction_en)
        aggr_mask_x = torch.sigmoid(self.mask_aggr_conv(aggr_mask_x))
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x

    def aggregate_features(self, x, transforms, adjacency_matrix, ppm,
                           cf_h, cf_w, center_x, center_y, noise_correction_en) -> torch.Tensor:
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == False)[0]
            if noise_correction_en:
                # if so2 noise is not estimated
                if not self.feat_matching_net.estimated:
                    noisy_relative_img_tfs = get_single_relative_img_transform(transforms, i, ppm, center_x, center_y).to(transforms.device)
                    noisy_warped_features = kornia.warp_affine(x, noisy_relative_img_tfs, dsize=(cf_h, cf_w), mode=self.aggregation_type)
                    self.feat_matching_net(noisy_warped_features[i], noisy_warped_features, i)
                # rectfiy the transform using estimated noise and warp (from scratch)
                relative_img_tfs = get_rectified_single_relative_img_transform(transforms, self.feat_matching_net.estimated_noise[i],
                                                                               i, ppm, center_x, center_y).to(transforms.device)
                warped_features = kornia.warp_affine(x, relative_img_tfs, dsize=(cf_h, cf_w), mode=self.aggregation_type)
            else:
                noisy_relative_img_tfs = get_single_relative_img_transform(transforms, i, ppm, center_x, center_y).to(transforms.device)
                warped_features = kornia.warp_affine(x, noisy_relative_img_tfs, dsize=(cf_h, cf_w), mode=self.aggregation_type)
            # applying the adjacency matrix
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)

        # set estimated flag to true
        self.feat_matching_net.estimated = True
        return aggregated_features

class LatentFeatureMatcher(nn.Module):
    """
    latent feature matcher takes two feature maps and
    returns their relative transform.
    """
    def __init__(self, input_ch, input_h, input_w, rotation_scale):
        super(LatentFeatureMatcher, self).__init__()
        self.rotation_scale = rotation_scale
        # feature matching network
        self.feature_matcher = nn.Sequential(
            nn.Conv2d(input_ch * 2, input_ch, kernel_size=10, stride=2, groups=input_ch, bias=False),
            # nn.InstanceNorm2d(input_ch),
            nn.ReLU(True),
            nn.Conv2d(input_ch, input_ch, kernel_size=10, stride=2, bias=False),
            # nn.InstanceNorm2d(input_ch),
            nn.ReLU(True),
            nn.Conv2d(input_ch, input_ch // 2, kernel_size=5, stride=2, bias=False),
            # nn.InstanceNorm2d(input_ch // 2),
            nn.ReLU(True),
        )
        output_h, output_w = calculate_conv2d_sequence_output_size(input_h, input_w, self.feature_matcher)
        self.linear = nn.Sequential(
            nn.Linear(output_h * output_w, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.lie_so3 = LieSE3()
        self.estimated_noise = None
        self.estimated = False
    
    def refresh(self, agent_count, device):
        self.estimated = False
        if self.estimated_noise is not None:
            del self.estimated_noise
        self.estimated_noise = torch.zeros(size=(agent_count, agent_count, 4, 4),
                                           dtype=torch.float32,
                                           device=device)

    def forward(self, feat_x, feat_y, agent_index):
        # feat_x: C x 80 x 108
        # feat_y: A x C x 80 x 108
        agent_count, channels, feat_h, feat_w = feat_y.shape
        # rep_feat_x: A x C x 80 x 108
        rep_feat_x = feat_x.unsqueeze(0).repeat(agent_count, 1, 1, 1)
        # interleaved: A x 2C x 80 x 108
        x = torch.stack((rep_feat_x, feat_y), dim=2).view(agent_count, channels * 2, feat_h, feat_w)
        x = self.feature_matcher(x)
        # flatten features and pass through linear layer
        x = self.linear(torch.mean(x, dim=1).view(agent_count, -1))
        # get lie_so3 transform
        self.estimated_noise[agent_index] = self.lie_so3(x)
        # return self.estimated_noise[agent_index]

def calculate_conv2d_output_size(fsize_h, fsize_w, kernel_size_h, kernel_size_w,
                                 padding_h, padding_w, stride_h, stride_w):
    output_size_h = math.floor((fsize_h + 2 * padding_h - kernel_size_h) / stride_h) + 1
    output_size_w = math.floor((fsize_w + 2 * padding_w - kernel_size_w) / stride_w) + 1
    return output_size_h, output_size_w

def calculate_conv2d_sequence_output_size(fsize_h, fsize_w, nn_sequential):
    """
    calculate the output size of a sequence of conv2d layers. it is assumed
    that the sequence does not contain any other resizing layers.
    """
    for layer in nn_sequential:
        if isinstance(layer, nn.Conv2d):
            fsize_h, fsize_w = calculate_conv2d_output_size(fsize_h, fsize_w,
                                                            layer.kernel_size[0],
                                                            layer.kernel_size[1],
                                                            layer.padding[0],
                                                            layer.padding[1],
                                                            layer.stride[0],
                                                            layer.stride[1])
    return fsize_h, fsize_w
