import math
import torch
import kornia
from torch import nn
import torch.nn.functional as F
from data.mask_warp import get_single_relative_img_transform, get_modified_single_relative_img_transform
from data.config import SemanticCloudConfig
from model.dual_mcnn import DualTransposedMCNN3x
from model.modules.lie_so3.lie_so3_m import LieSE3
from model.base import SoloAggrSemanticsMask

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
        gt_relative_noise = kwargs.get('gt_relative_noise', None)
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
                                              self.sem_center_x, self.sem_center_y,
                                              gt_relative_noise, noise_correction_en)
        aggr_sseg_x = self.graph_aggr_conv1(aggr_sseg_x)
        aggr_sseg_x = self.aggregate_features(aggr_sseg_x, transforms, adjacency_matrix,
                                              self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y,
                                              gt_relative_noise, noise_correction_en)
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
                                              self.msk_center_x, self.msk_center_y,
                                              gt_relative_noise, noise_correction_en)
        aggr_mask_x = torch.sigmoid(self.mask_aggr_conv(aggr_mask_x))
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x

    def aggregate_features(self, x, transforms, adjacency_matrix, ppm, cf_h, cf_w,
                           center_x, center_y, gt_relative_noise, noise_correction_en) -> torch.Tensor:
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == False)[0]
            if noise_correction_en:
                # if so2 noise is not estimated
                if not self.feat_matching_net.estimated:
                    noisy_img_tfs = get_modified_single_relative_img_transform(transforms, gt_relative_noise[i], i, ppm, center_x, center_y).to(transforms.device)
                    noisy_warped_features = kornia.warp_affine(x, noisy_img_tfs, dsize=(cf_h, cf_w), mode=self.aggregation_type)
                    self.feat_matching_net(noisy_warped_features[i], noisy_warped_features, i)
                # rectfiy the transform using estimated noise and warp (from scratch)
                fixed_noise = gt_relative_noise[i] @ self.feat_matching_net.estimated_noise[i].inverse()
                denoised_img_tfs = get_modified_single_relative_img_transform(transforms, fixed_noise, i, ppm, center_x, center_y).to(transforms.device)
                warped_features = kornia.warp_affine(x, denoised_img_tfs, dsize=(cf_h, cf_w), mode=self.aggregation_type)
            else:
                noisy_img_tfs = get_modified_single_relative_img_transform(transforms, gt_relative_noise[i], i, ppm, center_x, center_y).to(transforms.device)
                warped_features = kornia.warp_affine(x, noisy_img_tfs, dsize=(cf_h, cf_w), mode=self.aggregation_type)
            # applying the adjacency matrix
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)

        # set estimated flag to true
        self.feat_matching_net.estimated = True
        return aggregated_features

class NoisyMCNNT3xRT(SoloAggrSemanticsMask):
    """
    same as NoisyMCNNT3x but resumed from a checkpoint and detached, apart from the
    noise canceling network.
    """
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str, mcnnt3x_path: str):
        super().__init__()
        self.feat_matching_net = LatentFeatureMatcher(128, 80, 108)
        self.mcnnt3x = DualTransposedMCNN3x(num_classes, output_size, sem_cfg, aggr_type)
        # if training, load the checkpoint
        try:
            self.mcnnt3x.load_state_dict(torch.load(mcnnt3x_path))
        except:
            print("failed to load mcnnt3x submodel."); exit(-1)
        for p in self.mcnnt3x.parameters():
            p.requires_grad = False
        print('disabled gradient calculation for mcnnt3x.')
        # semantic aggregation parameters
        self.sem_cf_h, self.sem_cf_w = 80, 108
        self.sem_ppm = sem_cfg.pix_per_m(self.sem_cf_h, self.sem_cf_w)
        self.sem_center_x = sem_cfg.center_x(self.sem_cf_w)
        self.sem_center_y = sem_cfg.center_y(self.sem_cf_h)
        # mask aggregation parameters
        self.msk_cf_h, self.msk_cf_w = 256, 205
        self.msk_ppm = sem_cfg.pix_per_m(self.msk_cf_h, self.msk_cf_w)
        self.msk_center_x = sem_cfg.center_x(self.msk_cf_w)
        self.msk_center_y = sem_cfg.center_y(self.msk_cf_h)
        # model specification
        self.output_count = 4
        self.model_type = 'semantic+mask'
        self.notes = 'using latent feature matching to counter noise'

    def forward(self, rgbs, transforms, adjacency_matrix, car_masks, **kwargs):
        noise_correction_en = kwargs.get('noise_correction_en', True)
        gt_relative_noise = kwargs.get('gt_relative_noise', None)
        # B, 3, 480, 640: input size
        # B, 64, 80, 108
        shared = self.mcnnt3x.sseg_mcnn.learning_to_downsample(rgbs)
        # B, 128, 80, 108
        sseg_x = self.mcnnt3x.sseg_mcnn.global_feature_extractor(shared)
        mask_x = self.mcnnt3x.mask_feature_extractor(shared)
        # B, 128, 80, 108
        sseg_x = self.mcnnt3x.sseg_mcnn.feature_fusion(shared, sseg_x)
        mask_x = self.mcnnt3x.mask_feature_fusion(shared, mask_x)
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
                                              self.sem_center_x, self.sem_center_y,
                                              gt_relative_noise, noise_correction_en)
        aggr_sseg_x = self.mcnnt3x.graph_aggr_conv1(aggr_sseg_x)
        aggr_sseg_x = self.aggregate_features(aggr_sseg_x, transforms, adjacency_matrix,
                                              self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y,
                                              gt_relative_noise, noise_correction_en)
        aggr_sseg_x = self.mcnnt3x.graph_aggr_conv2(aggr_sseg_x)
        # solo mask estimation
        # B, 1, 80, 108
        solo_mask_x = torch.sigmoid(self.mcnnt3x.mask_classifier(mask_x))
        # B, 1, 256, 205
        solo_mask_x = F.interpolate(solo_mask_x, self.mcnnt3x.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size
        # B, 1, 256, 205
        aggr_mask_x = self.aggregate_features(solo_mask_x, transforms, adjacency_matrix,
                                              self.msk_ppm, self.msk_cf_h, self.msk_cf_w,
                                              self.msk_center_x, self.msk_center_y,
                                              gt_relative_noise, noise_correction_en)
        aggr_mask_x = torch.sigmoid(self.mcnnt3x.mask_aggr_conv(aggr_mask_x))
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.mcnnt3x.sseg_mcnn.classifier(     sseg_x), self.mcnnt3x.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.mcnnt3x.sseg_mcnn.classifier(aggr_sseg_x), self.mcnnt3x.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x

    def aggregate_features(self, x, transforms, adjacency_matrix, ppm, cf_h, cf_w,
                           center_x, center_y, gt_relative_noise, noise_correction_en) -> torch.Tensor:
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == False)[0]
            if noise_correction_en:
                # if so2 noise is not estimated
                if not self.feat_matching_net.estimated:
                    noisy_img_tfs = get_modified_single_relative_img_transform(transforms, gt_relative_noise[i], i, ppm, center_x, center_y).to(transforms.device)
                    noisy_warped_features = kornia.warp_affine(x, noisy_img_tfs, dsize=(cf_h, cf_w), mode=self.mcnnt3x.aggregation_type)
                    self.feat_matching_net(noisy_warped_features[i], noisy_warped_features, i)
                # rectfiy the transform using estimated noise and warp (from scratch)
                fixed_noise = gt_relative_noise[i] @ self.feat_matching_net.estimated_noise[i].inverse()
                denoised_img_tfs = get_modified_single_relative_img_transform(transforms, fixed_noise, i, ppm, center_x, center_y).to(transforms.device)
                warped_features = kornia.warp_affine(x, denoised_img_tfs, dsize=(cf_h, cf_w), mode=self.mcnnt3x.aggregation_type)
            else:
                noisy_img_tfs = get_modified_single_relative_img_transform(transforms, gt_relative_noise[i], i, ppm, center_x, center_y).to(transforms.device)
                warped_features = kornia.warp_affine(x, noisy_img_tfs, dsize=(cf_h, cf_w), mode=self.mcnnt3x.aggregation_type)
            # applying the adjacency matrix
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)

        # set estimated flag to true
        self.feat_matching_net.estimated = True
        return aggregated_features
    
    @torch.no_grad()
    def infer_preaggr(self, rgb, car_mask, **kwargs):
        """
        only used for inference profiling. output may be invalid.
        """
        # 1, 3, 480, 640: input size
        # 1, 64, 80, 108
        shared = self.mcnnt3x.sseg_mcnn.learning_to_downsample(rgb)
        # 1, 128, 80, 108
        sseg_x = self.mcnnt3x.sseg_mcnn.global_feature_extractor(shared)
        mask_x = self.mcnnt3x.mask_feature_extractor(shared)
        # 1, 128, 80, 108
        sseg_x = self.mcnnt3x.sseg_mcnn.feature_fusion(shared, sseg_x)
        mask_x = self.mcnnt3x.mask_feature_fusion(shared, mask_x)
        # add ego car mask
        sseg_x = sseg_x + F.interpolate(car_mask.unsqueeze(1), size=(self.sem_cf_h, self.sem_cf_w), mode='bilinear', align_corners=True)
        mask_x = mask_x + F.interpolate(car_mask.unsqueeze(1), size=(self.sem_cf_h, self.sem_cf_w), mode='bilinear', align_corners=True)
        # 1, 128, 80, 108
        ego_mask = F.interpolate(torch.sigmoid(self.mcnnt3x.mask_classifier(mask_x)), self.mcnnt3x.output_size, mode='bilinear', align_corners=True)
        # return latenlty masked features and ego mask
        return sseg_x * mask_x, ego_mask

    @torch.no_grad()
    def infer_aggregate(self, agent_index, latent_features, ego_masks, transforms, **kwargs):
        """
        only used for inference profiling. output may be invalid.
        """
        agent_count = latent_features.shape[0]
        # B, 128, 80, 108
        # first message passing stage (done for all agents)
        aggr_features = torch.zeros_like(latent_features)
        for i in range(agent_count):
            noisy_relative_tfs = get_single_relative_img_transform(transforms, i, self.sem_ppm, self.sem_center_x, self.sem_center_y).to(transforms.device)
            noisy_warped_features = kornia.warp_affine(latent_features, noisy_relative_tfs, dsize=(self.sem_cf_h, self.sem_cf_w), mode=self.mcnnt3x.aggregation_type)
            # estimate the noise
            self.feat_matching_net(noisy_warped_features[i], noisy_warped_features, i)
            # rewarp
            inverted_noise = self.feat_matching_net.estimated_noise[i].inverse()
            denoised_img_tfs = get_modified_single_relative_img_transform(transforms, inverted_noise, i, self.sem_ppm, self.sem_center_x, self.sem_center_y).to(transforms.device)
            denoised_warped_features = kornia.warp_affine(latent_features, denoised_img_tfs, dsize=(self.sem_cf_h, self.sem_cf_w), mode=self.mcnnt3x.aggregation_type)
            aggr_features[i] = denoised_warped_features.sum(dim=0)
        aggr_features = self.mcnnt3x.graph_aggr_conv1(aggr_features)
        # second message passing stage (only done for ego agent)
        denoised_img_tfs = get_modified_single_relative_img_transform(transforms, inverted_noise, agent_index, self.sem_ppm, self.sem_center_x, self.sem_center_y).to(transforms.device)
        warped_features = kornia.warp_affine(aggr_features, denoised_img_tfs, dsize=(self.sem_cf_h, self.sem_cf_w), mode=self.mcnnt3x.aggregation_type)
        final_features = self.mcnnt3x.graph_aggr_conv2(warped_features.sum(dim=0).unsqueeze(0))
        final_semantics = F.interpolate(self.mcnnt3x.sseg_mcnn.classifier(final_features), self.mcnnt3x.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size (only done for ego agent)
        denoised_img_tfs = get_modified_single_relative_img_transform(transforms, inverted_noise, agent_index, self.msk_ppm, self.msk_center_x, self.msk_center_y).to(transforms.device)
        warped_masks = kornia.warp_affine(ego_masks, denoised_img_tfs, dsize=(self.msk_cf_h, self.msk_cf_w), mode=self.mcnnt3x.aggregation_type)
        final_mask = torch.sigmoid(self.mcnnt3x.mask_aggr_conv(warped_masks.sum(dim=0).unsqueeze(0)))
        return final_semantics, final_mask


class LatentFeatureMatcher(nn.Module):
    """
    latent feature matcher takes two feature maps and
    returns their relative transform.
    """
    def __init__(self, input_ch, input_h, input_w):
        super(LatentFeatureMatcher, self).__init__()
        # feature matching network
        self.feature_matcher = nn.Sequential(
            nn.Conv2d(input_ch * 2, input_ch, kernel_size=10, stride=2, groups=2, bias=False),
            nn.Conv2d(input_ch, input_ch // 2, kernel_size=5, stride=2, bias=False),
            nn.Conv2d(input_ch // 2, input_ch // 4, kernel_size=5, stride=1, bias=False),
            nn.Conv2d(input_ch // 4, input_ch // 8, kernel_size=3, stride=2, bias=False),
            nn.Conv2d(input_ch // 8, input_ch // 16, kernel_size=3, stride=1, bias=False),
            nn.PReLU(),
        )
        output_h, output_w = calculate_conv2d_sequence_output_size(input_h, input_w, self.feature_matcher)
        self.linear = nn.Sequential(
            nn.Linear((input_ch // 16) * output_h * output_w, 64),
            nn.PReLU(),
            nn.Linear(64, 8),
            nn.PReLU(),
            nn.Linear(8, 3)
        )
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
        """
        # channel wise interleaving can be checked with the following code
            for i in range(agent_count):
                for c in range(128):
                    if (x[i, c * 2] == feat_x[c]).unique() != torch.tensor([True], device=x.device):
                        import pdb; pdb.set_trace()
            for i in range(agent_count):
                for c in range(128):
                    if (x[i, c * 2  + 1] == feat_y[i, c]).unique() != torch.tensor([True], device=x.device):
                        import pdb; pdb.set_trace()
        """
        # feat_x: C x 80 x 108
        # feat_y: A x C x 80 x 108
        agent_count, channels, feat_h, feat_w = feat_y.shape
        # rep_feat_x: A x C x 80 x 108
        rep_feat_x = feat_x.unsqueeze(0).repeat(agent_count, 1, 1, 1)
        # interleaved: A x 2C x 80 x 108
        x = torch.stack((rep_feat_x, feat_y), dim=2).view(agent_count, channels * 2, feat_h, feat_w)
        x = self.linear(self.feature_matcher(x).flatten(1, -1)).squeeze()
        tf = torch.eye(4, dtype=torch.float32, device=feat_x.device).unsqueeze(0).repeat(agent_count, 1, 1)
        angles = torch.tanh(x[:, 0]) * math.pi
        tf[:, 0, 0] =  torch.cos(angles)
        tf[:, 0, 1] = -torch.sin(angles)
        tf[:, 1, 0] =  torch.sin(angles)
        tf[:, 1, 1] =  torch.cos(angles)
        tf[:, 0, 3] = x[:, 1]
        tf[:, 1, 3] = x[:, 2]
        self.estimated_noise[agent_index] = tf


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
