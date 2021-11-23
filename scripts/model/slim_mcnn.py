import torch
import kornia
import torch.nn.functional as F

from data.mask_warp import get_single_relative_img_transform
from data.config import SemanticCloudConfig
from model.dual_mcnn import DualTransposedMCNN3x
from model.large_mcnn import _DSConv

class SlimMCNNT3x(DualTransposedMCNN3x):
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str, compression_channels: int):
        super().__init__(num_classes, output_size, sem_cfg, aggr_type)
        self.compression_encoder = _DSConv(128, compression_channels)
        self.compression_decoder = _DSConv(compression_channels, 128)
    
    def forward(self, rgbs, transforms, adjacency_matrix, car_masks, **kwargs):
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
        # simulate compression/decompression
        sseg_x_comp = self.compression_encoder(sseg_x)
        sseg_x_comp = self.compression_decoder(sseg_x_comp)
        # B, 128, 80, 108
        # 2 stage message passing for semantics
        aggr_sseg_x = self.aggregate_compressed_features(mask_x * sseg_x, mask_x * sseg_x_comp, transforms, adjacency_matrix)
        aggr_sseg_x = self.graph_aggr_conv1(aggr_sseg_x)
        aggr_sseg_x = self.aggregate_compressed_features(aggr_sseg_x, aggr_sseg_x, transforms, adjacency_matrix)
        aggr_sseg_x = self.graph_aggr_conv2(aggr_sseg_x)
        # solo mask estimation
        # B, 1, 80, 108
        solo_mask_x = torch.sigmoid(self.mask_classifier(mask_x))
        # B, 1, 256, 205
        solo_mask_x = F.interpolate(solo_mask_x, self.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size
        # B, 1, 256, 205
        aggr_mask_x = self.aggregate_mask_features(solo_mask_x.detach(), transforms, adjacency_matrix)
        aggr_mask_x = torch.sigmoid(self.mask_aggr_conv(aggr_mask_x))
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x


    def aggregate_compressed_features(self, x_org, x_comp, transforms, adjacency_matrix) -> torch.Tensor:
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x_comp)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, self.sem_ppm,
                                                             self.sem_center_x, self.sem_center_y).to(transforms.device)
            warped_features = kornia.warp_affine(x_comp, relative_tfs, dsize=(self.sem_cf_h, self.sem_cf_w),
                                                 mode=self.aggregation_type)
            # replace the compressed ego-features with original ego-features (only in 1st step)
            warped_features[i] = x_org[i]
            # apply the adjacency matrix (difficulty)
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)
        return aggregated_features

    def aggregate_mask_features(self, x, transforms, adjacency_matrix) -> torch.Tensor:
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, self.msk_ppm,
                                                             self.msk_center_x, self.msk_center_y).to(transforms.device)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(self.msk_cf_h, self.msk_cf_w),
                                                 mode=self.aggregation_type)
            # applying the adjacency matrix (difficulty)
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)
        return aggregated_features