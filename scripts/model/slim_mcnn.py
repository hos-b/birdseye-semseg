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
        msg_x_comp = self.compression_encoder(mask_x * sseg_x)
        msg_x_decomp = self.compression_decoder(msg_x_comp)
        # B, 128, 80, 108
        # 2 stage message passing for semantics
        aggr_sseg_x = self.aggregate_compressed_features(mask_x * sseg_x, msg_x_decomp, transforms, adjacency_matrix)
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
    
    @torch.no_grad()
    def infer_preaggr(self, rgb, car_mask, **kwargs):
        """
        only used for inference profiling. output may be invalid.
        """
        # 1, 3, 480, 640: input size
        # 1, 64, 80, 108
        shared = self.sseg_mcnn.learning_to_downsample(rgb)
        # 1, 128, 80, 108
        sseg_x = self.sseg_mcnn.global_feature_extractor(shared)
        mask_x = self.mask_feature_extractor(shared)
        # 1, 128, 80, 108
        sseg_x = self.sseg_mcnn.feature_fusion(shared, sseg_x)
        mask_x = self.mask_feature_fusion(shared, mask_x)
        # add ego car mask
        sseg_x = sseg_x + F.interpolate(car_mask.unsqueeze(1), size=(self.sem_cf_h, self.sem_cf_w), mode='bilinear', align_corners=True)
        mask_x = mask_x + F.interpolate(car_mask.unsqueeze(1), size=(self.sem_cf_h, self.sem_cf_w), mode='bilinear', align_corners=True)
        # 1, 128, 80, 108
        ego_mask = F.interpolate(torch.sigmoid(self.mask_classifier(mask_x)), self.output_size, mode='bilinear', align_corners=True)
        # return compressed masked features and ego mask
        msg_x_comp = self.compression_encoder(mask_x * sseg_x)
        return msg_x_comp, ego_mask

    @torch.no_grad()
    def infer_aggregate(self, agent_index, latent_features, ego_masks, transforms, **kwargs):
        """
        only used for inference profiling. output may be invalid.
        """
        # decompress messages
        latent_features = self.compression_decoder(latent_features)
        agent_count = latent_features.shape[0]
        # B, 128, 80, 108
        # first message passing stage (done for all agents)
        aggr_features = torch.zeros_like(latent_features)
        for i in range(agent_count):
            relative_tfs = get_single_relative_img_transform(transforms, i, self.sem_ppm, self.sem_center_x, self.sem_center_y).to(transforms.device)
            warped_features = kornia.warp_affine(latent_features, relative_tfs, dsize=(self.sem_cf_h, self.sem_cf_w), mode=self.aggregation_type)
            aggr_features[i] = warped_features.sum(dim=0)
        aggr_features = self.graph_aggr_conv1(aggr_features)
        # second message passing stage (only done for ego agent)
        relative_tfs = get_single_relative_img_transform(transforms, agent_index, self.sem_ppm, self.sem_center_x, self.sem_center_y).to(transforms.device)
        warped_features = kornia.warp_affine(aggr_features, relative_tfs, dsize=(self.sem_cf_h, self.sem_cf_w), mode=self.aggregation_type)
        final_features = self.graph_aggr_conv2(warped_features.sum(dim=0).unsqueeze(0))
        final_semantics = F.interpolate(self.sseg_mcnn.classifier(final_features), self.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size (only done for ego agent)
        relative_tfs = get_single_relative_img_transform(transforms, agent_index, self.msk_ppm, self.msk_center_x, self.msk_center_y).to(transforms.device)
        warped_masks = kornia.warp_affine(ego_masks, relative_tfs, dsize=(self.msk_cf_h, self.msk_cf_w), mode=self.aggregation_type)
        final_mask = torch.sigmoid(self.mask_aggr_conv(warped_masks.sum(dim=0).unsqueeze(0)))
        return final_semantics, final_mask