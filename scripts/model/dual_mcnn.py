import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from data.config import SemanticCloudConfig
from data.mask_warp import get_single_relative_img_transform
from model.large_mcnn import TransposedMCNN
from model.base import SoloAggrSemanticsMask

class DualTransposedMCNN4x(SoloAggrSemanticsMask):
    """
    two MCNNTs, one for mask, other for semantics. outputs solo & aggr versions of both.
    """
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str):
        super().__init__()
        self.output_size = output_size
        self.sem_cfg = sem_cfg
        self.aggregation_type = aggr_type
        self.mask_mcnn = TransposedMCNN(1          , output_size, sem_cfg, aggr_type)
        self.sseg_mcnn = TransposedMCNN(num_classes, output_size, sem_cfg, aggr_type)
        # aggregation parameters
        self.cf_h, self.cf_w = 80, 108
        self.ppm = self.sem_cfg.pix_per_m(self.cf_h, self.cf_w)
        self.center_x = self.sem_cfg.center_x(self.cf_w)
        self.center_y = self.sem_cfg.center_y(self.cf_h)
        # model specification
        self.output_count = 4
        self.model_type = 'semantic+mask'
        self.notes = 'not small, probably not fast but all in one'
    
    def forward(self, rgbs, transforms, adjacency_matrix, car_masks):
        # B, 3, 480, 640: input size
        # B, 64, 80, 108
        sseg_shared = self.sseg_mcnn.learning_to_downsample(rgbs)
        mask_shared = self.mask_mcnn.learning_to_downsample(rgbs)
        # B, 128, 15, 20
        sseg_x = self.sseg_mcnn.global_feature_extractor(sseg_shared)
        mask_x = self.mask_mcnn.global_feature_extractor(mask_shared)
        # B, 128, 80, 108
        sseg_x = self.sseg_mcnn.feature_fusion(sseg_shared, sseg_x)
        mask_x = self.mask_mcnn.feature_fusion(mask_shared, mask_x)
        # add ego car masks
        sseg_x = sseg_x + F.interpolate(car_masks.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear', align_corners=True)
        mask_x = mask_x + F.interpolate(car_masks.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear', align_corners=True)
        # B, 128, 80, 108
        aggr_sseg_x = self.aggregate_features(mask_x * sseg_x, transforms, adjacency_matrix)
        aggr_mask_x = self.aggregate_features(mask_x         , transforms, adjacency_matrix)
        # B, 7, 128, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        solo_mask_x = F.interpolate(self.mask_mcnn.classifier(     mask_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_mask_x = F.interpolate(self.mask_mcnn.classifier(aggr_mask_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x

    def aggregate_features(self, x, transforms, adjacency_matrix) -> torch.Tensor:
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