import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from data.config import SemanticCloudConfig
from data.mask_warp import get_single_relative_img_transform
from model.large_mcnn import LearningToDownsampleWide, TransposedMCNN
from model.large_mcnn import _DSConv, Classifier, TransposedClassifier
from model.large_mcnn import GlobalFeatureExtractor, FeatureFusionModule
from model.base import SoloAggrSemanticsMask, AggrSemanticsSoloMask

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
        self.graph_aggr_conv1 = _DSConv(dw_channels=128, out_channels=128)
        self.graph_aggr_conv2 = _DSConv(dw_channels=128, out_channels=128)

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
        # B, 128, 80, 108
        sseg_x = self.sseg_mcnn.global_feature_extractor(sseg_shared)
        mask_x = self.mask_mcnn.global_feature_extractor(mask_shared)
        # B, 128, 80, 108
        sseg_x = self.sseg_mcnn.feature_fusion(sseg_shared, sseg_x)
        mask_x = self.mask_mcnn.feature_fusion(mask_shared, mask_x)
        # add ego car masks
        sseg_x = sseg_x + F.interpolate(car_masks.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear', align_corners=True)
        mask_x = mask_x + F.interpolate(car_masks.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear', align_corners=True)
        # B, 128, 80, 108
        # 2 stage message passing
        aggr_sseg_x = self.aggregate_features(mask_x * sseg_x, transforms, adjacency_matrix)
        aggr_sseg_x = self.graph_aggr_conv1(aggr_sseg_x)
        aggr_sseg_x = self.aggregate_features(aggr_sseg_x    , transforms, adjacency_matrix)
        aggr_sseg_x = self.graph_aggr_conv2(aggr_sseg_x)
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


class DualTransposedMCNN3x(SoloAggrSemanticsMask):
    """
    two MCNNTs, one for mask, other for semantics. outputs solo & aggr versions of both.
    the mask mcnn is shallower than semantic. mask aggregation is directly performed on
    the output of mask subnets.
    """
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str):
        super().__init__()
        self.output_size = output_size
        self.sem_cfg = sem_cfg
        self.aggregation_type = aggr_type
        # semantic subnet
        self.sseg_mcnn = TransposedMCNN(num_classes, output_size, sem_cfg, aggr_type)
        self.graph_aggr_conv1 = _DSConv(dw_channels=128, out_channels=128)
        self.graph_aggr_conv2 = _DSConv(dw_channels=128, out_channels=128)
        # mask subnet
        self.mask_feature_extractor = GlobalFeatureExtractor(in_channels=64,
                                                             block_channels=(64, 96, 128),
                                                             t=6, num_blocks=(3, 3, 3),
                                                             pool_sizes=(2, 4, 6, 8))
        self.mask_feature_fusion = FeatureFusionModule(highres_in_channels=64,
                                                      lowres_in_channels=128,
                                                      out_channels=128,
                                                      scale_factor=4)
        self.mask_classifier = TransposedClassifier(128, 1)
        self.mask_aggr_conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        # semantic aggregation parameters
        self.sem_cf_h, self.sem_cf_w = 80, 108
        self.sem_ppm = self.sem_cfg.pix_per_m(self.sem_cf_h, self.sem_cf_w)
        self.sem_center_x = self.sem_cfg.center_x(self.sem_cf_w)
        self.sem_center_y = self.sem_cfg.center_y(self.sem_cf_h)
        # mask aggregation parameters
        self.msk_cf_h, self.msk_cf_w = 128, 108
        self.msk_ppm = self.sem_cfg.pix_per_m(self.msk_cf_h, self.msk_cf_w)
        self.msk_center_x = self.sem_cfg.center_x(self.msk_cf_w)
        self.msk_center_y = self.sem_cfg.center_y(self.msk_cf_h)
        # model specification
        self.output_count = 4
        self.model_type = 'semantic+mask'
        self.notes = 'not small, probably not fast but all in one'

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
        # B, 128, 80, 108
        # 2 stage message passing for semantics
        aggr_sseg_x = self.aggregate_features(mask_x * sseg_x, transforms, adjacency_matrix,
                                              self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y)
        aggr_sseg_x = self.graph_aggr_conv1(aggr_sseg_x)
        aggr_sseg_x = self.aggregate_features(aggr_sseg_x, transforms, adjacency_matrix,
                                              self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y)
        aggr_sseg_x = self.graph_aggr_conv2(aggr_sseg_x)
        # solo mask estimation
        # B, 1, 128, 108
        solo_mask_preds = self.mask_classifier(mask_x)
        # B, 1, 256, 205
        solo_mask_x = F.interpolate(solo_mask_preds, self.output_size, mode='bilinear', align_corners=True)
        # mask aggregation
        # B, 1, 128, 108
        aggr_mask_x = self.aggregate_features(solo_mask_preds, transforms, adjacency_matrix,
                                              self.msk_ppm, self.msk_cf_h, self.msk_cf_w,
                                              self.msk_center_x, self.msk_center_y)
        # B, 1, 256, 205
        aggr_mask_x = F.interpolate(aggr_mask_x, self.output_size, mode='bilinear', align_corners=True)
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x

    def aggregate_features(self, x, transforms, adjacency_matrix, ppm, cf_h, cf_w, center_x, center_y) -> torch.Tensor:
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, ppm, cf_h, cf_w,
                                                             center_x, center_y).to(transforms.device)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(cf_h, cf_w),
                                                 flags=self.aggregation_type)
            # applying the adjacency matrix (difficulty)
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)
        return aggregated_features

class DualTransposedMCNN2x(AggrSemanticsSoloMask):
    """
    large & wide MCNN with solo mask & aggreagated semantic prediction
    """
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str):
        super().__init__()
        self.output_size = output_size
        self.sem_cfg = sem_cfg
        self.learning_to_downsample = LearningToDownsampleWide(dw_channels1=32,
                                                               dw_channels2=48,
                                                               out_channels=64)
        self.semantic_global_feature_extractor = GlobalFeatureExtractor(in_channels=64,
                                                                        block_channels=(64, 96, 128),
                                                                        t=8,
                                                                        num_blocks=(3, 3, 3),
                                                                        pool_sizes=(2, 4, 6, 8))
        self.semantic_feature_fusion = FeatureFusionModule(highres_in_channels=64,
                                                           lowres_in_channels=128,
                                                           out_channels=128,
                                                           scale_factor=4)
        self.classifier = TransposedClassifier(128, num_classes)
        # none of the pyramid stages can be 1 due to instance norm issue
        # https://github.com/pytorch/pytorch/issues/45687
        self.mask_global_feature_extractor = GlobalFeatureExtractor(in_channels=64,
                                                                    block_channels=(64, 96, 128),
                                                                    t=6,
                                                                    num_blocks=(3, 3, 3),
                                                                    pool_sizes=(2, 3, 4, 5))
        self.mask_feature_fusion = FeatureFusionModule(highres_in_channels=64,
                                                       lowres_in_channels=128,
                                                       out_channels=128,
                                                       scale_factor=4)
        self.maskifier = Classifier(128, 1)
        # aggregation parameters
        self.cf_h, self.cf_w = 80, 108
        self.ppm = self.sem_cfg.pix_per_m(self.cf_h, self.cf_w)
        self.center_x = self.sem_cfg.center_x(self.cf_w)
        self.center_y = self.sem_cfg.center_y(self.cf_h)
        # mask aggregation parameters
        self.msk_cf_h, self.msk_cf_w = 128, 108
        self.msk_ppm = self.sem_cfg.pix_per_m(self.msk_cf_h, self.msk_cf_w)
        self.msk_center_x = self.sem_cfg.center_x(self.msk_cf_w)
        self.msk_center_y = self.sem_cfg.center_y(self.msk_cf_h)
        self.aggregation_type = aggr_type
        # model specification
        self.output_count = 2
        self.model_type = 'semantic+mask'
        self.notes = 'small, fast, solo mask for latent masking'

    def forward(self, x, transforms, adjacency_matrix):
        # B, 3, 480, 640: input size
        # B, 64, 80, 108
        shared = self.learning_to_downsample(x)
        # ------------mask branch------------
        # B, 128, 15, 20
        x_mask = self.mask_global_feature_extractor(shared)
        # B, 128, 80, 108
        x_mask = self.mask_feature_fusion(shared, x_mask)
        # ----------semantic branch----------
        # B, 128, 15, 20
        x_semantic = self.semantic_global_feature_extractor(shared)
        # B, 128, 80, 108
        x_semantic = self.semantic_feature_fusion(shared, x_semantic)
        # --latent masking into aggregation--
        # B, 128, 80, 108
        x_semantic = self.aggregate_features(torch.sigmoid(x_mask) * x_semantic,
                                             transforms, adjacency_matrix)
        # B, 7, 80, 108
        x_semantic = self.classifier(x_semantic)
        # B, 1, 80, 108
        x_mask = self.maskifier(x_mask)
        # ----------- upsampling ------------
        # B, 7, 256, 205
        x_semantic = F.interpolate(x_semantic, self.output_size, mode='bilinear', align_corners=True)
        # B, 1, 256, 205
        x_mask = torch.sigmoid(F.interpolate(x_mask, self.output_size, mode='bilinear', align_corners=True))
        return x_semantic, x_mask

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