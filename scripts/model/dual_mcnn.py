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

    def forward(self, rgbs, transforms, adjacency_matrix, car_masks, **kwargs):
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
        solo_mask_x = F.interpolate(torch.sigmoid(self.mask_mcnn.classifier(mask_x)), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_mask_x = F.interpolate(torch.sigmoid(self.mask_mcnn.classifier(aggr_mask_x)), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x

    def aggregate_features(self, x, transforms, adjacency_matrix) -> torch.Tensor:
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, self.ppm,
                                                             self.center_x, self.center_y).to(transforms.device)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(self.cf_h, self.cf_w),
                                                 mode=self.aggregation_type)
            # applying the adjacency matrix (difficulty)
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)
        return aggregated_features


class DualTransposedMCNN3x(SoloAggrSemanticsMask):
    """
    two MCNNTs, one for mask, other for semantics. outputs solo & aggr versions of both.
    the mask mcnn is shallower than semantic. mask aggregation is directly performed on
    the output of mask subnets. unlike MCNN4x, the original L2DS features are shared.
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
        # mask subnet (t=6 instead of 8 as in MCNNT)
        self.mask_feature_extractor = GlobalFeatureExtractor(in_channels=64,
                                                             block_channels=(64, 96, 128),
                                                             t=6, num_blocks=(3, 3, 3),
                                                             pool_sizes=(2, 4, 6, 8))
        self.mask_feature_fusion = FeatureFusionModule(highres_in_channels=64,
                                                       lowres_in_channels=128,
                                                       out_channels=128,
                                                       scale_factor=4)
        self.mask_classifier = Classifier(128, 1)
        self.mask_aggr_conv = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
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
        self.notes = 'not small, probably not fast but all in one'

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
        # B, 1, 80, 108
        solo_mask_x = torch.sigmoid(self.mask_classifier(mask_x))
        # B, 1, 256, 205
        solo_mask_x = F.interpolate(solo_mask_x, self.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size
        # B, 1, 256, 205
        aggr_mask_x = self.aggregate_features(solo_mask_x.detach(), transforms, adjacency_matrix,
                                              self.msk_ppm, self.msk_cf_h, self.msk_cf_w,
                                              self.msk_center_x, self.msk_center_y)
        aggr_mask_x = torch.sigmoid(self.mask_aggr_conv(aggr_mask_x))
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x

    def aggregate_features(self, x, transforms, adjacency_matrix, ppm, cf_h, cf_w, center_x, center_y) -> torch.Tensor:
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, ppm,
                                                             center_x, center_y).to(transforms.device)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(cf_h, cf_w),
                                                 mode=self.aggregation_type)
            # applying the adjacency matrix (difficulty)
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)
        return aggregated_features

    @torch.no_grad()
    def infer_preaggr(self, rgb, car_mask, **kwargs):
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
        # return latenlty masked features and ego mask
        return sseg_x * mask_x, ego_mask

    @torch.no_grad()
    def infer_aggregate(self, agent_index, latent_features, ego_masks, transforms, **kwargs):
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
        # applying the adjacency matrix (difficulty)
        final_features = self.graph_aggr_conv2(warped_features.sum(dim=0).unsqueeze(0))
        final_semantics = F.interpolate(self.sseg_mcnn.classifier(final_features), self.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size (only done for ego agent)
        relative_tfs = get_single_relative_img_transform(transforms, agent_index, self.msk_ppm, self.msk_center_x, self.msk_center_y).to(transforms.device)
        warped_masks = kornia.warp_affine(ego_masks, relative_tfs, dsize=(self.msk_cf_h, self.msk_cf_w), mode=self.aggregation_type)
        final_mask = torch.sigmoid(self.mask_aggr_conv(warped_masks.sum(dim=0).unsqueeze(0)))
        return final_semantics, final_mask


class DualTransposedMCNN3x_1x(DualTransposedMCNN3x):
    """
    DualTransposedMCNN3x but with a single aggregation step, no post-processing
    for evaluation purposes
    """
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
        # B, 128, 80, 108
        # 2 stage message passing for semantics
        aggr_sseg_x = self.aggregate_features(mask_x * sseg_x, transforms, adjacency_matrix,
                                              self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y)
        # solo mask estimation
        # B, 1, 80, 108
        solo_mask_x = torch.sigmoid(self.mask_classifier(mask_x))
        # B, 1, 256, 205
        solo_mask_x = F.interpolate(solo_mask_x, self.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size
        # B, 1, 256, 205
        aggr_mask_x = self.aggregate_features(solo_mask_x.detach(), transforms, adjacency_matrix,
                                              self.msk_ppm, self.msk_cf_h, self.msk_cf_w,
                                              self.msk_center_x, self.msk_center_y)
        aggr_mask_x = torch.sigmoid(self.mask_aggr_conv(aggr_mask_x))
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x


class DualTransposedMCNN3x_1xPost(DualTransposedMCNN3x):
    """
    DualTransposedMCNN3x but with a single aggregation step + post-processing
    for evaluation purposes
    """
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
        # B, 128, 80, 108
        # 2 stage message passing for semantics
        aggr_sseg_x = self.aggregate_features(mask_x * sseg_x, transforms, adjacency_matrix,
                                              self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y)
        aggr_sseg_x = self.graph_aggr_conv1(aggr_sseg_x)
        # solo mask estimation
        # B, 1, 80, 108
        solo_mask_x = torch.sigmoid(self.mask_classifier(mask_x))
        # B, 1, 256, 205
        solo_mask_x = F.interpolate(solo_mask_x, self.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size
        # B, 1, 256, 205
        aggr_mask_x = self.aggregate_features(solo_mask_x.detach(), transforms, adjacency_matrix,
                                              self.msk_ppm, self.msk_cf_h, self.msk_cf_w,
                                              self.msk_center_x, self.msk_center_y)
        aggr_mask_x = torch.sigmoid(self.mask_aggr_conv(aggr_mask_x))
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x


class DualTransposedMCNN3xFlatMasking(DualTransposedMCNN3x):
    """
    normal DualTransposedMCNN3x with flat (instead of tensor) masking
    """
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
        # solo mask estimation
        # B, 1, 80, 108
        solo_mask_x = torch.sigmoid(self.mask_classifier(mask_x))
        # B, 128, 80, 108
        # 2 stage message passing for semantics
        aggr_sseg_x = self.aggregate_features(solo_mask_x * sseg_x, transforms, adjacency_matrix,
                                              self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y)
        aggr_sseg_x = self.graph_aggr_conv1(aggr_sseg_x)
        aggr_sseg_x = self.aggregate_features(aggr_sseg_x, transforms, adjacency_matrix,
                                              self.sem_ppm, self.sem_cf_h, self.sem_cf_w,
                                              self.sem_center_x, self.sem_center_y)
        aggr_sseg_x = self.graph_aggr_conv2(aggr_sseg_x)
        # mask upsampling
        # B, 1, 256, 205
        solo_mask_x = F.interpolate(solo_mask_x, self.output_size, mode='bilinear', align_corners=True)
        # mask aggregation on full size
        # B, 1, 256, 205
        aggr_mask_x = self.aggregate_features(solo_mask_x.detach(), transforms, adjacency_matrix,
                                              self.msk_ppm, self.msk_cf_h, self.msk_cf_w,
                                              self.msk_center_x, self.msk_center_y)
        aggr_mask_x = torch.sigmoid(self.mask_aggr_conv(aggr_mask_x))
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x


class DualMCNNT3Expansive(DualTransposedMCNN3x):
    """
    aggregation in output size to save details.
    """
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str):
        super().__init__(num_classes, output_size, sem_cfg, aggr_type)
        # aggregation parameters
        self.feature_h, self.feature_w = 80, 108
        self.aggr_h, self.aggr_w = 256, 205
        self.aggr_ppm = self.sem_cfg.pix_per_m(self.aggr_h, self.aggr_w)
        self.aggr_center_x = self.sem_cfg.center_x(self.aggr_w)
        self.aggr_center_y = self.sem_cfg.center_y(self.aggr_h)
        # model specification
        self.output_count = 4
        self.model_type = 'semantic+mask'
        self.notes = 'not small, probably not fast but all in one'

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
        sseg_x = sseg_x + F.interpolate(car_masks.unsqueeze(1), size=(self.feature_h, self.feature_w), mode='bilinear', align_corners=True)
        mask_x = mask_x + F.interpolate(car_masks.unsqueeze(1), size=(self.feature_h, self.feature_w), mode='bilinear', align_corners=True)
        # upsampling to the output size before aggregation
        # B, 128, 256, 205
        sseg_x = F.interpolate(sseg_x, size=self.output_size, mode='bilinear', align_corners=True)
        mask_x = F.interpolate(mask_x, size=self.output_size, mode='bilinear', align_corners=True)
        # 2 stage message passing for semantics
        aggr_sseg_x = self.aggregate_features(mask_x * sseg_x, transforms, adjacency_matrix)
        aggr_sseg_x = self.graph_aggr_conv1(aggr_sseg_x)
        aggr_sseg_x = self.aggregate_features(aggr_sseg_x, transforms, adjacency_matrix)
        aggr_sseg_x = self.graph_aggr_conv2(aggr_sseg_x)
        # solo mask estimation
        # B, 1, 256, 205
        solo_mask_x = torch.sigmoid(self.mask_classifier(mask_x))
        # mask aggregation on full size
        # B, 1, 256, 205
        aggr_mask_x = self.aggregate_features(solo_mask_x.detach(), transforms, adjacency_matrix)
        aggr_mask_x = torch.sigmoid(self.mask_aggr_conv(aggr_mask_x))
        # downsampling semantics back to the original feature size
        # B, 128, 80, 108
        sseg_x      = F.interpolate(sseg_x     , size=(self.feature_h, self.feature_w), mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(aggr_sseg_x, size=(self.feature_h, self.feature_w), mode='bilinear', align_corners=True)
        # B, 7, 256, 205
        solo_sseg_x = F.interpolate(self.sseg_mcnn.classifier(     sseg_x), self.output_size, mode='bilinear', align_corners=True)
        aggr_sseg_x = F.interpolate(self.sseg_mcnn.classifier(aggr_sseg_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_sseg_x, solo_mask_x, aggr_sseg_x, aggr_mask_x

    def aggregate_features(self, x, transforms, adjacency_matrix) -> torch.Tensor:
        agent_count = transforms.shape[0]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, self.aggr_ppm,
                                                             self.aggr_center_x, self.aggr_center_y).to(transforms.device)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(self.aggr_h, self.aggr_w),
                                                 mode=self.aggregation_type)
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
        # inference aggregation parameters
        self.inf_cf_h, self.inf_cf_w = 256, 205
        self.inf_ppm = self.sem_cfg.pix_per_m(self.inf_cf_h, self.inf_cf_w)
        self.inf_center_x = self.sem_cfg.center_x(self.inf_cf_w)
        self.inf_center_y = self.sem_cfg.center_y(self.inf_cf_h)
        self.aggregation_type = aggr_type
        # model specification
        self.output_count = 2
        self.model_type = 'semantic+mask'
        self.notes = 'small, fast, solo mask for latent masking'

    def forward(self, rgbs, transforms, adjacency_matrix, car_masks, **kwargs):
        # B, 3, 480, 640: input size
        # B, 64, 80, 108
        shared = self.learning_to_downsample(rgbs)
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
        # -----------add ego mask-----------
        x_semantic = x_semantic + F.interpolate(car_masks.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear', align_corners=True)
        x_mask     = x_mask     + F.interpolate(car_masks.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear', align_corners=True)
        # --latent masking into aggregation--
        # B, 128, 80, 108
        x_semantic = self.aggregate_features(x_mask * x_semantic, transforms, adjacency_matrix)
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
            relative_tfs = get_single_relative_img_transform(transforms, i, self.ppm,
                                                             self.center_x, self.center_y).to(transforms.device)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(self.cf_h, self.cf_w),
                                                 mode=self.aggregation_type)
            # applying the adjacency matrix (difficulty)
            warped_features[outside_fov] = 0
            aggregated_features[i] = warped_features.sum(dim=0)
        return aggregated_features
    
    @torch.no_grad()
    def infer_preaggr(self, rgb, car_mask, **kwargs):
        # 1, 3, 480, 640: input size
        shared = self.learning_to_downsample(rgb)
        x_mask = self.mask_global_feature_extractor(shared)
        x_mask = self.mask_feature_fusion(shared, x_mask)
        x_semantic = self.semantic_global_feature_extractor(shared)
        x_semantic = self.semantic_feature_fusion(shared, x_semantic)
        # -----------add ego mask-----------
        x_semantic = x_semantic + F.interpolate(car_mask.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear', align_corners=True)
        x_mask     = x_mask     + F.interpolate(car_mask.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear', align_corners=True)
        # 1, 7, 80, 108
        x_semantic = self.classifier(x_mask * x_semantic)
        # 1, 1, 80, 108
        x_mask = torch.sigmoid(self.maskifier(x_mask))
        # ----------- upsampling ------------
        # 1, 7, 256, 205
        x_semantic = F.interpolate(x_semantic, self.output_size, mode='bilinear', align_corners=True)
        # 1, 1, 256, 205
        x_mask     = F.interpolate(x_mask, self.output_size, mode='bilinear', align_corners=True)
        return x_semantic, x_mask

    @torch.no_grad()
    def infer_aggregate(self, agent_index, ego_semantics, ego_masks, transforms, **kwargs):
        # hard masking
        ego_semantics *= ego_masks
        # mask aggregation
        relative_tfs = get_single_relative_img_transform(transforms, agent_index, self.inf_ppm, self.inf_center_x, self.inf_center_y).to(transforms.device)
        warped_masks = kornia.warp_affine(ego_masks, relative_tfs, dsize=(self.inf_cf_h, self.inf_cf_w), mode=self.aggregation_type)
        final_mask = torch.sigmoid(warped_masks.sum(dim=0).unsqueeze(0))
        # aggregate semantic predictions
        warped_semantics = kornia.warp_affine(ego_semantics, relative_tfs, dsize=(self.inf_cf_h, self.inf_cf_w), mode='nearest')
        final_semantics = warped_semantics.sum(dim=0)
        return final_semantics, final_mask