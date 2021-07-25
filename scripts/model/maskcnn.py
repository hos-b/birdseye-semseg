import torch
import kornia
from torch.nn import functional as F
from data.mask_warp import get_single_relative_img_transform
from data.config import SemanticCloudConfig
from model.large_mcnn import LearningToDownsampleWide, GlobalFeatureExtractor, FeatureFusionModule, Classifier

class MaskCNN(torch.nn.Module):
    """
    FastSCNN slightly modified to output full masks
    with aggregation from other agents.
    """
    def __init__(self, output_size, sem_cfg: SemanticCloudConfig, aggr_type):
        super().__init__()
        self.learning_to_downsample = LearningToDownsampleWide(dw_channels1=32,
                                                               dw_channels2=48,
                                                               out_channels=64)
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels=64,
                                                               block_channels=(64, 96, 128),
                                                               t=8,
                                                               num_blocks=(3, 3, 3),
                                                               pool_sizes=(2, 4, 6, 8))
        self.feature_fusion = FeatureFusionModule(highres_in_channels=64,
                                                 lowres_in_channels=128,
                                                 out_channels=128,
                                                 scale_factor=4)
        self.mask_prediction = Classifier(128, 1)
        self.output_size = output_size
        self.sem_cfg = sem_cfg
        # set aggregation parameters
        self.aggregation_type = aggr_type
        self.cf_h, self.cf_w = 80, 108
        self.ppm = self.sem_cfg.pix_per_m(self.cf_h, self.cf_w)
        self.center_x = self.sem_cfg.center_x(self.cf_w)
        self.center_y = self.sem_cfg.center_y(self.cf_h)
        # model specification
        self.output_count = 2
        self.model_type = 'mask-only'
        self.notes = 'small, fast'

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