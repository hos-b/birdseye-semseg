#########################################################
# based on  https://github.com/Tramac/Fast-SCNN-pytorch #
#########################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from data.config import SemanticCloudConfig
from data.mask_warp import get_single_relative_img_transform

class TransposedMCNN(nn.Module):
    """
    large and wide MCNN with extra deconv layers in the segmentation decoder
    """
    def __init__(self, num_classes, output_size, sem_cfg: SemanticCloudConfig, aggr_type: str):
        super().__init__()
        self.output_size = output_size
        self.sem_cfg = sem_cfg
        self.aggregation_type = aggr_type
        self.learning_to_downsample = LearningToDownsampleWide(dw_channels1=32,
                                                               dw_channels2=48,
                                                               out_channels=64)
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels=64,
                                                                        block_channels=(64, 128, 256),
                                                                        t=8,
                                                                        num_blocks=(4, 8, 8),
                                                                        pool_sizes=(6, 8, 10, 12))
        self.feature_fusion = FeatureFusionModule(highres_in_channels=64,
                                                           lowres_in_channels=256,
                                                           out_channels=256,
                                                           scale_factor=4)
        self.classifier = TransposedClassifier(256, num_classes)
        self.cf_h, self.cf_w = 80, 108
        self.ppm = self.sem_cfg.pix_per_m(self.cf_h, self.cf_w)
        self.center_x = self.sem_cfg.center_x(self.cf_w)
        self.center_y = self.sem_cfg.center_y(self.cf_h)

    def forward(self, x, transforms, adjacency_matrix, car_masks):
        # B, 3, 480, 640: input size
        # B, 64, 80, 108
        shared = self.learning_to_downsample(x)
        # B, 256, 15, 20
        x = self.global_feature_extractor(shared)
        # B, 256, 80, 108
        x = self.feature_fusion(shared, x)
        # add ego car masks
        x = x + F.interpolate(car_masks.unsqueeze(1), size=(self.cf_h, self.cf_w), mode='bilinear', align_corners=True)
        # B, 256, 80, 108
        aggr_x = self.aggregate_features(x, transforms, adjacency_matrix)
        # B, 7, 128, 205
        solo_x = F.interpolate(self.classifier(x), self.output_size, mode='bilinear', align_corners=True)
        aggr_x = F.interpolate(self.classifier(aggr_x), self.output_size, mode='bilinear', align_corners=True)
        return solo_x, aggr_x

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

# ------------------------------------------------------ Modules ------------------------------------------------------

class LearningToDownsample(nn.Module):
    """Learning to downsample module"""
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvINReLU(in_channels=3, out_channels=dw_channels1, kernel_size=3, stride=2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x

class LearningToDownsampleWide(nn.Module):
    """Learning to downsample module"""
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsampleWide, self).__init__()
        self.conv = _ConvINReLU(in_channels=3, out_channels=dw_channels1, kernel_size=(4, 3),stride=3)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 1, kernel_size=(3, 2))
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2, kernel_size=(3, 2))

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x

class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 t=6, num_blocks=(3, 3, 3), pool_sizes=(1, 2, 3, 6)):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], block_channels[2], pool_sizes)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x

class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highres_in_channels, lowres_in_channels, out_channels, scale_factor=4):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lowres_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.InstanceNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highres_in_channels, out_channels, 1),
            nn.InstanceNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)

class Classifier(nn.Module):
    """Classifier"""

    def __init__(self, dw_channels, num_classes, stride=1):
        super(Classifier, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x

class TransposedClassifier(nn.Module):
    """
    Classifier + transposed convolution for upsampling
    """
    def __init__(self, dw_channels, num_classes, stride=1):
        super(TransposedClassifier, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(dw_channels, dw_channels, kernel_size=(25, 1))
        self.tconv2 = nn.ConvTranspose2d(dw_channels, dw_channels, kernel_size=(25, 1))
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x

class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""
    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvINReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out

class PyramidPooling(nn.Module):
    """Pyramid pooling module"""
    def __init__(self, in_channels, out_channels, pool_sizes):
        super(PyramidPooling, self).__init__()
        stage_count = len(pool_sizes)
        assert in_channels % stage_count == 0, f'{in_channels} is not divisble by number of pooling stages {stage_count}'
        intermediate_channels = in_channels // stage_count
        self.stages = nn.ModuleList([self._make_stage(in_channels, intermediate_channels, size) for size in pool_sizes])
        self.bottleneck = _ConvINReLU(in_channels * 2, out_channels, 1)

    def _make_stage(self, in_channels, out_channels, avgpool_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(avgpool_size),
            _ConvINReLU(in_channels, out_channels, kernel_size=1)
        )

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        pooled_features = [F.interpolate(stage(x), size=size, mode='bilinear', align_corners=True) for stage in self.stages]
        x = torch.cat(pooled_features + [x], dim=1)
        return self.bottleneck(x)

class _ConvINReLU(nn.Module):
    """Conv-BN-ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(_ConvINReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""
    def __init__(self, dw_channels, out_channels, stride=1, kernel_size=3):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, kernel_size, stride, 1, groups=dw_channels, bias=False),
            nn.InstanceNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)