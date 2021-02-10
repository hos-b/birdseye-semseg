import torch
import kornia
import torch.nn as nn
import torch.functional as F

from data.mask_warp import get_relative_img_transform
from data.config import SemanticCloudConfig
#pylint: disable=E1101
#pylint: disable=not-callable 
"""
input --> downsample --> bottleneck --------
            |                               |
            |                                --> aggregation layer ---> total mask, total prediction [loss #2]
            |                                           ^   ^
             --> mask prediction [loss #1] -------------|   |
                                                        |   |
                 transform -----------------------------    |
                                                            |
transform and compressed features from other agents --------
"""




class AgentPool(torch.nn.Module):
    def __init__(self, config: SemanticCloudConfig, num_classes, conn_drop_prob = 0.01, output_size=(256, 205)):
        super(AgentPool).__init__()
        self.cfg = config
        self.drop_probability = conn_drop_prob

        self.downsample = LearningToDownsample(in_channels=32, mid_channels=48, out_channels=64)
        # 3 x 3 stages of linear bottleneck for feature compression
        self.compression = nn.Sequential(
            LinearBottleneck(in_channels=64, out_channels=64, expansion=4, stride=2),
            LinearBottleneck(in_channels=64, out_channels=64, expansion=4, stride=1), # w/ skip connection
            LinearBottleneck(in_channels=64, out_channels=64, expansion=4, stride=1), # w/ skip connection
            # ----------------------------------------------------------------------
            LinearBottleneck(in_channels=64, out_channels=96, expansion=4, stride=2),
            LinearBottleneck(in_channels=96, out_channels=96, expansion=4, stride=1), # w/ skip connection
            LinearBottleneck(in_channels=96, out_channels=96, expansion=4, stride=1), # w/ skip connection
            # ----------------------------------------------------------------------
            LinearBottleneck(in_channels= 96, out_channels=128, expansion=4, stride=2),
            LinearBottleneck(in_channels=128, out_channels=128, expansion=4, stride=1), # w/ skip connection
            LinearBottleneck(in_channels=128, out_channels=128, expansion=4, stride=1), # w/ skip connection
        )
        # 2 x 2 stages of linear bottleneck for fov estimation
        self.mask_prediction = nn.Sequential(
            LinearBottleneck(in_channels=64, out_channels=64, expansion=4, stride=2),
            LinearBottleneck(in_channels=64, out_channels=64, expansion=4, stride=1), # w/ skip connection
            # ----------------------------------------------------------------------
            LinearBottleneck(in_channels=64, out_channels=96, expansion=4, stride=2),
            LinearBottleneck(in_channels=96, out_channels=96, expansion=4, stride=1), # w/ skip connection
            # DSConv with sigmoid activation
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1,
                      padding=1, groups=96, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.Conv2d(in_channels=96, out_channels=1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.pyramid_pooling = PyramidPooling(in_channels=128, out_channels=128)
        self.classifier = nn.Sequential(
            DWConv(in_channels=128, out_channels=128, kernel_size=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            DSConv(in_channels=128, out_channels=128, stride=1),
            DSConv(in_channels=128, out_channels=128, stride=1),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)
        )
    
    def forward(self, rgbs, transforms, masks):
        """
        input:
            rgbs:       agent_count x 480 x 640
            transforms: agent_count x 4 x 4
            masks:      agent_count x 256 x 205
        output:
            predicted masks: agent_count x 256 x 205
            aggr_masks:      agent_count x 256 x 205
            aggr_preds:      agent_count x 256 x 205
        """
        rgbs, transforms, masks = self.drop_agent_data(rgbs, transforms, masks)

        hi_res_feature = self.downsample(rgbs)
        compressed_features = self.compression(hi_res_feature)
        predicted_masks = self.mask_prediction(hi_res_feature)
        aggregated_features = self.aggregate_data(compressed_features,
                                                  predicted_masks,
                                                  transforms)
        pooled_features = self.pyramid_pooling(aggregated_features)
        upsampled_features = F.interpolate(pooled_features, scale_factor=4, mode='bilinear', align_corners=True)
        return self.classifier(upsampled_features)
    
    def aggregate_data(self, compressed_features, predicted_masks, transforms):
        """
        warps compressed features into the view of each agent, before pooling 
        the results
        """
        agent_count = transforms.shape[0]
        # TODO: make constant
        h, w = compressed_features.shape[2], compressed_features.shape[3]
        ppm = ((h / self.cfg.cloud_x_span) + (w / self.cfg.cloud_y_span)) / 2.0
        center_x = int((self.cfg.cloud_max_x / self.cfg.cloud_x_span) * h)
        center_y = int((self.cfg.cloud_max_y / self.cfg.cloud_y_span) * w)

        # aggregating
        aggregated_features = torch.zeros_like(compressed_features)
        for i in range(agent_count):
            relative_tfs = get_relative_img_transform(transforms, i, ppm, h,
                                                      w, center_x, center_y)
            warped_features = kornia.warp_affine(compressed_features, relative_tfs, dsize=(h, w), flags='bilinear')
            aggregated_features[i, ...] = warped_features.sum(dim=0) / agent_count

        return aggregated_features
    
    def drop_agent_data(self, rgbs, transforms, masks):
        # connection drops
        drop_probs = torch.full((rgbs.shape[0], ), self.drop_probability)
        drops = torch.bernoulli(drop_probs).long()
        return rgbs[drops != 1, :, :], transforms[drops != 1, :, :], masks[drops != 1, :, :]

class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, in_channels=32, mid_channels=48, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.conv = ConvBNReLU(in_channels=3, out_channels=in_channels, kernel_size=3, stride=2)
        self.dsconv1 = DSConv(in_channels, mid_channels, 2)
        self.dsconv2 = DSConv(mid_channels, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x

class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class DSConv(nn.Module):
    """
    Depthwise Separable Convolutions:
    factorization of a normal convolution into a depth-wise convolution
    followed by a 1 x 1 convolution to get the desired number of output
    channels. uses significantly less parameters than a typical Conv2d.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                      padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False), # 1x1 conv
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class DWConv(nn.Module):
    """
    Depthwise Convolution:
    each input channel is convolved with `out_channels`/`in_channels`
    filters to produce the same number of output channels. the number
    of `out_channels` should therefor be divisible by `in_channels`.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""
    def __init__(self, in_channels, out_channels, expansion=6, stride=2):
        super(LinearBottleneck, self).__init__()
        self.skip_enable = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            ConvBNReLU(in_channels, in_channels * expansion, 1),
            # dw
            DWConv(in_channels * expansion, in_channels * expansion, stride),
            # pw-linear
            nn.Conv2d(in_channels * expansion, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.skip_enable:
            return x + self.block(x)
        return self.block(x)

class PyramidPooling(nn.Module):
    """Pyramid pooling module"""
    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv2 = ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv3 = ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv4 = ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.out = ConvBNReLU(in_channels * 2, out_channels, kernel_size=1)

        self.pool_1 = nn.AdaptiveAvgPool2d(1)
        self.pool_2 = nn.AdaptiveAvgPool2d(2)
        self.pool_3 = nn.AdaptiveAvgPool2d(3)
        self.pool_6 = nn.AdaptiveAvgPool2d(6)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool_1(x)), size)
        feat2 = self.upsample(self.conv2(self.pool_2(x)), size)
        feat3 = self.upsample(self.conv3(self.pool_3(x)), size)
        feat4 = self.upsample(self.conv4(self.pool_6(x)), size)
        return self.out(torch.cat([x, feat1, feat2, feat3, feat4], dim=1))