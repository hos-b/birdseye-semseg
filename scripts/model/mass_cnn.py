import torch
import kornia
import torch.nn as nn
import torch.nn.functional as F

from data.mask_warp import get_relative_img_transform
from data.config import SemanticCloudConfig

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

__all__ = ['MassCNN']

class MassCNN(torch.nn.Module):
    def __init__(self, config: SemanticCloudConfig, num_classes, conn_drop_prob = 0.01, output_size=(256, 205)):
        super(MassCNN, self).__init__()
        self.cfg = config
        self.drop_probability = conn_drop_prob
        self.output_size = output_size
        self.downsample = LearningToDownsample(in_channels=32, mid_channels=48, out_channels=64)
        # 3 x 3 stages of linear bottleneck for feature compression
        self.compression = nn.Sequential(
            LinearBottleneck(in_channels=64, out_channels=64, expansion=4, stride=2, skip_en=False),
            LinearBottleneck(in_channels=64, out_channels=64, expansion=4, stride=1, skip_en=True),
            LinearBottleneck(in_channels=64, out_channels=64, expansion=4, stride=1, skip_en=True),
            # -----------------------------------------------------------------------------------------------
            LinearBottleneck(in_channels=64, out_channels=96, expansion=4, stride=2, skip_en=False),
            LinearBottleneck(in_channels=96, out_channels=96, expansion=4, stride=1, skip_en=True),
            LinearBottleneck(in_channels=96, out_channels=96, expansion=4, stride=1, skip_en=True),
            # # ---------------------------------------------------------------------------------------------
            LinearBottleneck(in_channels= 96, out_channels=128, expansion=4, stride=2, skip_en=False),
            LinearBottleneck(in_channels=128, out_channels=128, expansion=4, stride=1, skip_en=True),
            LinearBottleneck(in_channels=128, out_channels=128, expansion=4, stride=1, skip_en=True),
        )
        # 2 x 2 stages of linear bottleneck for fov estimation
        self.mask_prediction = nn.Sequential(
            LinearBottleneck(in_channels=64, out_channels=64, expansion=4, stride=2),
            LinearBottleneck(in_channels=64, out_channels=64, expansion=4, stride=1, skip_en=True),
            # ----------------------------------------------------------------------
            LinearBottleneck(in_channels=64, out_channels=96, expansion=4, stride=2),
            LinearBottleneck(in_channels=96, out_channels=96, expansion=4, stride=1, skip_en=True),
            # DSConv with sigmoid activation + average pooling for size
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 6), padding=1, groups=96, bias=False),
            nn.AdaptiveAvgPool2d(self.output_size),
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
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        )
    
    def forward(self, rgbs, transforms):
        """
        input:
            rgbs:       agent_count x 480 x 640
            transforms: agent_count x 4 x 4
        output:
            predicted masks: agent_count x 256 x 205
            aggr_masks:      agent_count x 256 x 205
        """
        rgbs, transforms = self.drop_agent_data(rgbs, transforms)
        # [A, 64, 241, 321]
        hi_res_features = self.downsample(rgbs)
        # [A, 128, 238, 318]
        compressed_features = self.compression(hi_res_features)
        # [A,   1, 256, 206]
        predicted_masks = self.mask_prediction(hi_res_features)
        # [A, 128, 238, 318]
        aggregated_features = self.aggregate_data(compressed_features,
                                                  predicted_masks,
                                                  transforms)

        # [A, 128, 256, 205]
        pooled_features = self.pyramid_pooling(aggregated_features, self.output_size)
        return predicted_masks, self.classifier(pooled_features)
    
    def aggregate_data(self, compressed_features, predicted_masks, transforms):
        """
        warps compressed features into the view of each agent, before pooling 
        the results
        """
        # calculating constants
        agent_count = transforms.shape[0]
        cf_h, cf_w = 238, 318 # compressed_features.shape[2], compressed_features.shape[3]
        ppm = 12.71 # ((cf_h / self.cfg.cloud_x_span) + (cf_w / self.cfg.cloud_y_span)) / 2.0
        center_x = 190 # int((self.cfg.cloud_max_x / self.cfg.cloud_x_span) * cf_h)
        center_y = 159 # int((self.cfg.cloud_max_y / self.cfg.cloud_y_span) * cf_w)
        # aggregating [A, 128, 238, 318]
        aggregated_features = torch.zeros_like(compressed_features)
        for i in range(agent_count):
            relative_tfs = get_relative_img_transform(transforms, i, ppm, cf_h, cf_w, center_x, center_y)
            warped_features = kornia.warp_affine(compressed_features, relative_tfs, dsize=(cf_h, cf_w), flags='bilinear')
            aggregated_features[i, ...] = warped_features.sum(dim=0) / agent_count

        return aggregated_features
    
    def drop_agent_data(self, rgbs, transforms):
        """
        simulate connection drops between cars or non transmitting cars
        """
        drop_probs = torch.ones((rgbs.shape[0], ), dtype=torch.float32) * self.drop_probability
        drops = torch.bernoulli(drop_probs).long()
        return rgbs[drops != 1, :, :], transforms[drops != 1, :, :]

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
    def __init__(self, in_channels, out_channels, expansion=6, stride=2, padding=0, skip_en=False):
        super(LinearBottleneck, self).__init__()
        self.skip_enable = skip_en
        self.block = nn.Sequential(
            # pw
            ConvBNReLU(in_channels, in_channels * expansion, kernel_size=3, stride=1, padding=padding),
            # dw
            DWConv(in_channels * expansion, in_channels * expansion, stride),
            # pw-linear
            nn.Conv2d(in_channels * expansion, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        blk = self.block(x)
        if self.skip_enable:
            if x.shape[2:] != blk.shape[2:]:
                print(f"{x.shape[2:]} != {blk.shape[2:]}")
            return x + blk
        return blk

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

        self.pool_1 = nn.AdaptiveAvgPool2d(4)  # 1
        self.pool_2 = nn.AdaptiveAvgPool2d(16) # 2
        self.pool_3 = nn.AdaptiveAvgPool2d(32) # 3
        self.pool_4 = nn.AdaptiveAvgPool2d(64) # 6

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x, size):
        feat2 = self.upsample(self.conv2(self.pool_2(x)), size)
        feat3 = self.upsample(self.conv3(self.pool_3(x)), size)
        feat1 = self.upsample(self.conv1(self.pool_1(x)), size)
        feat4 = self.upsample(self.conv4(self.pool_4(x)), size)
        return self.out(torch.cat([self.upsample(x, size), feat1, feat2, feat3, feat4], dim=1))