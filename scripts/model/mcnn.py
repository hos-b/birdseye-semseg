import torch
import kornia
from torch import nn
from torch.nn import functional as F
from data.mask_warp import get_single_relative_img_transform
from data.config import SemanticCloudConfig

class MCNN(torch.nn.Module):
    def __init__(self, input_channel, num_classes, output_size, sem_cfg: SemanticCloudConfig):
        super().__init__()
        self.learning_to_downsample = LearningToDownsample(input_channel)
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusionModule()
        self.classifier = Classifier(num_classes)
        self.mask_prediction = Classifier(1)
        self.output_size = output_size
        self.sem_cfg = sem_cfg

    def forward(self, x, transforms, adjacency_matrix):
        # B, 3, 480, 640: input size
        # B, 64, 60, 80
        shared = self.learning_to_downsample(x)
        # B, 128, 15, 20
        x = self.global_feature_extractor(shared)
        # B, 128, 60, 80
        x = self.feature_fusion(shared, x)
        # B, 128, 60, 80
        x = self.aggregate_features(x, transforms, adjacency_matrix)
        # B, 7, 60, 80
        sseg = self.classifier(x)
        # B, 7, 480, 640
        sseg = F.interpolate(sseg, self.output_size, mode='bilinear', align_corners=True)
        # B, 1, 60, 80
        mask = self.mask_prediction(x)
        # B, 1, 480, 640
        mask = F.interpolate(mask, self.output_size, mode='bilinear', align_corners=True)
        return sseg, mask

    def aggregate_features(self, x, transforms, adjacency_matrix):
        # calculating constants
        agent_count = transforms.shape[0]
        cf_h, cf_w = 60, 80 # x.shape[2], x.shape[3]
        ppm = 3.2 # ((cf_h / self.sem_cfg.cloud_x_span) + (cf_w / self.sem_cfg.cloud_y_span)) / 2.0
        center_x = 48 # int((self.sem_cfg.cloud_max_x / self.sem_cfg.cloud_x_span) * cf_h)
        center_y = 40 # int((self.sem_cfg.cloud_max_y / self.sem_cfg.cloud_y_span) * cf_w)
        # aggregating [A, 128, 238, 318]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, ppm, cf_h, cf_w, center_x, center_y).to(transforms.device)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(cf_h, cf_w), flags='bilinear')
            warped_features[outside_fov] = 0
            aggregated_features[i, ...] = warped_features.sum(dim=0) / adjacency_matrix[i].sum()
        return aggregated_features
    
    def parameter_count(self):
        """
        returns the number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MCNN4(torch.nn.Module):
    def __init__(self, input_channel, num_classes, output_size, sem_cfg: SemanticCloudConfig):
        super().__init__()
        self.learning_to_downsample = LearningToDownsample(input_channel)
        self.semantic_global_feature_extractor = GlobalFeatureExtractor()
        self.semantic_feature_fusion = FeatureFusionModule()
        self.mask_global_feature_extractor = GlobalFeatureExtractor()
        self.mask_feature_fusion = FeatureFusionModule()
        self.classifier = Classifier(num_classes)
        self.maskifier = Classifier(1)
        self.output_size = output_size
        self.sem_cfg = sem_cfg

    def forward(self, x, transforms, adjacency_matrix):
        # B, 3, 480, 640: input size
        # B, 64, 60, 80
        shared = self.learning_to_downsample(x)
        
        # ------------mask branch------------
        # B, 128, 15, 20
        x_semantic = self.semantic_global_feature_extractor(shared)
        # B, 128, 60, 80
        x_semantic = self.semantic_feature_fusion(shared, x_semantic)
        # ----------semantic branch----------
        # B, 128, 15, 20
        x_mask = self.mask_global_feature_extractor(shared)
        # B, 128, 60, 80
        x_mask = self.mask_feature_fusion(shared, x_mask)
        # --latent masking into aggregation--
        # B, 128, 60, 80
        x_semantic = self.aggregate_features(detached_normalized(x_mask) * x_semantic, transforms, adjacency_matrix)
        # B, 7, 60, 80
        x_semantic = self.classifier(x_semantic)
        # B, 7, 480, 640
        sseg = F.interpolate(x_semantic, self.output_size, mode='bilinear', align_corners=True)
        # B, 1, 60, 80
        mask = self.maskifier(x_mask)
        # B, 1, 480, 640
        mask = torch.sigmoid(F.interpolate(mask, self.output_size, mode='bilinear', align_corners=True))
        return sseg, mask

    def aggregate_features(self, x, transforms, adjacency_matrix):
        # calculating constants
        agent_count = transforms.shape[0]
        cf_h, cf_w = 60, 80 # x.shape[2], x.shape[3]
        ppm = 3.2 # ((cf_h / self.sem_cfg.cloud_x_span) + (cf_w / self.sem_cfg.cloud_y_span)) / 2.0
        center_x = 48 # int((self.sem_cfg.cloud_max_x / self.sem_cfg.cloud_x_span) * cf_h)
        center_y = 40 # int((self.sem_cfg.cloud_max_y / self.sem_cfg.cloud_y_span) * cf_w)
        # aggregating [A, 128, 238, 318]
        aggregated_features = torch.zeros_like(x)
        for i in range(agent_count):
            outside_fov = torch.where(adjacency_matrix[i] == 0)[0]
            relative_tfs = get_single_relative_img_transform(transforms, i, ppm, cf_h, cf_w, center_x, center_y).to(transforms.device)
            warped_features = kornia.warp_affine(x, relative_tfs, dsize=(cf_h, cf_w), flags='bilinear')
            warped_features[outside_fov] = 0
            aggregated_features[i, ...] = warped_features.sum(dim=0) / adjacency_matrix[i].sum()
        return aggregated_features
    
    def parameter_count(self):
        """
        returns the number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def detached_normalized(tensor: torch.Tensor):
    """
    tight sigmoid shifted forward
    """
    return 1 / (1 + torch.exp(-5.0 * (tensor - 0.5)))

class LearningToDownsample(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=32, stride=2)
        self.sconv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 48, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.sconv2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, dilation=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.sconv1(x)
        x = self.sconv2(x)
        return x

class GlobalFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_block = nn.Sequential(InvertedResidual(64, 64, 2, 6),
                                         InvertedResidual(64, 64, 1, 6),
                                         InvertedResidual(64, 64, 1, 6))
        self.second_block = nn.Sequential(InvertedResidual(64, 96, 2, 6),
                                          InvertedResidual(96, 96, 1, 6),
                                          InvertedResidual(96, 96, 1, 6))
        self.third_block = nn.Sequential(InvertedResidual(96, 128, 1, 6),
                                         InvertedResidual(128, 128, 1, 6),
                                         InvertedResidual(128, 128, 1, 6))
        self.ppm = PSPModule(128, 128)

    def forward(self, x):
        x = self.first_block(x)
        x = self.second_block(x)
        x = self.third_block(x)
        x = self.ppm(x)
        return x

# Modified from https://github.com/tonylins/pytorch-mobilenet-v2
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Modified from https://github.com/Lextal/pspnet-pytorch/blob/master/pspnet.py
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h,w), mode='bilinear',
                                align_corners=True) for stage in self.stages] + [feats]
        # import pdb;pdb.set_trace()
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class FeatureFusionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sconv1 = ConvBlock(in_channels=128, out_channels=128, stride=1, dilation=1, groups=128)
        self.conv_low_res = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv_high_res = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, high_res_input, low_res_input):
        low_res_input = F.interpolate(input=low_res_input, scale_factor=4, mode='bilinear', align_corners=True)
        low_res_input = self.sconv1(low_res_input)
        low_res_input = self.conv_low_res(low_res_input)

        high_res_input = self.conv_high_res(high_res_input)
        x = torch.add(high_res_input, low_res_input)
        return self.relu(x)

class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.sconv1 = ConvBlock(in_channels=128, out_channels=128, stride=1, dilation=1, groups=128)
        self.sconv2 = ConvBlock(in_channels=128, out_channels=128, stride=1, dilation=1, groups=128)
        self.conv = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.sconv1(x)
        x = self.sconv1(x)
        return self.conv(x)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

if __name__ == '__main__':
    model = MCNN(input_channel=3, num_classes=10)
    x = torch.rand(2, 3, 256, 256)
    y, z = model(x)