import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pyrocc.transformer import DenseTransformer

class TransformerPyramid(nn.Module):

    def __init__(self, in_channels, channels, resolution, extents, ymin, ymax,
                 focal_length):
        super().__init__()
        self.transformers = nn.ModuleList()
        print('transformer extents:')
        for i in range(3):
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3)
            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3])
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]
            print(f'x: [{extents[0]}, {extents[2]}], z: [{zmin:.2f}, {zmax:.2f}]')
            # Build transformers
            tfm = DenseTransformer(in_channels, channels, resolution,
                                   subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)

    def forward(self, feature_maps, calib, oov_depth):
        bev_feats = list()
        # use the bottom 3 pyramid layers for in view regions
        for i, transformer in enumerate(self.transformers):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2 ** i
            calib_downsamp = calib.clone()
            calib_downsamp[:2] = calib[:2] / scale
            # Apply orthographic transformation to each feature map separately
            bev_feats.append(transformer(feature_maps[i], calib_downsamp))
        B, C, _, W = bev_feats[0].shape
        bev_feats.append(torch.zeros((B, C, oov_depth, W),
                                     dtype=torch.float32,
                                     device=calib.device))
        # combine birds-eye-view & oov feature maps along the depth axis
        return torch.cat(bev_feats[::-1], dim=-2).flip([2])
