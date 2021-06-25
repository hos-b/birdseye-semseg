import torch.nn as nn

from model.pyrocc.fpn import FPN50
from model.pyrocc.pyramid import TransformerPyramid
from model.pyrocc.topdown import TopdownNetwork
from model.pyrocc.classifier import LinearClassifier, BayesianClassifier

from operator import mul
from functools import reduce

class PyramidOccupancyNetwork(nn.Module):
    def __init__(self, map_resolution = 0.25, tfm_channels = 64, bayesian_classifer = False,
                 map_extents = [-25.0, 1.0, 25.0, 50.0], ymin = -2, ymax = 4,
                 focal_length = 630, topdown_channels = 128,
                 topdown_strides = [1, 2], topdown_layers = [4, 4],
                 topdown_blocktype = 'blocktype', num_classes = 7, prior = None):
        super().__init__()

        # Build frontend
        frontend = FPN50()

        # Build transformer pyramid
        # 0.5 = 0.25 * mul([1, 2])
        tfm_resolution = map_resolution * reduce(mul, topdown_strides)
        transformer = TransformerPyramid(256, tfm_channels, tfm_resolution,
                                        map_extents, ymin, 
                                        ymax, focal_length)

        # Build topdown network
        topdown = TopdownNetwork(tfm_channels, topdown_channels,
                                topdown_layers, topdown_strides,
                                topdown_blocktype)
        
        # Build classifier
        if bayesian_classifer:
            classifier = BayesianClassifier(topdown.out_channels, num_classes)
        else:
            classifier = LinearClassifier(topdown.out_channels, num_classes)
        classifier.initialise(prior)
    

    def forward(self, image, calib, *args):
        # image: [B, 3, 600, 800]
        # calib: [B, 3, 3]

        # Extract multiscale feature maps
        # 0: [B, 256, 75, 100]
        # 1: [B, 256, 38,  50]
        # 2: [B, 256, 19,  25]
        # 3: [B, 256, 10,  13]
        # 4: [B, 256,  5,   7]
        feature_maps = self.frontend(image)

        # Transform image features to birds-eye-view
        # [B, 64, 98, 100]
        bev_feats = self.transformer(feature_maps, calib)

        # Apply topdown network
        # [B, 256, 196, 200]
        td_feats = self.topdown(bev_feats)

        # Predict individual class log-probabilities
        # [B, class_count, 196, 200]
        logits = self.classifier(td_feats)
        return logits