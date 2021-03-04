import torch
import kornia

from model.mass_cnn import MassCNN
from data.mask_warp import get_relative_img_transform

class AgentPool:
    """
    creates a pool of agents that distribute the computation of
    the full architecture by simulating message passing.
    """
    def __init__(self, model: MassCNN, device: torch.device, output_size) -> None:
        self.model = model
        self.agent_count = 0
        self.device = device
        self.detached_features = None
        self.compressed_features = None
        self.mask_predictions = None
        self.output_size = output_size
    
    def calculate_detached_messages(self, rgbs):
        self.agent_count = rgbs.shape[0]
        # ~100 MB per agent
        with torch.no_grad():
            # [A, 64, 241, 321]
            hi_res_features = self.model.downsample(rgbs)
            # [A, 96, 239, 319]
            latent_compressed_features = self.model.compression_l1(hi_res_features)
            # [A, 96, 239, 319]
            latent_mask_prediction = self.model.mask_prediction_l1(hi_res_features)
            # [A, 128, 238, 318]
            self.detached_features = self.model.compression_l2(
                    latent_compressed_features * latent_mask_prediction
            )

        del latent_mask_prediction
        del latent_compressed_features
        del hi_res_features
        # 2 > 1!
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    def calculate_agent_mask(self, rgb):
        hi_res_features = self.model.downsample(rgb.unsqueeze(0))
        # [A, 96, 239, 319]
        latent_compressed_features = self.model.compression_l1(hi_res_features)
        # [A, 96, 239, 319]
        latent_mask_prediction = self.model.mask_prediction_l1(hi_res_features)
        # [A, 128, 238, 318]
        self.agent_features = self.model.compression_l2(
            latent_compressed_features * latent_mask_prediction
        )
        return self.model.mask_prediction_l2(latent_mask_prediction)


    def aggregate_messages(self, agent_idx, transforms):
        cf_h, cf_w = 238, 318 # compressed_features.shape[2], compressed_features.shape[3]
        ppm = 12.71 # ((cf_h / self.cfg.cloud_x_span) + (cf_w / self.cfg.cloud_y_span)) / 2.0
        center_x = 190 # int((self.cfg.cloud_max_x / self.cfg.cloud_x_span) * cf_h)
        center_y = 159 # int((self.cfg.cloud_max_y / self.cfg.cloud_y_span) * cf_w)
        # aggregating [A, 128, 238, 318]
        aggregated_features = torch.zeros_like(self.detached_features)
        compressed_features = self.detached_features.clone()
        compressed_features[agent_idx] = self.agent_features
        
        relative_tfs = get_relative_img_transform(transforms, agent_idx, ppm, cf_h, cf_w, center_x, center_y).to(self.device)
        warped_features = kornia.warp_affine(compressed_features, relative_tfs, dsize=(cf_h, cf_w), flags='bilinear')
        aggregated_features = warped_features.sum(dim=0) / self.agent_count
        # [A, 128, 256, 205]
        pooled_features = self.model.pyramid_pooling(aggregated_features.unsqueeze(0), self.output_size)
        return self.model.classifier(pooled_features)
