import torch
import kornia
from torch._C import dtype

from model.mass_cnn import MassCNN
from data.mask_warp import get_single_relative_img_transform, get_all_aggregate_masks

class CurriculumPool:
    """
    creates a pool of agents that distribute the computation of
    the full architecture by simulating message passing. as the
    training progresses, the number of agents that are allowed
    to propagate messages increases, starting with 1.
    """
    def __init__(self, model: MassCNN, device: torch.device, output_size,
                 starting_difficulty, maximum_difficulty, maximum_agent_count) -> None:
        self.model = model
        self.agent_count = 0
        self.device = device
        self.detached_features = None
        self.compressed_features = None
        self.mask_predictions = None
        self.output_size = output_size
    
        # connection strategy
        self.difficulty = starting_difficulty
        self.maximum_difficulty = maximum_difficulty
        self.combined_masks = None
        self.adjacency_matrix = None
        self.max_agent_count = maximum_agent_count
                
    def calculate_detached_messages(self, rgbs):
        self.agent_count = rgbs.shape[0]
        # ~100 MB per agent
        with torch.no_grad():
            # [A, 64, 241, 321]
            hi_res_features = self.model.downsample(rgbs)
            # [A, 96, 239, 319]
            latent_compressed_features = self.model.compression_l1(hi_res_features)
            # [A, 96, 239, 319]
            latent_mask_predictions = self.model.mask_prediction_l1(hi_res_features)
            latent_mask_predictions = torch.minimum(latent_mask_predictions,
                                      torch.ones_like(latent_mask_predictions))
            # [A, 128, 238, 318]
            self.detached_features = self.model.compression_l2(
                    latent_compressed_features * latent_mask_predictions
            )

    def calculate_agent_mask(self, rgb):
        hi_res_features = self.model.downsample(rgb.unsqueeze(0))
        # [1 96, 239, 319]
        latent_compressed_features = self.model.compression_l1(hi_res_features)
        # [1, 96, 239, 319]
        latent_mask_prediction = self.model.mask_prediction_l1(hi_res_features)
        latent_mask_prediction = torch.minimum(latent_mask_prediction,
                                 torch.ones_like(latent_mask_prediction))
        # [1, 128, 238, 318]
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
        relative_tfs = get_single_relative_img_transform(transforms, agent_idx, ppm, cf_h, cf_w, center_x, center_y).to(self.device)
        warped_features = kornia.warp_affine(self.detached_features, relative_tfs, dsize=(cf_h, cf_w), flags='bilinear')
        # the same features but with gradient
        warped_features[agent_idx] = self.agent_features
        aggregated_features = warped_features.sum(dim=0) / self.agent_count
        # [A, 128, 256, 205]
        pooled_features = self.model.pyramid_pooling(aggregated_features.unsqueeze(0), self.output_size)
        return self.model.classifier(pooled_features)

    def generate_connection_strategy(self, ids, masks, transforms, pixels_per_meter, h, w, center_x, center_y):
        """
        combines all masks, find the best masks for each agent &
        create adjacency matrix based on current difficulty
        """
        self.agent_count = masks.shape[0]
        # also no calculations necessary for difficulty = 1
        if self.difficulty ==  1:
            self.combined_masks = masks
            self.adjacency_matrix = torch.eye(self.agent_count, dtype=torch.bool)
            return
        # no calculations necessary for max possible difficulty (!= max_difficulty)
        elif self.difficulty == self.max_agent_count:
            self.combined_masks = get_all_aggregate_masks(masks, transforms, pixels_per_meter, h, w, center_x, center_y).long()
            self.adjacency_matrix = torch.ones((self.agent_count, self.agent_count), dtype=torch.bool)
            return
        # for other cases, need to do some stuff
        self.adjacency_matrix = torch.eye(self.agent_count, dtype=torch.bool)
        ids = ids.squeeze()
        # identifying the masks
        for i in range(len(ids)):
            masks[i] *= 1 << ids[i].item()
        self.combined_masks = get_all_aggregate_masks(masks, transforms, pixels_per_meter, h, w,
                                                      center_x, center_y, 'nearest').long()
        # using the unique values of the mask to find agent view overlaps
        for i in range(self.agent_count):
            possible_connections, counts = self.combined_masks[i].unique(sorted=False, return_counts=True)
            possible_connections = possible_connections.long().cpu().tolist()
            counts = counts.cpu().tolist()
            # get mask IDs sorted based on count
            possible_connections = [x for _, x in sorted(zip(counts, possible_connections), reverse=True)]
            possible_connections.remove(0)
            # the ego-mask is always pre-selected
            accepted_connections = 1 << i
            accepted_connection_count = 1
            # while 
            while accepted_connection_count < self.difficulty or len(possible_connections) > 0:
                current_connection = possible_connections.pop(0)
                # if already accepted this connection earlier in another
                if (~accepted_connections & current_connection) == 0:
                    continue
                # considering the constituent mask elements
                for agent_idx in decompose_binary_elements(current_connection):
                    # if agent has not been considered before
                    if (agent_idx & ~accepted_connections):
                        self.adjacency_matrix[i, agent_idx] = 1
                        accepted_connections |= agent_idx
                        accepted_connection_count += 1
                        # not adding too many
                        if accepted_connection_count == self.difficulty:
                            break
            # combining the mask of selected connections and setting everything else to 0
            self.combined_masks[i] = self.combined_masks[i] & accepted_connections
            self.combined_masks[i][self.combined_masks[i] != 0] = 1

def decompose_binary_elements(mask_value) -> list:
    """
    as a part of generating a connection strategy, the mask value that has had
    the highest overlap with the current agent is decomposed into its elements
    Example: mask_value = 2 + 16 + 32 + 64 -> [1, 4, 5, 6]
    """
    elements = []
    max_mask_value = 1
    while mask_value >= max_mask_value:
        if mask_value & max_mask_value:
            elements.append(max_mask_value)
        max_mask_value = max_mask_value << 1
    return elements