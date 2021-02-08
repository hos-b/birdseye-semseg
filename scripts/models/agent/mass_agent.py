import torch
import kornia

from models.mass_cnn import MassCNN
#pylint: disable=E1101
#pylint: disable=not-callable 


class AgentPool(torch.nn.Module):
    def __init__(self, agent_count, connection_drop = 0.01):
        super(AgentPool).__init__()
        self.agent_count = agent_count
        self.drop_probs = torch.full((agent_count, ), connection_drop)
        self.mass_cnn = MassCNN()
    
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
        hi_res_feature = self.mass_cnn.downsample(rgbs)
        compressed_features = self.mass_cnn.bottleneck(hi_res_feature)
        predicted_masks = self.mass_cnn.mask_prediction(hi_res_feature)
        aggr_masks, aggr_preds = self.mass_cnn.aggregation(compressed_features,
                                                           predicted_masks,
                                                           transforms)
        # connection drops
        drops = torch.bernoulli(self.drop_probs).long()
        aggr_masks[drops == 1, :, :] = torch.zeros_like(aggr_masks[0])
        aggr_preds[drops == 1, :, :] = torch.zeros_like(aggr_preds[0])
        predicted_masks[drops == 1, :, :] = torch.zeros_like(predicted_masks[0])

        return predicted_masks, aggr_masks, aggr_preds
