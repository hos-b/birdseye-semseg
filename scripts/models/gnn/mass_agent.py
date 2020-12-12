import torch
import kornia

#pylint: disable=E1101
#pylint: disable=not-callable 


class MASSAgent(torch.nn.Module):
    def __init__(self, agent_id, detection_thresh, bev_rows, bev_cols):
        super().__init__()
        self.agent_id = agent_id
        self.detection_threshold = detection_thresh
        self.mask = torch.zeros(bev_rows, bev_cols)

    def forward(self, agent_ids, target_bev, masks, transforms):
        for agent_id in agent_ids:
            # if the mask belong to the current agent, apply directly
            if agent_id == self.agent_id:
                self.mask += masks[agent_id]
            else:
                pass