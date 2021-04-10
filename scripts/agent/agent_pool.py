import torch
import kornia

from data.mask_warp import get_single_relative_img_transform, get_all_aggregate_masks

class CurriculumPool:
    """
    controls the difficulity of the segmentation task by limiting the
    extents of message passing between the agents. for each connected 
    graph, the adjacency matrix is calculated and used to control the
    information flow between agents.
    as the training progresses, the number of agents that are allowed
    to propagate messages increases, starting with 1.
    """
    def __init__(self, starting_difficulty, maximum_difficulty, maximum_agent_count, device):
        self.device = device
        self.agent_count = 0
        # connection strategy
        self.difficulty = starting_difficulty
        self.maximum_difficulty = maximum_difficulty
        self.combined_masks = None
        self.adjacency_matrix = None
        self.max_agent_count = maximum_agent_count

    def generate_connection_strategy(self, ids, masks, transforms, pixels_per_meter, h, w, center_x, center_y):
        """
        combines all masks, find the best masks for each agent &
        create adjacency matrix based on current difficulty
        """
        self.agent_count = masks.shape[0]
        # also no calculations necessary for difficulty = 1
        if self.difficulty ==  1:
            self.combined_masks = masks
            self.adjacency_matrix = torch.eye(self.agent_count, dtype=torch.bool, device=self.device)
            return
        # no calculations necessary for max possible difficulty (!= max_difficulty)
        elif self.difficulty == self.max_agent_count:
            self.combined_masks = get_all_aggregate_masks(masks, transforms, pixels_per_meter, h, w, center_x, center_y).long()
            self.adjacency_matrix = torch.ones((self.agent_count, self.agent_count), dtype=torch.bool, device=self.device)
            return
        # for other cases, need to do some stuff
        self.adjacency_matrix = torch.eye(self.agent_count, dtype=torch.bool)
        # identifying the masks
        for i in range(ids.shape[1]):
            masks[i] *= 1 << ids[0, i, 0].item() # the id tensors are fucked but squeeze() makes it worse
        self.combined_masks = get_all_aggregate_masks(masks, transforms, pixels_per_meter, h, w,
                                                      center_x, center_y, 'nearest').long()
        # using the unique values of the mask to find agent view overlaps
        for i in range(self.agent_count):
            possible_connections, counts = self.combined_masks[i].unique(sorted=False, return_counts=True)
            possible_connections = possible_connections.long().cpu().tolist()
            counts = counts.cpu().tolist()
            # get mask IDs sorted based on count
            possible_connections = [x for _, x in sorted(zip(counts, possible_connections), reverse=True)]
            try:
                # no one cares where mask is 0
                possible_connections.remove(0)
            except ValueError:
                pass
            # the ego-mask is always pre-selected
            accepted_connections = 1 << i
            accepted_connection_count = 1
            # while 
            while accepted_connection_count < self.difficulty and len(possible_connections) > 0:
                current_connection = possible_connections.pop(0)
                # if already accepted this connection earlier in another
                if (~accepted_connections & current_connection) == 0:
                    continue
                # considering the constituent mask elements
                for uniq_mask_id in decompose_binary_elements(current_connection):
                    # if agent has not been considered before
                    if (uniq_mask_id & ~accepted_connections):
                        self.adjacency_matrix[i, uniq_mask_id] = 1
                        accepted_connections |= uniq_mask_id
                        accepted_connection_count += 1
                        # not adding too many
                        if accepted_connection_count == self.difficulty:
                            break
            # combining the mask of selected connections and setting everything else to 0
            self.combined_masks[i] = self.combined_masks[i] & accepted_connections
            self.combined_masks[i][self.combined_masks[i] != 0] = 1

        self.adjacency_matrix.to(self.device)

def decompose_binary_elements(mask_value) -> list:
    """
    as a part of generating a connection strategy, the mask value that has had
    the highest overlap with the current agent is decomposed into its elements
    Example: mask_value = 2 + 16 + 32 + 64 -> [1, 4, 5, 6]
    """
    elements = []
    shifts = 0
    max_mask_value = 1 << shifts
    while mask_value >= max_mask_value:
        if mask_value & max_mask_value:
            elements.append(shifts)
        shifts += 1
        max_mask_value = 1 << shifts
    return elements
