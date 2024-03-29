import torch

from data.mask_warp import get_all_aggregate_masks

class CurriculumPool:
    """
    controls the difficulity of the segmentation task by limiting the
    extents of message passing between the agents. for each connected 
    graph, the adjacency matrix is calculated and used to control the
    information flow between agents.
    as the training progresses, the number of agents that are allowed
    to propagate messages increases, starting with 1.
    """
    def __init__(self, starting_difficulty, maximum_difficulty, maximum_agent_count, enforce_adj_calc, device):
        self.device = device
        self.agent_count = 0
        # connection strategy
        self.difficulty = starting_difficulty
        self.maximum_difficulty = maximum_difficulty
        self.combined_masks = None
        self.adjacency_matrix = None
        self.max_agent_count = maximum_agent_count
        self.enforce_adj_calc = enforce_adj_calc

    def generate_connection_strategy(self, masks, transforms, pixels_per_meter, h, w, center_x, center_y):
        """
        combines all masks, find the best masks for each agent &
        create adjacency matrix based on current difficulty
        """
        self.agent_count = masks.shape[0]
        # no calculations necessary for difficulty = 1 -----------------------------------------------------
        if self.difficulty ==  1:
            self.combined_masks = masks.clone()
            self.adjacency_matrix = torch.eye(self.agent_count, dtype=torch.bool, device=self.device)
            return
        # --------------------------------------------------------------------------------------------------
        elif self.difficulty == self.max_agent_count:
            # calculate adjacency matrix for maximum difficulty
            if self.enforce_adj_calc:
                new_masks = masks.clone()
                for i in range(self.agent_count):
                    new_masks[i] *= 1 << i
                self.combined_masks = get_all_aggregate_masks(new_masks, transforms, pixels_per_meter, h, w,
                                                              center_x, center_y, 'nearest', False).long()
                self.adjacency_matrix = torch.eye(self.agent_count, dtype=torch.bool, device=self.device)
                for i in range(self.agent_count):
                    possible_connections = self.combined_masks[i].unique(
                        sorted=False, 
                        return_counts=False
                    ).long().cpu().tolist()
                    try:
                        # no one cares where mask is 0
                        possible_connections.remove(0)
                        # or if the agent view overlaps itself
                        possible_connections.remove(1 << i)
                    except ValueError:
                        pass
                    while len(possible_connections) > 0:
                        current_connection = possible_connections.pop(0)
                        # consider the constituent mask elements
                        for agent_id in decompose_binary_elements(current_connection):
                            self.adjacency_matrix[i, agent_id] = 1
                # recalculate the merged masks
                self.combined_masks = get_all_aggregate_masks(masks, transforms, pixels_per_meter, h, w,
                                                              center_x, center_y, merge_masks=True).long()
                # make sure adjacecy matrix is symmetric
                self.adjacency_matrix = self.adjacency_matrix.transpose(0, 1) & self.adjacency_matrix
            # assume all agents are connected
            else:
                self.combined_masks = get_all_aggregate_masks(masks, transforms, pixels_per_meter, h, w,
                                                              center_x, center_y, merge_masks=True).long()
                self.adjacency_matrix = torch.ones((self.agent_count, self.agent_count),
                                                    dtype=torch.bool, device=self.device)
            return
        # --------------------------------------------------------------------------------------------------
        # for other cases, calculate adjacency matrix based on current difficulty
        self.adjacency_matrix = torch.eye(self.agent_count, dtype=torch.bool, device=self.device)
        # identifying the masks (giving them ids)
        new_masks = masks.clone()
        for i in range(self.agent_count):
            new_masks[i] *= 1 << i
        self.combined_masks = get_all_aggregate_masks(new_masks, transforms, pixels_per_meter, h, w,
                                                      center_x, center_y, 'nearest', False).long()
        # using the unique ids of the masks to find biggest 
        for i in range(self.agent_count):
            possible_connections, counts = self.combined_masks[i].unique(sorted=False, return_counts=True)
            possible_connections = possible_connections.long().cpu().tolist()
            counts = counts.cpu().tolist()
            # get mask IDs sorted based on count
            possible_connections = [x for _, x in sorted(zip(counts, possible_connections), reverse=True)]
            try:
                # no one cares where mask is 0
                possible_connections.remove(0)
                # or if the agent view overlaps itself
                possible_connections.remove(1 << i)
            except ValueError:
                pass
            # the ego-mask is always pre-selected
            accepted_connections = 1 << i
            accepted_connection_count = 1
            while accepted_connection_count < self.difficulty and len(possible_connections) > 0:
                current_connection = possible_connections.pop(0)
                # if already accepted this connection earlier
                if (~accepted_connections & current_connection) == 0:
                    continue
                # considering the constituent mask elements
                for agent_id in decompose_binary_elements(current_connection):
                    uniq_mask_id = 1 << agent_id
                    # if agent has not been considered before
                    if (uniq_mask_id & ~accepted_connections) != 0:
                        self.adjacency_matrix[i, agent_id] = 1
                        accepted_connections |= uniq_mask_id
                        accepted_connection_count += 1
                        # not adding too many
                        if accepted_connection_count == self.difficulty:
                            break
            # combining the mask of selected connections and setting everything else to 0
            self.combined_masks[i] = self.combined_masks[i] & accepted_connections
            self.combined_masks[i][self.combined_masks[i] != 0] = 1

        # make sure adjacecy matrix is symmetric
        self.adjacency_matrix = self.adjacency_matrix.transpose(0, 1) & self.adjacency_matrix

def decompose_binary_elements(mask_value) -> list:
    """
    as a part of generating a connection strategy, the mask value that has had
    the highest overlap with the current agent is decomposed into its elements
    Example: mask_value = 1 + 2 + 16 + 32 + 64 -> [0, 1, 4, 5, 6]
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
