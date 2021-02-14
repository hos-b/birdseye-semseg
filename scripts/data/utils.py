import torch

def drop_agent_data(rgbs, labels, masks, transforms, drop_probability):
    """
    simulate connection drops between cars or non transmitting cars
    input:
        - rgbs:         batch_size x agent_count x 3 x H x W
        - labels:       batch_size x agent_count x H x W
        - masks:        batch_size x agent_count x H x W
        - transforms:   batch_size x agent_count x 16 x 16
    """
    drop_probs = torch.ones((rgbs.shape[1], ), dtype=torch.float32) * drop_probability
    drops = torch.bernoulli(drop_probs).long()
    return rgbs[0, drops != 1, ...], labels[0, drops != 1, ...], \
           masks[0, drops != 1, ...], transforms[0, drops != 1, ...]

def squeeze_all(rgbs, labels, masks, transforms):
    return rgbs.squeeze(), labels.squeeze(), masks.squeeze(), transforms.squeeze()