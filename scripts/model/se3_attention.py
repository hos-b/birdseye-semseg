import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F

def get_distance_attention(transforms: torch.Tensor, max_attention = 2.0):
    """
    returns a B x B attention matrix based on euclidean
    distance. max attention will be 2 and min will be 1.
    """
    agent_count = transforms.shape[0]
    distance_attention = torch.zeros((agent_count, agent_count),
                                     dtype=torch.float32,
                                     device=transforms.device)
    if agent_count == 1:
        distance_attention[0, 0] = max_attention
        return distance_attention
    for i in range(agent_count):
        delta_dist = transforms[:, :3, 3] - transforms[i, :3, 3]
        distance_attention[i] = torch.sqrt((delta_dist * delta_dist).sum(dim = 1))
        distance_attention[i] = max_attention - (distance_attention[i] / distance_attention[i].max())
    return distance_attention

class FlatSE3AttentionHead(nn.Module):
    """
    self-attention for transform matrices. x & y rotation
    alongside translation are considered for calculating
    the attention matrices
    input:  B x 4 x 4
    output: B x 9
    """
    def __init__(self):
        super(FlatSE3AttentionHead, self).__init__()
        self.q_rotx = nn.Linear(3, 3)
        self.k_rotx = nn.Linear(3, 3)
        self.v_rotx = nn.Linear(3, 3)
        self.q_roty = nn.Linear(3, 3)
        self.k_roty = nn.Linear(3, 3)
        self.v_roty = nn.Linear(3, 3)
        self.v_trans = nn.Linear(3, 3)

    def forward(self, transforms: torch.Tensor):
        # input shape: B x 4 x 4
        agent_count = transforms.shape[0]
        # x-rot attention
        xr = transforms[:, :3, 0].float()
        xr_q = self.q_rotx(xr)
        xr_v = self.v_rotx(xr)
        xr_k = self.k_rotx(xr)
        xr_attention = xr_q @ xr_k.transpose(0, 1)
        soft_xr_attention = torch.sigmoid(xr_attention / np.sqrt(3), dim=-1)
        # y-rot attention
        yr = transforms[:, :3, 0].float()
        yr_q = self.q_roty(yr)
        yr_v = self.v_roty(yr)
        yr_k = self.k_roty(yr)
        yr_attention = yr_q @ yr_k.transpose(0, 1)
        soft_yr_attention = torch.sigmoid(yr_attention / np.sqrt(3), dim=-1)
        # translation attention
        tr_v = self.v_trans(transforms[:, :3, 3].float())
        tr_attention = get_distance_attention(transforms)
        # batch_size x 9
        return torch.cat([tr_attention @ tr_v, soft_yr_attention @ yr_v, soft_xr_attention @ xr_v], dim=-1)

class SE3MultiHeadAttention(nn.Module):
    """
    """
    def __init__(self, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([FlatSE3AttentionHead() for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * 9, 9)

    def forward(self, transforms: torch.Tensor):
        return self.linear(torch.cat([h(transforms) for h in self.heads], dim=-1))

class ResidualNorm(nn.Module):
    def __init__(self, sublayer: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(9)
        self.dropout = nn.Dropout(dropout)

    def forward(self, transforms: torch.Tensor):
        input_v = torch.cat([transforms[:, :3, 0],
                             transforms[:, :3, 1],
                             transforms[:, :3, 3]],
                             dim=-1).float()
        return self.norm(input_v + self.dropout(self.sublayer(transforms)))

class SE3Transfomer(nn.Module):
    def __init__(self, num_heads=6, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.attention = ResidualNorm(
            SE3MultiHeadAttention(num_heads),
            dropout=dropout,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(9, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, transforms: torch.Tensor):
        return torch.sigmoid(self.feed_forward(self.attention(transforms)))