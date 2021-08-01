from modules.lie_so3.lie_so3_f import LieSO3Function

import torch

class LieSO3(torch.nn.Module):
    def __init__(self):
        super(LieSO3, self).__init__()

    def forward(self, r_param):
        return LieSO3Function.apply(r_param)