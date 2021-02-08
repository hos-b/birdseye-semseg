import torch

"""
input --> downsample --> bottleneck --------
            |                               |--> aggregation layer ---> total mask x total prediction [loss #2]
            |                                           ^
             --> mask prediction [loss #1] -------------|
                                                        |
transform ----------------------------------------------
"""

class MassCNN(torch.nn.Module):
    def __init__(self):
        super(MassCNN, self).__init__()
        self.downsample = None
        self.bottleneck = None
        self.mask_prediction = None
        self.aggregation = None
    
    def forward(self, x):
        return x