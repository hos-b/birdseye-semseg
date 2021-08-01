import torch
from torch.autograd import Function
import sys
sys.path.append("modules/lie_so3/dist/lie_so3-0.0.0-py3.8-linux-x86_64.egg")
import lie_so3


class LieSO3Function(Function):
    def __init__(self):
        super(LieSO3Function, self).__init__()

    @staticmethod
    def forward(self, r_param):
        self.r_param = r_param
        self.batch_size = r_param.size(0)

        output = torch.zeros((self.batch_size, 1, 3, 3)).cuda()

        lie_so3.forward(r_param, output)

        self.output = output

        return output

    @staticmethod
    def backward(self, grad_output):
        r_param = self.r_param
        output = self.output

        grad_r_param = torch.zeros((self.batch_size, 1, 1, 3)).cuda()

        lie_so3.backward(r_param, output, grad_output, grad_r_param)

        return grad_r_param
