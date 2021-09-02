import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
from lie_so3_m import LieSO3

liet_cuda = LieSO3()
params = torch.rand((1, 3)).float()
params[:, 0] = 0.0
params[:, 1] = 0.0
params[:, 2] = -0.1943
print(params)
params = params * 0.1

print("Start testing...")

params_gpu = Variable(params.cuda(), requires_grad = True)
out_cuda = liet_cuda(params_gpu)
import pdb; pdb.set_trace()
if gradcheck(LieSO3(), (params_gpu), eps=1e-3, atol=1e-1, rtol=1e-1):
    print("Gradients tested successfully !!")
else:
    print("Gradients failed")
