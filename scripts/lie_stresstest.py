import torch
from model.modules.lie_so3.lie_so3_m import LieSO3
from torch.autograd import Variable

liet_cuda = LieSO3()
device = torch.device('cuda')

print("starting LieSO3 stress test...")
test_len = 10000000

for i in range(test_len):
    print(f'{i}/{test_len}', end='\r')
    params = torch.rand((1, 3), device=device, dtype=torch.float32) * 0.1
    params_gpu = Variable(params.cuda(), requires_grad = True)
    out_cuda = liet_cuda(params_gpu)

