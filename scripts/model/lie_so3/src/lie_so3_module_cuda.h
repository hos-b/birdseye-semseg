int lie_so3_forward_cuda(THCudaTensor * r_param, THCudaTensor * output);
int lie_so3_backward_cuda(THCudaTensor * r_param, THCudaTensor * output, THCudaTensor * grad_output, THCudaTensor * grad_r_param);
