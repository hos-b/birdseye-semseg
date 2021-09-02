#include <THC/THC.h>
#include <ATen/ATen.h>
#include <math.h>
#include <torch/torch.h>
#include "torch/extension.h"
#include "lie_so3.h"

extern THCState *state;
int lie_so3_forward_cuda(at::Tensor r_param, at::Tensor output)
{
    // Grab the input tensor
    const float * r_param_flat = r_param.data<float>();
    const int batch_size = r_param.size(0);

    float * output_flat = output.data<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    lie_so3(r_param_flat, batch_size, output_flat, stream);

    return 1;
}

int lie_so3_backward_cuda(at::Tensor r_param, at::Tensor output, at::Tensor grad_output, at::Tensor grad_r_param)
{
    const  float * r_param_flat = r_param.data<float>();
    const  float * grad_output_flat = grad_output.data<float>();
    const  float * output_flat = output.data<float>();
    const int batch_size = r_param.size(0);

    float * grad_r_param_flat = grad_r_param.data<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    lie_so3_backward(r_param_flat, output_flat, batch_size, grad_output_flat, grad_r_param_flat, stream);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lie_so3_forward_cuda, "lie so3 forward (CUDA)");
  m.def("backward", &lie_so3_backward_cuda, "lie so3 variation backward (CUDA)");
}