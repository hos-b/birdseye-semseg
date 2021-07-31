#ifndef _LIE_SO3_KERNEL
#define _LIE_SO3_KERNEL

#include <stdio.h>
#include <math.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif


struct float3x3{
    float x11, x12, x13,
          x21, x22, x23,
          x31, x32, x33;
};

struct float3x9{
    float x11, x12, x13, x14, x15, x16, x17, x18, x19,
          x21, x22, x23, x24, x25, x26, x27, x28, x29,
          x31, x32, x33, x34, x35, x36, x37, x38, x39;
};

struct floatvec3{
    float x1, x2, x3;
};

void lie_so3(const float * r_param, const int batch_size, float * output, cudaStream_t stream);
void lie_so3_backward(const float * r_param, const float * output, const int batch_size, const float * grad_output, float * grad_r_param, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

