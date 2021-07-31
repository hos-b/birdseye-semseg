#include "lie_so3.h"
#include <stdio.h>
#define BLOCK 1024

__device__  __host__
float3x3 copyMatrixToStruct(const float *mat_f){
    float3x3 mat;
    mat.x11 = mat_f[0];
    mat.x12 = mat_f[1];
    mat.x13 = mat_f[2];
    mat.x21 = mat_f[3];
    mat.x22 = mat_f[4];
    mat.x23 = mat_f[5];
    mat.x31 = mat_f[6];
    mat.x32 = mat_f[7];
    mat.x33 = mat_f[8];
    return mat;
}

__device__  __host__
floatvec3 copyVectorToStruct(const float *vec_f){
    floatvec3 vec;
    vec.x1 = vec_f[0];
    vec.x2 = vec_f[1];
    vec.x3 = vec_f[2];
    return vec;
}

__device__ __host__
inline float3x3 matMul(float3x3 x, float3x3 y){
    float3x3 out;
    out.x11 = x.x11 * y.x11 + x.x12 * y.x21 + x.x13 * y.x31;
    out.x12 = x.x11 * y.x12 + x.x12 * y.x22 + x.x13 * y.x32;
    out.x13 = x.x11 * y.x13 + x.x12 * y.x23 + x.x13 * y.x33;

    out.x21 = x.x21 * y.x11 + x.x22 * y.x21 + x.x23 * y.x31;
    out.x22 = x.x21 * y.x12 + x.x22 * y.x22 + x.x23 * y.x32;
    out.x23 = x.x21 * y.x13 + x.x22 * y.x23 + x.x23 * y.x33;

    out.x31 = x.x31 * y.x11 + x.x32 * y.x21 + x.x33 * y.x31;
    out.x32 = x.x31 * y.x12 + x.x32 * y.x22 + x.x33 * y.x32;
    out.x33 = x.x31 * y.x13 + x.x32 * y.x23 + x.x33 * y.x33;

    return out;
}

__device__ __host__
inline floatvec3 matvecMul(float3x3 x, floatvec3 y){
    floatvec3 out;
    out.x1 = x.x11 * y.x1 + x.x12 * y.x2 + x.x13 * y.x3;
    out.x2 = x.x21 * y.x1 + x.x22 * y.x2 + x.x23 * y.x3;
    out.x3 = x.x31 * y.x1 + x.x32 * y.x2 + x.x33 * y.x3;
    return out;
}

__device__ __host__
inline floatvec3 vectorAdd (floatvec3 x, floatvec3 y){
    floatvec3 out;
    out.x1 = x.x1 + y.x1;
    out.x2 = x.x2 + y.x2;
    out.x3 = x.x3 + y.x3;
    return out;
}

__device__ __host__
inline floatvec3 vectorSet(floatvec3 in_vec, const int dim, const float val){
    if(dim == 0){
        in_vec.x1 = val;
    }else if(dim == 1){
        in_vec.x2 = val;
    }else{
        in_vec.x3 = val;
    }
    return in_vec;
}

__device__ __host__
inline float vectorGet(floatvec3 in, const int dim){
    if(dim == 0){
        return in.x1;
    }else if(dim == 1){
        return in.x2;
    }else{
        return in.x3;
    }
}

__device__ __host__
inline float3x3 matAdd (float3x3 x, float3x3 y){
    float3x3 out;
    out.x11 = x.x11 + y.x11;
    out.x12 = x.x12 + y.x12;
    out.x13 = x.x13 + y.x13;

    out.x21 = x.x21 + y.x21;
    out.x22 = x.x22 + y.x22;
    out.x23 = x.x23 + y.x23;

    out.x31 = x.x31 + y.x31;
    out.x32 = x.x32 + y.x32;
    out.x33 = x.x33 + y.x33;

    return out;
}

__device__ __host__
inline float3x3 matSub (float3x3 x, float3x3 y){
    float3x3 out;
    out.x11 = x.x11 - y.x11;
    out.x12 = x.x12 - y.x12;
    out.x13 = x.x13 - y.x13;

    out.x21 = x.x21 - y.x21;
    out.x22 = x.x22 - y.x22;
    out.x23 = x.x23 - y.x23;

    out.x31 = x.x31 - y.x31;
    out.x32 = x.x32 - y.x32;
    out.x33 = x.x33 - y.x33;

    return out;
}

__device__ __host__
inline float3x3 matMulScalar (const float3x3 in, const float y){
    float3x3 out;
    out.x11 = in.x11 * y;
    out.x12 = in.x12 * y;
    out.x13 = in.x13 * y;

    out.x21 = in.x21 * y;
    out.x22 = in.x22 * y;
    out.x23 = in.x23 * y;

    out.x31 = in.x31 * y;
    out.x32 = in.x32 * y;
    out.x33 = in.x33 * y;

    return out;
}

__device__ __host__
inline float3x3 matDivScalar (const float3x3 in, const float y){
    float3x3 out;
    out.x11 = in.x11 / y;
    out.x12 = in.x12 / y;
    out.x13 = in.x13 / y;

    out.x21 = in.x21 / y;
    out.x22 = in.x22 / y;
    out.x23 = in.x23 / y;

    out.x31 = in.x31 / y;
    out.x32 = in.x32 / y;
    out.x33 = in.x33 / y;

    return out;
}

__device__ __host__
inline float3x3 matMulElem (const float3x3 in, const float3x3 y){
    float3x3 out;
    out.x11 = in.x11 * y.x11;
    out.x12 = in.x12 * y.x12;
    out.x13 = in.x13 * y.x13;

    out.x21 = in.x21 * y.x21;
    out.x22 = in.x22 * y.x22;
    out.x23 = in.x23 * y.x23;

    out.x31 = in.x31 * y.x31;
    out.x32 = in.x32 * y.x32;
    out.x33 = in.x33 * y.x33;

    return out;
}

__device__ __host__
inline floatvec3 vectorSub (floatvec3 x, floatvec3 y){
    floatvec3 out;
    out.x1 = x.x1 - y.x1;
    out.x2 = x.x2 - y.x2;
    out.x3 = x.x3 - y.x3;
    return out;
}

__device__ __host__
inline floatvec3 vectorScalarDiv (floatvec3 x, float y){
    floatvec3 out;
    out.x1 = x.x1 / y;
    out.x2 = x.x2 / y;
    out.x3 = x.x3 / y;
    return out;
}

__device__ __host__
inline floatvec3 crossProduct(floatvec3 x, floatvec3 y){
    floatvec3 out;
    out.x1 = x.x2 * y.x3 - x.x3 * y.x2;
    out.x2 = x.x3 * y.x1 - x.x1 * y.x3;
    out.x3 = x.x1 * y.x2 - x.x2 * y.x1;
    return out;
}

__host__ __device__
void printMat(float3x3 matrix){
    printf("row: %f , %f , %f \n", matrix.x11, matrix.x12, matrix.x13);
    printf("row: %f , %f , %f \n", matrix.x21, matrix.x22, matrix.x23);
    printf("row: %f , %f , %f \n", matrix.x31, matrix.x32, matrix.x33);
}

dim3 cuda_gridsize(int n){
    int k = (n - 1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if(x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    return d;
}

 __global__ void lie_so3_kernel(const float * r_param, float * output, const int size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= size) return;
    int k = i;
    float omega_x = r_param[k * 3 + 0];
    float omega_y = r_param[k * 3 + 1];
    float omega_z = r_param[k * 3 + 2];

    float3x3 omega_skew = {0};
    omega_skew.x12 = - omega_z;
    omega_skew.x13 = omega_y;
    omega_skew.x21 = omega_z;
    omega_skew.x23 = - omega_x;
    omega_skew.x31 = - omega_y;
    omega_skew.x32 = omega_x;

    float3x3 omega_skew_squared = matMul(omega_skew, omega_skew);

    float epsilon = 0.000001; // For numerical stability when angle is very small

    float theta_sqr = omega_x * omega_x + omega_y * omega_y + omega_z * omega_z + epsilon;
    float theta = sqrt(theta_sqr);

    float sin_theta = sin(theta);
    float sin_theta_div_theta = sin_theta / theta;

    float one_minus_cos_theta = 1.0 - cos(theta);
    float one_minus_cos_div_theta_sqr = one_minus_cos_theta / theta_sqr;

    float3x3 identity = {0};
    identity.x11 = 1.0;
    identity.x22 = 1.0;
    identity.x33 = 1.0;

    float3x3 rot_mat = matAdd(identity, matAdd(matMulScalar(omega_skew, sin_theta_div_theta), matMulScalar(omega_skew_squared, one_minus_cos_div_theta_sqr)));

    const int output_offset = k * 9;
    output[output_offset + 0] = rot_mat.x11;
    output[output_offset + 1] = rot_mat.x12;
    output[output_offset + 2] = rot_mat.x13;

    output[output_offset + 3] = rot_mat.x21;
    output[output_offset + 4] = rot_mat.x22;
    output[output_offset + 5] = rot_mat.x23;

    output[output_offset + 6] = rot_mat.x31;
    output[output_offset + 7] = rot_mat.x32;
    output[output_offset + 8] = rot_mat.x33;
}

__global__ void lie_so3_backward_kernel(const float * r_param, const float * output, const float * grad_output, float * grad_r_param, const int size){
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= size) return;
    int k = i;

    float omega_x = r_param[k * 3 + 0];
    float omega_y = r_param[k * 3 + 1];
    float omega_z = r_param[k * 3 + 2];

    floatvec3 omega_vec = copyVectorToStruct(r_param + k * 3);

    float3x3 omega_skew = {0};
    omega_skew.x12 = - omega_z;
    omega_skew.x13 = omega_y;
    omega_skew.x21 = omega_z;
    omega_skew.x23 = - omega_x;
    omega_skew.x31 = - omega_y;
    omega_skew.x32 = omega_x;

    float3x3 identity = {0};
    identity.x11 = 1.0;
    identity.x22 = 1.0;
    identity.x33 = 1.0;

    float3x3 grad_output_s = copyMatrixToStruct(grad_output + k * 9);

    float3x3 out_p = copyMatrixToStruct(output + k * 9);

    float3x3 i_minus_rotmats = matSub(identity, out_p);

    float epsilon = 0.000001; // For numerical stability when angle is very small

    float omega_mag = omega_x * omega_x + omega_y * omega_y + omega_z * omega_z + epsilon;

    int j = 0;
    for(j = 0; j < 3; j++){
        floatvec3 basis_vector = {0};
        basis_vector = vectorSet(basis_vector, j, 1.0);

        floatvec3 id_minus_r_ei = matvecMul(i_minus_rotmats, basis_vector);
        floatvec3 v_cross_id_minus_r_ei = crossProduct(omega_vec, id_minus_r_ei);

        float3x3 v_cross_skew = {0};
        v_cross_skew.x12 = - v_cross_id_minus_r_ei.x3;
        v_cross_skew.x13 = v_cross_id_minus_r_ei.x2;
        v_cross_skew.x21 = v_cross_id_minus_r_ei.x3;
        v_cross_skew.x23 = - v_cross_id_minus_r_ei.x1;
        v_cross_skew.x31 = - v_cross_id_minus_r_ei.x2;
        v_cross_skew.x32 = v_cross_id_minus_r_ei.x1;

        float v_i = vectorGet(omega_vec, j);
        float3x3 v_i_times_omega_skew = matMulScalar(omega_skew , v_i);

        float3x3 left = matDivScalar(matAdd(v_i_times_omega_skew, v_cross_skew), omega_mag);

        float3x3 grad = matMulElem(grad_output_s, matMul(left, out_p));

        grad_r_param[j + k * 3] = grad.x11 + grad.x12 + grad.x13 + grad.x21 + grad.x22 + grad.x23 + grad.x31 + grad.x32 + grad.x33;
    }

}


void lie_so3(const float * r_param, const int batch_size, float * output, cudaStream_t stream){
    cudaError_t err;
    long size = batch_size;

    lie_so3_kernel <<<cuda_gridsize(size), BLOCK, 0, stream>>>(r_param, output, size);

	err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "lieso3 cudaCheckError() failed : %s\n", cudaGetErrorString( err ));
            exit(-1);
        }
}

void printDeviceData(const float * data, const int size){
    float* test = (float*)malloc(size*sizeof(float));
    cudaMemcpy(test, data, size*sizeof(float), cudaMemcpyDeviceToHost);
    int i = 0;
    for(i = 0; i < size; i++){
        printf("Val: %f \n", test[i]);
    }
}

void lie_so3_backward(const float * r_param, const float * output, const int batch_size, const float * grad_output, float * grad_r_param, cudaStream_t stream){
    cudaError_t err;
	long size = batch_size;

    lie_so3_backward_kernel <<<cuda_gridsize(size), BLOCK, 0, stream>>>(r_param, output, grad_output, grad_r_param, size);

    err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ));
            exit(-1);
        }
}
