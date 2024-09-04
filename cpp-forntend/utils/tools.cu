#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <ATen/ATen.h>

__device__ unsigned int block = 32;

// __global__ void MatAdd_kernel(at::Tensor A, at::Tensor B, at::Tensor C, int N, int M)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     if (i < N && j < M) C[i][j] = A[i][j] + B[i][j];
// }

