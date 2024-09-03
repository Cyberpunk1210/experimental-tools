#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <ATen/ATen.h>

__device__ unsigned int block = 32;

__global__ void MatAdd_kernel(at::Tensor A, at::Tensor B, at::Tensor C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < A.sizes()[0] && j < A.sizes()[1])
    {
        C[i][j] = A[i][j] + B[i][j];
    }
}


__global__ void cmp_kelnel(std::string s, at::Tensor dt, at::Tensor t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockIdx.y + threadIdx.y;
    
}

