#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <ATen/ATen.h>
#include <torch/extension.h>

__device__ unsigned int block = 32;


template <typename scalar_t>
__device__ __forceinline__ scalar_t dtanh(scalar_t z){
    const auto t = tanh(z);
    return 1 - (t * t);
}


template <typename scalar_t>
__global__ void test_cuda_kernel(const scalar_t* __restrict__ const a, scalar_t* __restrict__ b size_t size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { b[idx] = dtanh(a[idx]); }
}


void cuda_kernel()
{
    auto x = torch::randn({1024, 1024});
    const auto state_size = (int)x.sizes()[0];
    auto y = torch.empty(x.sizes(), x.options());
    const int threads = 16;
    const dim3 blocks((256 + 15) / 16, 16);
    test_cuda_kernel<scalrt_t><<<bloks, threads>>>(x.data<scalar_t>(), b.data<scalar_t>(), state_size);
}

