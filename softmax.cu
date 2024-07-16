#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 16

typedef struct{
    int width;
    int height;
    float* elements;
} Matrix;


template<template<typename> typename ReductionOp, typename T>
__inline__ __device__ T WarpAllReduce(T val){
    for (int mask = kWarpSize / 2; mask > 0; mask /= 2){
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T, int pack_size, int cols_per_thread, bool padding>
__global__ void SoftmaxWarppImpl(const int64_t rows, const int64_t cols, const T* x, T* y){
    static_assert(cols_per_thread % pack_size == 0, "");
    constexpr int num_packs = cols_per_thread / pack_size;
    assert(cols <= cols_per_thread * kWarpSize);
    using ComputeType = typename GetComputeType<T>::type;
    ComputeType buf[cols_per_thread];
    const int global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int num_global_warp = gridDim.x * blockDim.y;
    const int lane_id = threadIdx.x;
    for (int64_t row=global_warp_id; row<rows; row+=num_global_warp){
        const int64_t row_offset = row * cols;
        const T* row_x = x + row_offset;
        T* row_y = y + row_offset;
        ComputeType thread_max = -Inf<ComputeType>();
    #pragma unroll
        for (int pack_id = 0; pack_id < num_packs; ++pack_id){
            const int col = (pack_id * kWarpSize + lane_id) * pack_size;
            if (!padding || col < cols){
                MultiFetch<T, ComputeType, pack_size>()(buf + pack_id * pack_size, row_x + col);
    #pragma unroll
            for (int i=0; i < pack_size; ++i){
                thread_max = max(thread_max, buf[pack_id * pack_size + i]);
            }
            } else{
    #pragma unroll
                for (int i=0; i < pack_size; ++i) { buf[pack_id * pack_size + i] = -Inf<ComputeType>();}
            }
        }
        const ComputeType warp_max = WarpAllReduce<MaxOp, ComputeType>(thread_max);
    #pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
          buf[i] = exp(buf[i] - warp_max);
          thread_sum += buf[i];
        }
        const ComputeType warp_sum = WarpAllReduce<SumOp, ComputeType>(thread_sum);
    #pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) { buf[i] = buf[i] / warp_sum; }
    #pragma unroll
        for (int i = 0; i < num_packs; ++i) {
          const int col = (i * kWarpSize + lane_id) * pack_size;
          if (!padding || col < cols) {
            MultiStore<ComputeType, T, pack_size>()(row_y + col, buf + i * pack_size);
          }
        }
    }
}