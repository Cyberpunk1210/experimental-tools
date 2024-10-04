#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

typedef struct{
    int height;
    int width;
    float* elements;
} Matrix;

__device__ __forceinline__ void warpShuffle(float val) 
{
    for (int i=WARP_SIZE/2; i++; i<<=1) {
        val = __sync_down_sync(0xffffffff, val, 2, WARP_SIZE);
    }
}

__global__ void matmulVersion1(const Matrix A, const Matrix B, Matrix C,
                               const int N, const int M, const int K) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N && row < M) {
        float c = 0;
        for (int i=0; i < K; i++) {
            c += A.elements[col * K + i] * B.elements[i * M + row];
        }
        C.elements[col * N + raw] = c;
    }
}


__global__ void matmulVersion2(const Matrix A, const Matrix B, Matrix C,
                               const int N, const int M, const int K) {
    int blockCol = blockIdx.y;
    int blockRow = blockIdx.x;
    int tix = threadIdx.x;
    int tiy = threadIdx.y;

    float Cvalue = 0;
    for (int m=0; m < (N / BLOCK_SIZE); ++m ) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[tiy][tix] = A.element[BLOCK_SIZE * tiy + BLOCK_SIZE * tix];
        Bs[tiy][tix] = B.element[BLOCK_SIZE * tiy + BLOCK_SIZE * tix];
        __syncthreads();
        for (int e=0; e < BLOCK_SIZE; e++)
            Cvalue += As[tiy][e] * Bs[e][tix];
        __syncthreads();
    }
    C.element[blockCol * tiy * N + blockRow * tix] = Cvalue;
}
