#include <iostream>
#include <cuda.h>


__global__ void kernel(int x) {
    asm(".reg .u32 t1;\n\t"
        " mul.lo.u32 t1, %1, %1;\n\t"
        " mul.lo.u32 %0, t1, %1;"
        : "=r"(x) : "r"(x));
    printf("x=%d\n", x);
}

int main()
{
    unsigned int x = 5;
    kernel<<<1, 1>>>(x);
    cudaDeviceSynchronize();

    std::cout << "PTX code embedded successfully!" << std::endl;
    return 0;
}