#include "HandleError.cuh"

__global__ void hello() { printf("Hello from GPU!\n"); }

int main()
{
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}
