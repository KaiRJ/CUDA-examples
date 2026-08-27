#include "HandleError.cuh"

#include <array>
#include <iostream>

constexpr int g_N {10};

__global__ void add(int* a, int* b, int* c)
{
    int tid = blockIdx.x;
    if (tid < g_N)
        c[tid] = a[tid] + b[tid];
}

int main()
{

    std::array<int, g_N> a {};
    std::array<int, g_N> b {};
    std::array<int, g_N> c {};

    int* dev_a {};
    int* dev_b {};
    int* dev_c {};

    // allocate memory on the GPU
    HANDLE_ERROR(cudaMalloc(&dev_a, sizeof(a)));
    HANDLE_ERROR(cudaMalloc(&dev_b, sizeof(b)));
    HANDLE_ERROR(cudaMalloc(&dev_c, sizeof(c)));

    for (int i {0}; i < g_N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy  'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a.data(), sizeof(a), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b.data(), sizeof(b), cudaMemcpyHostToDevice));

    add<<<g_N, 1>>>(dev_a, dev_b, dev_c);

    // copy  'c' from the GPU
    HANDLE_ERROR(cudaMemcpy(c.data(), dev_c, sizeof(c), cudaMemcpyDeviceToHost));

    // display the results
    for (int i {0}; i < g_N; ++i)
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << "\n";

    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
