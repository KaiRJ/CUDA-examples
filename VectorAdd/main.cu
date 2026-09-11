#include "HandleError.cuh"

#include <array>
#include <iostream>

constexpr int g_vector_size {33 * 1024};

__global__ void add(int* a, int* b, int* c)
{
    unsigned int tid {threadIdx.x + (blockIdx.x * blockDim.x)};
    while (tid < g_vector_size)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main()
{

    std::array<int, g_vector_size> a {};
    std::array<int, g_vector_size> b {};
    std::array<int, g_vector_size> c {};

    int* dev_a {};
    int* dev_b {};
    int* dev_c {};

    // allocate memory on the GPU
    HANDLE_ERROR(cudaMalloc(&dev_a, sizeof(a)));
    HANDLE_ERROR(cudaMalloc(&dev_b, sizeof(b)));
    HANDLE_ERROR(cudaMalloc(&dev_c, sizeof(c)));

    for (int i {0}; i < g_vector_size; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy  'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a.data(), sizeof(a), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b.data(), sizeof(b), cudaMemcpyHostToDevice));

    add<<<128, 128>>>(dev_a, dev_b, dev_c);

    // copy  'c' from the GPU
    HANDLE_ERROR(cudaMemcpy(c.data(), dev_c, sizeof(c), cudaMemcpyDeviceToHost));

    // verify the program worked
    bool result {true};
    for (int i {0}; i < g_vector_size; ++i)
    {
        if (a[i] + b[i] != c[i])
        {
            result = false;
            std::cout << a[i] << " + " << b[i] << " != " << c[i] << "\n";
        }
    }
    if (result)
    {
        std::cout << "Success!" << "\n";
    }

    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
