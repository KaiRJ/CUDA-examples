#include "HandleError.cuh"

#include <algorithm>
#include <array>
#include <cuda_runtime_api.h>

namespace
{
    constexpr int g_array_size {33 * 1024};
    constexpr int g_threads_per_block {256}; // must be a power of 2 for reduction
    constexpr int g_blocks_per_grid {
        std::min(32, (g_array_size + g_threads_per_block - 1) / g_threads_per_block)};
} // namespace

__global__ void dot(const float* a, const float* b, float* c);

int main()
{
    std::array<float, g_array_size> a {};
    std::array<float, g_array_size> b {};
    std::array<float, g_blocks_per_grid> partial_c {}; // each blocks sum
    float c {};

    float* dev_a {};
    float* dev_b {};
    float* dev_partial_c {};

    // allocate memory on the GPU
    HANDLE_ERROR(cudaMalloc(&dev_a, sizeof(a)));
    HANDLE_ERROR(cudaMalloc(&dev_b, sizeof(b)));
    HANDLE_ERROR(cudaMalloc(&dev_partial_c, sizeof(partial_c)));

    // fill in the host memory with data
    for (int i {0}; i < g_array_size; ++i)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // copy  'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a.data(), sizeof(a), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b.data(), sizeof(b), cudaMemcpyHostToDevice));

    dot<<<g_blocks_per_grid, g_threads_per_block>>>(dev_a, dev_b, dev_partial_c);
    HANDLE_ERROR(cudaGetLastError());

    // copy  'c' from the GPU
    HANDLE_ERROR(cudaMemcpy(partial_c.data(), dev_partial_c, sizeof(partial_c),
                            cudaMemcpyDeviceToHost));

    // finish up on the CPU side
    c = 0;
    for (int i {0}; i < g_blocks_per_grid; ++i)
    {
        c += partial_c[i];
    }

    auto sum_squares = [](float num) { return num * (num + 1) * (2 * num + 1) / 6; };
    std::cout << "Does GPU value " << c << " = " << 2 * sum_squares(g_array_size - 1)
              << "\n";

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    return 0;
}

__global__ void dot(const float* a, const float* b, float* c)
{
    __shared__ float cache[g_threads_per_block];
    const unsigned int cache_index {threadIdx.x};
    unsigned int tid {threadIdx.x + (blockIdx.x * blockDim.x)};

    // get product of each corresponding element in vectors
    float temp {0};
    while (tid < g_array_size)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cache_index] = temp;

    __syncthreads(); // make sure all threads are finished multiplying

    // do reduction
    unsigned int i {blockDim.x / 2};
    while (i != 0)
    {
        if (cache_index < i)
        {
            cache[cache_index] += cache[cache_index + i];
        }

        __syncthreads();
        i /= 2;
    }

    if (cache_index == 0)
    {
        c[blockIdx.x] = cache[0];
    }
}
