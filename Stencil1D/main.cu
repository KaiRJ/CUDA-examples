#include "HandleError.cuh"

#include <algorithm>
#include <iostream>
#include <vector>

namespace
{
    constexpr int g_n_threads {4096};
    constexpr int g_radius {3};
    constexpr int g_block_size {16};
} // namespace

__global__ void stencil_1d(const int* in, int* out);

int main(void)
{
    // host arrays
    constexpr size_t length {g_n_threads + 2 * g_radius};
    std::vector<int> in(length);
    std::vector<int> out(length);

    // setup values
    std::fill_n(in, length, 1);
    std::fill_n(out, length, 1);

    // device copies of a, b, c
    int* d_in;
    int* d_out;

    // alloc space for device copies
    constexpr int size {length * sizeof(int)};
    HANDLE_ERROR(cudaMalloc(&d_in, size));
    HANDLE_ERROR(cudaMalloc(&d_out, size));

    // copy to device
    HANDLE_ERROR(cudaMemcpy(d_in, in.data(), size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_out, out.data(), size, cudaMemcpyHostToDevice));

    // launch stencil_1d() kernel on GPU
    stencil_1d<<<g_n_threads / g_block_size, g_block_size>>>(d_in + g_radius,
                                                             d_out + g_radius);

    // copy result back to host
    cudaMemcpy(out.data(), d_out, size, cudaMemcpyDeviceToHost);

    // error Checking
    for (int i = 0; i < g_n_threads + 2 * g_radius; i++)
    {
        if (i < g_radius || i >= g_n_threads + g_radius)
        {
            if (out[i] != 1)
            {
                std::cout << "Mismatch at index " << i << ", was: " << out[i]
                          << ", should be: " << 1 << "\n";
            }
        }
        else
        {
            if (out[i] != 1 + 2 * g_radius)
            {
                std::cout << "Mismatch at index " << i << ", was: " << out[i]
                          << ", should be: " << 1 + 2 * g_radius << "\n";
            }
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "Success!\n";
    return 0;
}

__global__ void stencil_1d(const int* in, int* out)
{
    __shared__ int temp[g_block_size + 2 * g_radius];

    unsigned int g_index {threadIdx.x + blockIdx.x * blockDim.x};
    unsigned int l_index {threadIdx.x + g_radius};

    // read input elements into shared memory
    temp[l_index] = in[g_index];

    // fill halo on either side of shared memory
    if (threadIdx.x < g_radius)
    {
        temp[l_index - g_radius] = in[g_index - g_radius];
        temp[l_index + g_block_size] = in[g_index + g_block_size];
    }

    // synchronize (ensure all the data is available)
    __syncthreads();

    // apply the stencil
    int result {0};
    for (int offset {-g_radius}; offset <= g_radius; ++offset)
    {
        result += temp[l_index + offset];
    }

    // store the result
    out[g_index] = result;
}
