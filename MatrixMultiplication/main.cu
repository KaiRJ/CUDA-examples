#include "HandleError.cuh"

#include <cstddef>
#include <cstdio>
#include <time.h>
#include <vector>

namespace
{
    constexpr int g_matrix_size {4096};
    constexpr int g_block_size {16}; // CUDA maximum is 1024 *total* threads in block
    constexpr float g_a_values {1.0f};
    constexpr float g_b_values {2.0f};
} // namespace

__global__ void mmul(const float* a, const float* b, float* c, int matrix_size);

int main()
{
    // host arrays (using vector as allocation is on the heap)
    std::vector<float> a(g_matrix_size * g_matrix_size);
    std::vector<float> b(g_matrix_size * g_matrix_size);
    std::vector<float> c(g_matrix_size * g_matrix_size);

    // device pointers
    float* dev_a;
    float* dev_b;
    float* dev_c;

    // fill host matrices
    for (int i {0}; i < g_matrix_size * g_matrix_size; ++i)
    {
        a[i] = g_a_values;
        b[i] = g_b_values;
        c[i] = 0;
    }

    // start timer
    clock_t t1 {clock()};
    std::cout << "Starting compute..." << "\n";

    // allocate memory on GPU
    const std::size_t bytes = a.size() * sizeof(float);
    HANDLE_ERROR(cudaMalloc(&dev_a, bytes));
    HANDLE_ERROR(cudaMalloc(&dev_b, bytes));
    HANDLE_ERROR(cudaMalloc(&dev_c, bytes));

    // copy data to GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a.data(), bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // launch kernel
    dim3 block(g_block_size, g_block_size); // dim3 variable holds 3 dimensions
    dim3 grid((g_matrix_size + block.x - 1) / block.x,
              (g_matrix_size + block.y - 1) / block.y);

    mmul<<<grid, block>>>(dev_a, dev_b, dev_c, g_matrix_size);
    HANDLE_ERROR(cudaGetLastError());

    // copy result to host
    cudaMemcpy(c.data(), dev_c, bytes, cudaMemcpyDeviceToHost);

    // verify results
    for (int i {0}; i < g_matrix_size * g_matrix_size; ++i)
    {
        if (c[i] != g_a_values * g_b_values * g_matrix_size)
        {
            constexpr float expected_value {g_a_values * g_b_values * g_matrix_size};
            std::cout << "mismatch at index " << i << ":" << "\n";
            std::cout << "\t was: " << c[i] << ", should be: " << expected_value << "\n";
            return -1;
        }
    }

    std::cout << "Success!" << "\n";
    return 0;
}

__global__ void mmul(const float* a, const float* b, float* c, int matrix_size)
{
    unsigned int x_idx {threadIdx.x + blockDim.x * blockIdx.x};
    unsigned int y_idx {threadIdx.y + blockDim.y * blockIdx.y};

    if ((x_idx < matrix_size) && (y_idx < matrix_size))
    {
        float temp {0};
        for (int i {0}; i < matrix_size; ++i)
        {
            // dot product of row and column
            temp += a[y_idx * matrix_size + i] * b[i * matrix_size + x_idx];
        }

        c[y_idx * matrix_size + x_idx] = temp;
    }
}
