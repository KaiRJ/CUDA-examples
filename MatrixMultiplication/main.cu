#include "HandleError.cuh"

#include <cstddef>
#include <cstdio>
#include <time.h>
#include <vector>

namespace
{
    constexpr int g_matrix_size {8192};
    constexpr int g_block_size {32}; // CUDA maximum is 1024 *total* threads in block
    constexpr float g_a_values {3.0f};
    constexpr float g_b_values {2.0f};
} // namespace

bool verifyResult(const std::vector<float>& c);
__global__ void mmul_naive(const float* a, const float* b, float* c);
__global__ void mmul_shared(const float* a, const float* b, float* c);

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

    // allocate memory on GPU
    const std::size_t bytes = a.size() * sizeof(float);
    HANDLE_ERROR(cudaMalloc(&dev_a, bytes));
    HANDLE_ERROR(cudaMalloc(&dev_b, bytes));
    HANDLE_ERROR(cudaMalloc(&dev_c, bytes));

    // copy data to GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a.data(), bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(g_block_size, g_block_size); // dim3 variable holds 3 dimensions
    dim3 grid((g_matrix_size + block.x - 1) / block.x,
              (g_matrix_size + block.y - 1) / block.y);

    std::cout << "Starting naive compute..." << "\n";
    clock_t t1 {clock()};

    // launch kernel
    mmul_naive<<<grid, block>>>(dev_a, dev_b, dev_c);
    HANDLE_ERROR(cudaGetLastError());

    // copy result to host
    cudaMemcpy(c.data(), dev_c, bytes, cudaMemcpyDeviceToHost);

    clock_t t2 {clock()};
    double dt = static_cast<double>(t2 - t1) / CLOCKS_PER_SEC;
    printf("Done. Compute took %f seconds\n", dt);

    if (!verifyResult(c))
    {
        return -1;
    }
    std::cout << "Success!" << "\n";

    std::cout << "Starting shared compute..." << "\n";
    t1 = clock();

    // launch kernel
    mmul_shared<<<grid, block>>>(dev_a, dev_b, dev_c);
    HANDLE_ERROR(cudaGetLastError());

    // copy result to host
    cudaMemcpy(c.data(), dev_c, bytes, cudaMemcpyDeviceToHost);

    t2 = clock();
    dt = static_cast<double>(t2 - t1) / CLOCKS_PER_SEC;
    printf("Done. Compute took %f seconds\n", dt);

    if (!verifyResult(c))
    {
        return -1;
    }
    std::cout << "Success!" << "\n";

    return 0;
}

bool verifyResult(const std::vector<float>& c)
{
    for (int i {0}; i < g_matrix_size * g_matrix_size; ++i)
    {
        if (c[i] != g_a_values * g_b_values * g_matrix_size)
        {
            constexpr float expected_value {g_a_values * g_b_values * g_matrix_size};
            std::cout << "mismatch at index " << i << ":" << "\n";
            std::cout << "\t was: " << c[i] << ", should be: " << expected_value << "\n";
            return false;
        }
    }

    return true;
}

__global__ void mmul_naive(const float* a, const float* b, float* c)
{
    unsigned int x_idx {threadIdx.x + blockDim.x * blockIdx.x};
    unsigned int y_idx {threadIdx.y + blockDim.y * blockIdx.y};

    if ((x_idx < g_matrix_size) && (y_idx < g_matrix_size))
    {
        float temp {0};
        for (int i {0}; i < g_matrix_size; ++i)
        {
            // dot product of row and column
            temp += a[y_idx * g_matrix_size + i] * b[i * g_matrix_size + x_idx];
        }

        c[y_idx * g_matrix_size + x_idx] = temp;
    }
}

__global__ void mmul_shared(const float* a, const float* b, float* c)
{
    // declare cache in shared memory
    __shared__ float s_a[g_block_size][g_block_size];
    __shared__ float s_b[g_block_size][g_block_size];

    // thread indexes
    unsigned int x_idx {threadIdx.x + (blockIdx.x * blockDim.x)};
    unsigned int y_idx {threadIdx.y + (blockIdx.y * blockDim.y)};

    // check if this thread is within the matrix
    if ((x_idx < g_matrix_size) && (y_idx < g_matrix_size))
    {
        // read data into shared memory in g_block_size chunks
        float temp {0};
        for (int i {0}; i < g_matrix_size / g_block_size; ++i)
        {
            // each thread loads one element of the matrix into shared memory
            const int offset {i * g_block_size};

            // load rows of chunk of matrix into shared memory
            s_a[threadIdx.y][threadIdx.x] =
                a[y_idx * g_matrix_size + (offset + threadIdx.x)];

            // load columns of chunk of matrix into shared memory
            s_b[threadIdx.y][threadIdx.x] =
                b[(offset + threadIdx.y) * g_matrix_size + x_idx];

            __syncthreads();

            // do matrix multiplication of chunk of the matrix
            for (int k {0}; k < g_block_size; ++k)
            {
                // dot product of row and column
                temp += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
            }

            __syncthreads();
        }

        c[y_idx * g_matrix_size + x_idx] = temp;
    }
}
