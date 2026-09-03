#include "CPUBitmap/CPUBitmap.hpp"
#include "HandleError.cuh"
#include "glad/glad.h"

constexpr int g_dim {16 * 50};

void generate_frame(CPUBitmap& bitmap, unsigned char* dev_bitmap, int ticks);
__global__ void kernel(unsigned char* ptr, int ticks);

int main()
{
    // initialise bitmap and allocate memory on GPU
    CPUBitmap bitmap {g_dim, g_dim};
    unsigned char* dev_bitmap;
    HANDLE_ERROR(cudaMalloc(&dev_bitmap, bitmap.image_size()));

    int ticks {0};
    Window window {g_dim, g_dim};
    while (!window.shouldClose())
    {
        generate_frame(bitmap, dev_bitmap, ticks++);
        bitmap.Draw();

        // swap buffers and handle input
        window.processInput();
        window.swapBuffers();
        glfwPollEvents();
    }

    cudaFree(dev_bitmap);
    return 0;
}

void generate_frame(CPUBitmap& bitmap, unsigned char* dev_bitmap, int ticks)
{
    dim3 blocks {g_dim / 16, g_dim / 16};
    dim3 threads {16, 16};
    kernel<<<blocks, threads>>>(dev_bitmap, ticks);
    HANDLE_ERROR(cudaMemcpy(bitmap.get_pixels(), dev_bitmap, bitmap.image_size(),
                            cudaMemcpyDeviceToHost));
}

__global__ void kernel(unsigned char* ptr, int ticks)
{
    // map from threadIdx/BlockIdx to pixel position
    unsigned int x {threadIdx.x + (blockIdx.x * blockDim.x)};
    unsigned int y {threadIdx.y + (blockIdx.y * blockDim.y)};
    unsigned int offset {x + (y * blockDim.x * gridDim.x)};

    // calculate value at this pixel
    auto fx {static_cast<float>(x - g_dim / 2.f)};
    auto fy {static_cast<float>(y - g_dim / 2.f)};
    float d {sqrtf(fx * fx + fy * fy)};

    auto grey {static_cast<unsigned char>(
        128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f))};

    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;
}
