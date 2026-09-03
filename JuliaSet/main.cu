#include "CPUBitmap/CPUBitmap.hpp"
#include "HandleError.cuh"
#include "glad/glad.h"

constexpr int g_dim {1000};
constexpr float g_scale {1.0};

__global__ void kernel(unsigned char* ptr);
__device__ int julia(int x, int y);

struct Complex
{
    float r {};
    float i {};

    __device__ Complex(float a, float b) : r {a}, i {b} {}

    __device__ float magnitude2() { return (r * r) + (i * i); }

    __device__ Complex operator*(const Complex& a)
    {
        return Complex((r * a.r) - (i * a.i), (i * a.r) + (r * a.i));
    }

    __device__ Complex operator+(const Complex& a) { return Complex(r + a.r, i + a.i); }
};

int main()
{
    // initialise bitmap and allocate memory on GPU
    CPUBitmap bitmap {g_dim, g_dim};
    unsigned char* dev_bitmap;
    HANDLE_ERROR(cudaMalloc(&dev_bitmap, bitmap.image_size()));

    // run kernal and copy results from GPU
    dim3 grid {g_dim, g_dim};
    kernel<<<grid, 1>>>(dev_bitmap);
    HANDLE_ERROR(cudaMemcpy(bitmap.get_pixels(), dev_bitmap, bitmap.image_size(),
                            cudaMemcpyDeviceToHost));

    // draw the results
    Window window {g_dim, g_dim};
    while (!window.shouldClose())
    {
        bitmap.Draw();

        // swap buffers and handle input
        window.processInput();
        window.swapBuffers();
        glfwPollEvents();
    }

    cudaFree(dev_bitmap);
    return 0;
}

__global__ void kernel(unsigned char* ptr)
{
    // map from threadIdx/BlockIdx to pixel position
    unsigned int x {blockIdx.x};
    unsigned int y {blockIdx.y};
    unsigned int offset {x + y * gridDim.x};

    // now calculate the value at that position
    int juliaValue {julia(x, y)};
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

__device__ int julia(int x, int y)
{
    float jx {g_scale * (g_dim / 2.f - x) / (g_dim / 2.f)};
    float jy = {g_scale * (g_dim / 2.f - y) / (g_dim / 2.f)};
    Complex c(-0.8, 0.156);
    Complex a(jx, jy);

    for (int i {0}; i < 200; ++i)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000.)
            return 0;
    }

    return 1;
}
