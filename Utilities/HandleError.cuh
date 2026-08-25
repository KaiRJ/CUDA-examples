#ifndef __HANDLE_ERROR_H__
#define __HANDLE_ERROR_H__

#include <iostream>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
static void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line
                  << "\n";
        exit(EXIT_FAILURE);
    }
}

#endif
