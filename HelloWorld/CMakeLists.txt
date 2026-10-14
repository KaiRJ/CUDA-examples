cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set correct compilers
set(CMAKE_C_COMPILER gcc-14)
set(CMAKE_CXX_COMPILER g++-14)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)

# set specific standards and architectures
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70)

project(main LANGUAGES C CXX CUDA)

set(UTIL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Utilities")

add_executable(main
  main.cu)

target_include_directories(main
  PRIVATE
  ${UTIL_DIR}
)
