cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER gcc-15)
set(CMAKE_CUDA_HOST_COMPILER g++-15)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

project(main LANGUAGES CXX CUDA)

add_executable(main main.cu)

target_include_directories(main PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../Utilities")
