cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

project(main LANGUAGES C CXX CUDA)

set(UTIL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Utilities")

add_executable(main
  main.cu)

target_include_directories(main
  PRIVATE
  ${UTIL_DIR}
)
