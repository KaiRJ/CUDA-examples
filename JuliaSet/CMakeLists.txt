cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_COMPILER gcc-15)
set(CMAKE_CUDA_HOST_COMPILER g++-15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

project(main LANGUAGES C CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

set(UTIL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Utilities")

add_executable(main
  main.cu
  ${UTIL_DIR}/Window/Window.cpp
  ${UTIL_DIR}/CPUBitmap/CPUBitmap.cpp
  ${UTIL_DIR}/glad/glad.c
)

target_include_directories(main
  PRIVATE
  ${UTIL_DIR}
)

target_link_libraries(main
  PRIVATE
  glfw
  OpenGL::GL
)
