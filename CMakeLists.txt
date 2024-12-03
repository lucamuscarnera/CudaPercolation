# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Project name and enable CUDA
project(percolation LANGUAGES CXX CUDA)

# Set the CUDA standard (optional)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add your CUDA source files
set(SRC_FILES main.cu)  # Add more .cu files if necessary

# Specify the compute capability (adjust according to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 60)  # Example for Pascal GPUs (sm_60)

# Define the executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Optional: Additional CUDA compiler options
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
