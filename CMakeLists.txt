# Translated originally from:
# https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE/blob/master/CMakeLists.txt
######################################
# General Chinese to English Translations from the Readme
#
# Configure NVCC compilation parameters.
# Modify `set(CUDA_NVCC_FLAGS -arch=compute_70; -code=compute_70)` in `CMakeLists.txt`.
#
# Configure the maximum matrix size for computation.
# Modify `size_len` in `sgemm.cu:16`. It's recommended to set it to 16 for the initial run. A larger size may cause the power supply to overload and the host to reboot.
# 
# Compile `cd build && cmake .. && make`.
# Run `run.sh` to calculate the computational efficiency of each kernel function and save the results in the `test` directory.
#
# Computational efficiency line plot: `python plot.py 0 1` plots a comparison chart of the computational efficiency of CUBLAS and kernel_1.
######################################

cmake_minimum_required(VERSION 3.0)
project(CUDA_GEMM_TESTING)

# gcc/g++ compilation parameter description:
# -O1 <> 3 represent four levels of compiler optimizations options，-O1 is the default. Higher levels result in better optimization but also increase compilation time.
# -std=c++11 compiles to c++11 standard.
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

# nvcc compilation parameter description:
# -g: Adds debugging information to the host code
# -G: Generates debugging information for the device code, which disables most compiler optimizations and causes the device code to run slowly
# -Xptxas -dlcm=ca enables the L1 Cache
# -Xptxas -dlcm=cg disables the L1 Cache

# set(CUDA_NVCC_FLAGS -g;-G;-Xptxas;-dlcm=ca)
# set(CUDA_NVCC_FLAGS -Xptxas;-dlcm=cg)
# set(CUDA_NVCC_FLAGS -arch=compute_70;-code=compute_70)
# For RTX A4000 Series GPUs
set(CUDA_NVCC_FLAGS -arch=compute_86;-code=compute_86)

# If a FIND CUDA ERROR occurs, add the correct config/envvars in ~/.bashrc
# CUDA_HOME=/usr/local/cuda
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
find_package(CUDA REQUIRED)

# Configure header file search path
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure source file search path for compilation
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)
# Executable file output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# Generate an executable file
CUDA_ADD_EXECUTABLE(sgemm sgemm.cu ${SRC})

# link cudart cublas
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})