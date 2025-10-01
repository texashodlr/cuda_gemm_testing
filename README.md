# CUDA GEMM Playgound
Practice, learning and performance testing CUDA GEMM Kernels on various Nvidia GPUs: 1070, 3070ti, 4070, A4000.

## Running the Code
1. Set your architecture inside of: `CMakeLists.txt`, this repo is compiled on an A4000 hence it uses: `set(CUDA_NVCC_FLAGS -arch=compute_86; -code=compute_86)`
2. Configure the maximum matrix size for computation:
    1. Modify `size_len` in `sgemm.cu:37`. It is recommended to set it to 16 for the initial run.
        1. Larger sizes _may_ cause the power supply to overload and the host to reboot.
3. Compile `cd build && cmake.. && make`.
4. Run `run.sh` to calculate the computational efficiency of each kernel function and save the results in the `test/` directory.
5. Computational efficiency line plot: `python[3] plot.py 0 1` plots a comparison chart of the computational efficiency of CUBLAS and `kernel_1`.

