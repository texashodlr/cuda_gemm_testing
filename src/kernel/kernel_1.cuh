#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

__global__ __launch_bounds__(1024) void
mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int gx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread X, column index in C and B
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // Global thread Y, row index in C and A

    // Inner product of a singular output Element of C
    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[gy * K + i] * B[i * N + gx];       // Two global memory accesses and one FMA: Fused Multiply Add
    }
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}

/*
    ### General Notes ###
Reminder: GEMM: α·A·B + β·C -> C     

Shape sizing:
    A = M × K
    B = K × N
    C = M × N (Assuming row major layout)

Thread's produce one output elemnt C[gy, gx]

To cover the whole C matrix you'd launch a 2D grid such that:
    gridDim.x * blockDim.x > N
    gridDim.y * blockDim.y > M
        (Blocks of 16 x 16 or etc.)

Inner product notes:
    A[gy * K + i] walks across row `gy` of A (contiguous in memory)
    B[i * N + gx] walks down column gx of B indexing is linear with stride `N` across rows for a fixed i..
        consecutive gx values access consecutive elements of row i of B good for coalescing across threads in x
    Loop for the dot prod of row gy of A and col gX of B.
        --> tmp = Σ_i A[gy,i] * B[i,gx]
    
    Writes: C[gy,gx] = α·tmp + β·C[gy,gx]

Determining Arithmetic Intensity:
    For each i, each thread does:
        2 Global Loads one from A and one from B
        1 FMA --> Very memory bound: lots of global traffic per flop
        No shared memory tiling every thread independently reloads the same A/B values that nearby threads also need,
        So each block cooperatively loads a tile of A (Mtile x Ktile) and B (Ktile and Ntile)

RTX A4000 -> 192 Tensor Cores, time to vary M, N and K
     - Looking at suggested 256x128 and 128x256 tile sizing -> 32768 elements



*/