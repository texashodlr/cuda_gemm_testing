#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*
=====================================
CUDA Operations
=====================================
*/
void cudaCheck(cudaError_t error, const char *file, int line); // CUDA Error Checking
void CudaDeviceInfo();                                         // Print CUDA Information

/*
=====================================
Matrix Operations
=====================================
*/
void randomize_matrix(float *mat, int N);                      // Randomized initialization of the matrix
void copy_matrix(float *src, float *dest, int N);              // Copy the matrix
void print_matrix(const float *A, int M, int N);               // Print the matrix
bool verify_matrix(float *mat1, float *mat2, int N);           // Validate the matrix

/*
=====================================
Time Operations
=====================================
*/
float get_current_sec();                                      // Get current time
float cpu_elapsed_time(float &beg, float &end);               // Get elapsed CPU time

/*
=====================================
Kernel Opeartions
=====================================
*/
// Call the specified kernel function to calculate matrix multiplication
void test_kernel(int kernel_num, int m, int n, int k, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle);