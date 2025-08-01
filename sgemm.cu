#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <kernel_run.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA's cuBLAS)"
			<< std::endl;
		exit(EXIT_FAILURE);
	}

	// Kernel number
	int deviceIdx = 0;
	if (getenv("DEVICE") != NULL) {
		deviceIdx = atoi(getenv("DEVICE"));
	}
	cudaCheck(cudaSetDevice(deviceIdx));

	printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

	cublasHandle_t handle;
	if (cublasCreate(&handle)) {
		std::cerr << "Create cublas handle error." << std::endl;
		exit(EXIT_FAILURE);
	};

	// Per siboehm, using cudaEvent for GPU stream timing, cudaEvent is equiv
	// to publishing event tasks in the target stream
	float elapsed_time;
	cudaEvent_t beg, end;
	cudaEventCreate(&beg);
	cudaEventCreate(&end);

	// cuBLAS FLOPs ceiling is reached at 8192 vector size
	std::vector<int> SIZE = { 128, 256, 512, 1024, 2048, 4096 };

	long m, n, k, max_size;
	max_size = SIZE[SIZE.size() - 1];
	std::cout << "Max size: " << max_size << std::endl;

	// GEMM input params, C = alpha*AB + beta*C
	float alpha = 0.5, beta = 3.0;

	// Host matrices
	float* A = nullptr, * B = nullptr, * C = nullptr,
		* C_ref = nullptr;
	
	// Device matrices
	float* dA = nullptr, * dB = nullptr, * dC = nullptr, * dC_ref = nullptr;

	A = (float*)malloc(sizeof(float)) * max_size * max_size);
	B = (float*)malloc(sizeof(float)) * max_size * max_size);
	C = (float*)malloc(sizeof(float)) * max_size * max_size);
	C_ref = (float*)malloc(sizeof(float)) * max_size * max_size);

	randomize_matrix(A, max_size * max_size);
	randomize_matrix(B, max_size * max_size);
	randomize_matrix(C, max_size * max_size);

	cudaCheck(cudaMalloc((void**)) & dA, sizeof(float) * max_size * max_size));
	cudaCheck(cudaMalloc((void**)) & dB, sizeof(float) * max_size * max_size));
	cudaCheck(cudaMalloc((void**)) & dC, sizeof(float) * max_size * max_size));
	cudaCheck(cudaMalloc((void**)) & dC_ref, sizeof(float) * max_size * max_size));

	cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

	int repeat_times = 50;

	for (int size : SIZE) {
		m = n = k = size;
		
		std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
				  << ", beta: " << beta << std::endl;

		// Verify the correctness of the calculation and execute it once before
		// the kernel func timing to avoid cold start errors

		if (kernel_num != 0) {
			run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
				handle); // cuBLAS
			run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
				handle); // Executes the kernel, modifies the result matrix
			cudaCheck(cudaDeviceSynchronize());
			cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
			cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
			cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

			if (!verify_matrix(C_ref, C, m * n)) {
				std::cout
					<< "Failed to pass the correctness verification against NVIDIA "
					"cuBLAS."
					<< std::endl;
				if (m <= 128) {
					std::cout << " Logging faulty output into " << errLogFile << "\n";
					std::ofstream fs;
					fs.open(errLogFile);
					fs << "A:\n";
					print_matrix(A, m, n, fs);
					fs << "B:\n";
					print_matrix(B, m, n, fs);
					fs << "C:\n";
					print_matrix(C, m, n, fs);
					fs << "Should:\n";
					print_matrix(C_ref, m, n, fs);
				}
				exit(EXIT_FAILURE);
			}
		}

		cudaEventRecord(beg);
		for (int j = 0; j < repeat_times; j++) {
			// siboehm: Don't reset dC between runs to save time
			run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
		}

		cudaEventRecord(end);
		cudaEventSynchronize(beg);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&elapsed_time, beg, end);
		elapsed_time /= 1000.; // conversion -> seconds

		long flops = 2 * m * n * k;
		printf(
			"Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
			"(%ld).\n",
			elapsed_time / repeat_times,
			(repeat_times * flops * 1e-9) / elapsed_time, m);
		fflush(stdout);
		// make dC and dC_ref equal again (we modified dC while calling our kernel
		// for benchmarking)
		cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
			cudaMemcpyDeviceToDevice));
	}
	// Free up CPU and GPU space
	free(A);
	free(B);
	free(C);
	free(C_ref);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	cudaFree(dC_ref);
	cublasDestroy(handle);

	return 0;
}