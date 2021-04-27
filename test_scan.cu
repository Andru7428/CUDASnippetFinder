/*
* test_scan.cu
*
* Файл содержит реализацию следующих функций:
* test_scan
*
* Автор: Гоглачев Андрей Игоревич, ЮУрГУ, 2021 год
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "scan.cuh"

void test_scan() {
	int n = 1 << 7;
	int bytes = n * sizeof(int);

	int numThreads = 1 << 5;
	int numBlocks = (n + (numThreads * 2 - 1)) / (numThreads * 2);
	int smemSize = numThreads * sizeof(int) * 2;

	int* h_idata = (int*)malloc(bytes);
	assert(h_idata != NULL);
	int* h_odata = (int*)malloc(bytes);
	assert(h_odata != NULL);
	int* h_cumsum = (int*)malloc(bytes);
	assert(h_cumsum != NULL);

	int sum = 0;
	for (int i = 0; i < n; i++) {
		h_cumsum[i] = sum;
		int a = (rand() & 0xFF);
		h_idata[i] = a;
		sum += a;	
	}

	int* d_idata = NULL;
	int* d_cumsum = NULL;
	int* d_blocksum = NULL;

	cudaError_t result = cudaMalloc((void**)&d_idata, bytes);
	assert(result == cudaSuccess);
	result = cudaMalloc((void**)&d_cumsum, bytes);
	assert(result == cudaSuccess);
	result = cudaMalloc((void**)&d_blocksum, numBlocks * sizeof(int));
	assert(result == cudaSuccess);

	result = cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);

	scan<int> << <numBlocks, numThreads, smemSize >> > (d_idata, d_cumsum, d_blocksum, n);
	scan<int> << <1, 32, 32 * sizeof(int) >> > (d_blocksum, d_blocksum, numBlocks);
	add<int> << <numBlocks, numThreads * 2 >> > (d_cumsum, d_cumsum, d_blocksum);

	result = cudaMemcpy(h_odata, d_cumsum, bytes, cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	for (int i = 0; i < n; i++) {
		assert(h_cumsum[i] == h_odata[i]);
	}

	cudaFree(d_idata);
	cudaFree(d_cumsum);
	cudaFree(d_blocksum);

	free(h_idata);
	free(h_odata);
	free(h_cumsum);

	printf("Scan test is completed");
}