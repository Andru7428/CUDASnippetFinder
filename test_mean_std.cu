/*
* test_mean_std.cu
*
* Файл содержит реализацию следующих функций:
* test_mean_std
*
* Автор: Гоглачев Андрей Игоревич, ЮУрГУ, 2021 год
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h> 

#include "scan.cuh"
#include "mean_std.cuh"

void test_mean_std() {
	int m = 5;
	int n = 1 << 7;
	int bytes = n * sizeof(int);

	int numThreads = 1 << 5;
	int numBlocks = (n + (numThreads * 2 - 1)) / (numThreads * 2);
	int smemSize = numThreads * sizeof(int) * 2;

	int* h_idata = (int*)malloc(bytes);
	assert(h_idata != NULL);
	int* h_cumsum = (int*)malloc(bytes);
	assert(h_cumsum != NULL);
	int* h_cumsum_sqr = (int*)malloc(bytes);
	assert(h_cumsum_sqr != NULL);
	float* h_mean = (float*)malloc(sizeof(float) * (n - m));
	assert(h_mean != NULL);
	float* h_std = (float*)malloc(sizeof(float) * (n - m));
	assert(h_std != NULL);

	for (int i = 0; i < n; i++) {
		h_idata[i] = (rand() & 0xFF);
	}

	int* d_idata = NULL;
	int* d_cumsum = NULL;
	int* d_cumsum_sqr = NULL;
	int* d_blocksum = NULL;
	float* d_mean = NULL;
	float* d_std = NULL;

	cudaError_t result = cudaMalloc((void**)&d_idata, bytes);
	assert(result == cudaSuccess);
	result = cudaMalloc((void**)&d_cumsum, bytes);
	assert(result == cudaSuccess);
	result = cudaMalloc((void**)&d_cumsum_sqr, bytes);
	assert(result == cudaSuccess);
	result = cudaMalloc((void**)&d_blocksum, numBlocks * sizeof(int));
	assert(result == cudaSuccess);
	result = cudaMalloc((void**)&d_mean, sizeof(float) * (n - m));
	assert(result == cudaSuccess);
	result = cudaMalloc((void**)&d_std, sizeof(float) * (n - m));
	assert(result == cudaSuccess);

	result = cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);

	scan<int> << <numBlocks, numThreads, smemSize >> > (d_idata, d_cumsum, d_blocksum, n);
	scan<int> << <1, 32, 32 * sizeof(int) >> > (d_blocksum, d_blocksum, numBlocks);
	add<int> << <numBlocks, numThreads * 2 >> > (d_cumsum, d_cumsum, d_blocksum);
	scan_sqr<int> << <numBlocks, numThreads, smemSize >> > (d_idata, d_cumsum_sqr, d_blocksum, n);
	scan<int> << <1, 32, 32 * sizeof(int) >> > (d_blocksum, d_blocksum, numBlocks);
	add<int> << <numBlocks, numThreads * 2 >> > (d_cumsum_sqr, d_cumsum_sqr, d_blocksum);
	mean_std<int> << <numBlocks * 2, numThreads >> > (d_cumsum, d_cumsum_sqr, d_mean, d_std, m, n);

	result = cudaMemcpy(h_cumsum, d_cumsum, bytes, cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
	result = cudaMemcpy(h_cumsum_sqr, d_cumsum_sqr, bytes, cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
	result = cudaMemcpy(h_mean, d_mean, sizeof(float) * (n - m), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
	result = cudaMemcpy(h_std, d_std, sizeof(float) * (n - m), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	for (int i = 0; i < n - m; i++) {
		float mn = (h_cumsum[i + m] - h_cumsum[i]) / float(m);
		float st = sqrt((h_cumsum_sqr[i + m] - h_cumsum_sqr[i]) / float(m) - mn * mn);
		assert(mn == h_mean[i]);
		assert(st == h_std[i]);
	}

	cudaFree(d_idata);
	cudaFree(d_cumsum);
	cudaFree(d_cumsum_sqr);
	cudaFree(d_blocksum);
	cudaFree(d_mean);
	cudaFree(d_std);

	free(h_idata);
	free(h_cumsum);
	free(h_cumsum_sqr);
	free(h_mean);
	free(h_std);
}

