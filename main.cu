/*
* main.cu
*  
* Автор: Гоглачев Андрей Игоревич, ЮУрГУ, 2021 год
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h> 

#include "mean_std.cuh"
#include "scan.cuh"
#include "ed_norm.cuh"
#include "MPdist.cuh"

#include "test_scan.cuh"
#include "test_mean_std.cuh"

int main() {
	//Длина подпоследовательности
	int m = 5;
	int l = 2;
	int n = 1 << 7;
	int bytes = n * sizeof(int);

	int numThreads = 1 << 5;
	int numBlocks = (n + (numThreads * 2 - 1)) / (numThreads * 2);
	int smemSize = numThreads * sizeof(int) * 2;

	//Выделение памяти на хосте
	int* h_idata = (int*)malloc(bytes);
	assert(h_idata != NULL);
	int* h_odata = (int*)malloc(bytes);
	assert(h_odata != NULL);
	float* h_mean = (float*)malloc(sizeof(float) * (n - m));
	assert(h_mean != NULL);
	float* h_std = (float*)malloc(sizeof(float) * (n - m));
	assert(h_std != NULL);
	float* h_ed_norm = (float*)malloc(sizeof(float) * n * sizeof(float) * n);
	assert(h_ed_norm != NULL);
	int* h_blocksum = (int*)malloc(sizeof(float) * numBlocks);
	assert(h_blocksum != NULL);
	float* h_all_pab = (float*)malloc(sizeof(float) * n * sizeof(float) * (n / m + 1));
	assert(h_all_pab != NULL);
	
	//Заполнение входного массива случайными целыми числами до 255
	for (int i = 0; i < n; i++) {
		h_idata[i] = (rand() & 0xFF);
	}

	int* d_idata = NULL;
	int* d_cumsum = NULL;
	int* d_cumsum_sqr = NULL;
	int* d_blocksum = NULL;
	float* d_mean = NULL;
	float* d_std = NULL;
	float* d_ed_norm = NULL;
	float* d_all_pab = NULL;

	//Выделение памяти на устройстве
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
	result = cudaMalloc((void**)&d_ed_norm, sizeof(float) * n * sizeof(float) * n);
	assert(result == cudaSuccess);
	result = cudaMalloc((void**)&d_all_pab, sizeof(float) * n * sizeof(float) * (n / m + 1));
	assert(result == cudaSuccess);

	//Копирование исходных данных на устройство
	result = cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);

	scan<int> << <numBlocks, numThreads, smemSize >> > (d_idata, d_cumsum, d_blocksum, n);
	scan<int> << <1, 32, 32 * sizeof(int) >> > (d_blocksum, d_blocksum, numBlocks);
	add<int> << <numBlocks, numThreads * 2 >> > (d_cumsum, d_cumsum, d_blocksum);
	scan_sqr<int> << <numBlocks, numThreads, smemSize >> > (d_idata, d_cumsum_sqr, d_blocksum, n);
	scan<int> << <1, 32, 32 * sizeof(int) >> > (d_blocksum, d_blocksum, numBlocks);
	add<int> << <numBlocks, numThreads * 2 >> > (d_cumsum_sqr, d_cumsum_sqr, d_blocksum);
	//scan_full<int> << <numBlocks * 2, numThreads, smemSize >> > (d_idata, d_cumsum, d_cumsum_sqr, d_blocksum, n);
	mean_std<int> << <numBlocks * 2, numThreads >> > (d_cumsum, d_cumsum_sqr, d_mean, d_std, m, n);
	ed_norm<int> << <dim3(numBlocks * 2, n - m, 1), numThreads >> > (d_idata, d_mean, d_std, d_ed_norm, m, n);
	find_all_pab << <dim3(numBlocks * 2, n / m + 1, 1), numThreads >> > (d_ed_norm, d_all_pab, m, l, n);

	//Копирование результов на хост 
	result = cudaMemcpy(h_odata, d_cumsum, bytes, cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
	result = cudaMemcpy(h_mean, d_mean, sizeof(float) * (n - m), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
	result = cudaMemcpy(h_std, d_std, sizeof(float) * (n - m), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
	result = cudaMemcpy(h_ed_norm, d_ed_norm, sizeof(float) * n * sizeof(float) * n, cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
	result = cudaMemcpy(h_all_pab, d_all_pab, sizeof(float) * n * sizeof(float) * (n / m + 1), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	for (int i = 0; i < numBlocks; i++) {
		printf("\nBlock %d:\n", i);
		for (int j = 0; j < numThreads * 2; j++) {
			printf("%d ", h_idata[2 * i * numThreads + j]);
		}
		printf("\nScan of block %d:\n", i);
		for (int j = 0; j < numThreads * 2; j++) {
			printf("%d ", h_odata[2 * i * numThreads + j]);
		}
	}

	int k = 0;
	printf("\nScan\n");
	for (int i = 0; i < n; i++) {
		printf("%d ", k);
		k += h_idata[i];
	}

	printf("\nMean:\n");
	for (int i = 0; i < (n - m); i++) {
		printf("%f ", h_mean[i]);
	}

	printf("\nStd:\n");
	for (int i = 0; i < (n - m); i++) {
		printf("%f ", h_std[i]);
	

	for (int i = 0; i < (n - m); i++) {
		printf("\nRow %d\n", i);
		for (int j = 0; j < (n - m); j++) {
			printf("%f ", h_ed_norm[i * n + j]);
		}
	}
	for (int i = 0; i < (n / m + 1); i++) {
		printf("\nRow min %d\n", i);
		for (int j = 0; j < (n - m); j++) {
			printf("%f ", h_all_pab[i * n + j]);
		}
	}
	
	//Освобождение памяти на устройстве
	cudaFree(d_idata);
	cudaFree(d_cumsum);
	cudaFree(d_cumsum_sqr);
	cudaFree(d_blocksum);
	cudaFree(d_mean);
	cudaFree(d_std);
	cudaFree(d_ed_norm);
	cudaFree(d_all_pab);

	//Освобождение памяти на хосте
	free(h_idata);
	free(h_odata);
	free(h_mean);
	free(h_std);
	free(h_blocksum);
	free(h_ed_norm);
	free(h_all_pab);

	test_scan();
	test_mean_std();
}
