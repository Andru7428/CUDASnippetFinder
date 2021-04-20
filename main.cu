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

int main() {
	//Длина подпоследовательности
	int m = 5;
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
	//float* h_ed_norm = (float*)malloc(sizeof(float*) * (n - m) * sizeof(float*) * (n - m));
	//assert(h_ed_norm != NULL);
	int* h_blocksum = (int*)malloc(sizeof(float) * numBlocks);
	assert(h_blocksum != NULL);
	
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
	//float* d_ed_norm = NULL;

	//Выделение памяти на устройстве
	assert(cudaMalloc((void**)&d_idata, bytes) == cudaSuccess);
	assert(cudaMalloc((void**)&d_cumsum, bytes) == cudaSuccess);
	assert(cudaMalloc((void**)&d_cumsum_sqr, bytes) == cudaSuccess);
	assert(cudaMalloc((void**)&d_blocksum, numBlocks * sizeof(int)) == cudaSuccess);
	assert(cudaMalloc((void**)&d_mean, sizeof(float) * (n - m)) == cudaSuccess);
	assert(cudaMalloc((void**)&d_std, sizeof(float) * (n - m)) == cudaSuccess);
	//assert(cudaMalloc((void**)&d_ed_norm, sizeof(float) * (n - m) * sizeof(float) * (n - m)) == cudaSuccess);

	//Копирование исходных данных на устройство
	assert(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice) == cudaSuccess);

	scan<int> << <numBlocks, numThreads, smemSize >> > (d_idata, d_cumsum, d_blocksum, n);
	scan<int> << <1, 32, 32 * sizeof(int) >> > (d_blocksum, d_blocksum, numBlocks);
	add<int> << <numBlocks, numThreads * 2 >> > (d_cumsum, d_cumsum, d_blocksum);
	scan_sqr<int> << <numBlocks, numThreads, smemSize >> > (d_idata, d_cumsum_sqr, d_blocksum, n);
	scan<int> << <1, 32, 32 * sizeof(int) >> > (d_blocksum, d_blocksum, numBlocks);
	add<int> << <numBlocks, numThreads * 2 >> > (d_cumsum_sqr, d_cumsum_sqr, d_blocksum);
	//scan_full<int> << <numBlocks * 2, numThreads, smemSize >> > (d_idata, d_cumsum, d_cumsum_sqr, d_blocksum, n);
	mean_std<int> << <numBlocks * 2, numThreads >> > (d_cumsum, d_cumsum_sqr, d_mean, d_std, m, n);
	//ed_norm<int> << <dim3(numBlocks, numBlocks, 1), numThreads >> > (d_idata, d_mean, d_std, d_ed_norm, m, n);

	//Копирование результов на хост 
	assert(cudaMemcpy(h_odata, d_cumsum, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(cudaMemcpy(h_mean, d_mean, sizeof(float) * (n - m), cudaMemcpyDeviceToHost) == cudaSuccess);
	assert(cudaMemcpy(h_std, d_std, sizeof(float) * (n - m), cudaMemcpyDeviceToHost) == cudaSuccess);
	//assert(cudaMemcpy(h_ed_norm, d_ed_norm, sizeof(float) * (n - m) * sizeof(float) * (n - m), cudaMemcpyDeviceToHost) == cudaSuccess);

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
	}

	//Освобождение памяти на устройстве
	cudaFree(d_idata);
	cudaFree(d_cumsum);
	cudaFree(d_cumsum_sqr);
	cudaFree(d_blocksum);
	cudaFree(d_mean);
	cudaFree(d_std);
	//cudaFree(d_ed_norm);

	//Освобождение памяти на хосте
	free(h_idata);
	free(h_odata);
	free(h_mean);
	free(h_std);
	free(h_blocksum);
	//free(h_ed_norm);
}
