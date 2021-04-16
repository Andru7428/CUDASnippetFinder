/*
* main.cu
*  
* Файл содержит следующие функции:
* scan - кумулятивная сумма элементов временного ряда
* scan_sqr - кумулятивная сумма квадратов элементов временного ряда
* add - добавление конечных сумм каждого блока к результатам суммирования
* mean_std - вычисление средних значений и среднеквадратических отклонений подпоследовательностей
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*
* Кумулятивная сумма элементов временного ряда
* 
* Аргументы
* g_idata - временной ряд
* g_odata - выходной массив кумулятивных сумм
* g_blocksum - выходной массив для конечной суммы блока
*/
template<class T>
__global__ void scan(T* g_idata, T* g_odata, T* g_blocksum) {
	extern __shared__ T s_data[];
	int tid = threadIdx.x;
	unsigned int i = blockDim.x * 2 * blockIdx.x + tid;
	s_data[tid] = g_idata[i];
	s_data[tid + blockDim.x] = g_idata[i + blockDim.x];
	__syncthreads();
	
	int offset = 1;

	//Up-sweep phase
	for (int d = blockDim.x; d > 0; d >>= 1) {
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			s_data[bi] += s_data[ai];
		}
		offset *= 2;
		__syncthreads();
	}
	g_blocksum[blockIdx.x] = s_data[blockDim.x * 2 - 1];
	if (tid == 0) { s_data[blockDim.x * 2 - 1] = 0; }
	
	//Down-sweep phase
	for (int d = 1; d < blockDim.x * 2; d <<= 1) {
		offset >>= 1;
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			T t = s_data[ai];
			s_data[ai] = s_data[bi];
			s_data[bi] += t;
		}
		__syncthreads();
	}
	
	g_odata[i] = s_data[tid];
	g_odata[i + blockDim.x] = s_data[tid + blockDim.x];
}

/*
* Кумулятивная сумма квадратов элементов временного ряда
*
* Аргументы
* g_idata - временной ряд
* g_odata - выходной массив кумулятивных сумм квадратов значений ряда
* g_blocksum - выходной массив для конечной суммы блока
*/
template<class T>
__global__ void scan_sqr(T* g_idata, T* g_odata, T* g_blocksum) {
	extern __shared__ T s_data[];
	int tid = threadIdx.x;
	unsigned int i = blockDim.x * 2 * blockIdx.x + tid;
	int x = g_idata[i];
	s_data[tid] = x * x;
	x = g_idata[i + blockDim.x];
	s_data[tid + blockDim.x] = x * x;
	__syncthreads();

	int offset = 1;

	//Up-sweep phase
	for (int d = blockDim.x; d > 0; d >>= 1) {
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			s_data[bi] += s_data[ai];
		}
		offset *= 2;
		__syncthreads();
	}
	g_blocksum[blockIdx.x] = s_data[blockDim.x * 2 - 1];
	if (tid == 0) { s_data[blockDim.x * 2 - 1] = 0; }

	//Down-sweep phase
	for (int d = 1; d < blockDim.x * 2; d <<= 1) {
		offset >>= 1;
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			T t = s_data[ai];
			s_data[ai] = s_data[bi];
			s_data[bi] += t;
		}
		__syncthreads();
	}

	g_odata[i] = s_data[tid];
	g_odata[i + blockDim.x] = s_data[tid + blockDim.x];
}

/*
* Добавление конечных сумм каждого блока к результатам суммирования
* 
* Аргументы
* g_idata - массив кумулятивных сумм временного ряда
* g_odata - выходной массив кумулятивных сумм
* g_blocksscan - массив с конечными суммами блоков
*/
template<class T>
__global__ void add(T* g_idata, T* g_odata, T* g_blocksscan) {
	int i = blockDim.x * (blockIdx.x + 2) + threadIdx.x;
	g_odata[i] = g_idata[i] + g_blocksscan[blockIdx.x];
}

/*
* Вычисление средних значений и среднеквадратических отклонений подпоследовательностей
* 
* Аргументы
* g_cumsum - массив кумулятивных сумм временного ряда
* g_cumsum_sqr - массив кумулятивных сумм квадратов значений временного ряда
* g_mean - выходной массив средних значений подпоследовательностей ряда
* g_std - выходной массив среднеквадратических отклонений подпоследовательностей ряда
* m - длина подпоследовательности
* size - длина временного ряда
*/
template<class T>
__global__ void mean_std(T* g_cumsum, T* g_sqr_cumsum, float* g_mean, float* g_std, int m, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size - m) {
		float mean = (g_cumsum[i + m] - g_cumsum[i]) / float(m);
		float std = sqrt((g_sqr_cumsum[i + m] - g_sqr_cumsum[i]) / float(m) - mean * mean);
		g_mean[i] = mean;
		g_std[i] = std;
	}
}

int main() {
	//Длина подпоследовательности
	int m = 5;
	int n = 1 << 7;
	int bytes = n * sizeof(int);

	assert(m < n);

	int numThreads = 1 << 5;
	int numBlocks = (n + (numThreads * 2 - 1)) / (numThreads * 2);
	int smemSize = numThreads * sizeof(int) * 2;

	assert(n % numThreads == 0);

	//Выделение памяти на хосте
	int* h_idata = (int*)malloc(bytes);
	int* h_odata = (int*)malloc(bytes);
	float* h_mean = (float*)malloc(sizeof(float) * (n - m));
	float* h_std = (float*)malloc(sizeof(float) * (n - m));

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

	//Выделение памяти на устройстве
	cudaMalloc((void**)&d_idata, bytes);
	cudaMalloc((void**)&d_cumsum, bytes);
	cudaMalloc((void**)&d_cumsum_sqr, bytes);
	cudaMalloc((void**)&d_blocksum, numBlocks * sizeof(int));
	cudaMalloc((void**)&d_mean, sizeof(float) * (n - m));
	cudaMalloc((void**)&d_std, sizeof(float) * (n - m));

	//Копирование исходных данных на устройство
	cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

	scan<int> << <numBlocks, numThreads, smemSize >> > (d_idata, d_cumsum, d_blocksum);
	add<int> << <(numBlocks - 1)  * 2, numThreads >> > (d_cumsum, d_cumsum, d_blocksum);
	scan_sqr<int> << <numBlocks, numThreads, smemSize >> > (d_idata, d_cumsum_sqr, d_blocksum);
	add<int> << <(numBlocks - 1) * 2, numThreads >> > (d_cumsum_sqr, d_cumsum_sqr, d_blocksum);
	mean_std<int> << <numBlocks * 2, numThreads >> > (d_cumsum, d_cumsum_sqr, d_mean, d_std, m, n);

	//Копирование результов на хост 
	cudaMemcpy(h_odata, d_cumsum_sqr, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_mean, d_mean, sizeof(float) * (n - m), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_std, d_std, sizeof(float) * (n - m), cudaMemcpyDeviceToHost);

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

	//Освобождение памяти на хосте
	free(h_idata);
	free(h_odata);
	free(h_mean);
	free(h_std);
}
