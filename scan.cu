/*
* scan.cu
* 
* Файл содержит реализацию следующих функций:
* scan - кумулятивная сумма элементов временного ряда
* scan_sqr - кумулятивная сумма квадратов элементов временного ряда
* scan_full - кумулятивная сумма элементов временного ряда и их квадратов
* add - добавление конечных сумм каждого блока к результатам суммирования
*
* Автор: Гоглачев Андрей Игоревич, ЮУрГУ, 2021 год
*/

#include "scan.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* Кумулятивная сумма элементов временного ряда
*
* Парметры шаблона
* T - числовой тип данных временного ряда
* Аргументы
* g_idata - временной ряд
* g_odata - выходной массив кумулятивных сумм
* g_blocksum - выходной массив для конечной суммы блока
* size - длина временного ряда
*/
template<class T>
__global__ void scan(T* g_idata, T* g_odata, T* g_blocksum, int size) {
	extern __shared__ T s_data[];
	int tid = threadIdx.x;
	unsigned int i = 2 * blockDim.x * blockIdx.x + tid;

	if (i < size) {
		s_data[tid] = g_idata[i];
	}
	else {
		s_data[tid] = 0;
	}
	if (i + blockDim.x < size) {
		s_data[tid + blockDim.x] = g_idata[i + blockDim.x];
	}
	else {
		s_data[tid + blockDim.x] = 0;
	}
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
		offset <<= 1;
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

template __global__ void scan<int>(int* g_idata, int* g_odata, int* g_blocksum, int size);

/*
* Кумулятивная сумма элементов временного ряда без сохранения конечных сумм блоков
*
* Парметры шаблона
* T - числовой тип данных временного ряда
* Аргументы
* g_idata - временной ряд
* g_odata - выходной массив кумулятивных сумм
* size - длина временного ряда
*/
template<class T>
__global__ void scan(T* g_idata, T* g_odata, int size) {
	extern __shared__ T s_data[];
	int tid = threadIdx.x;
	unsigned int i = 2 * blockDim.x * blockIdx.x + tid;

	if (i < size) {
		s_data[tid] = g_idata[i];
	}
	else {
		s_data[tid] = 0;
	}
	if (i + blockDim.x < size) {
		s_data[tid + blockDim.x] = g_idata[i + blockDim.x];
	}
	else {
		s_data[tid + blockDim.x] = 0;
	}
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
		offset <<= 1;
		__syncthreads();
	}
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

template __global__ void scan<int>(int* g_idata, int* g_odata, int size);

/*
* Кумулятивная сумма квадратов элементов временного ряда
*
* Парметры шаблона
* T - числовой тип данных временного ряда
* Аргументы
* g_idata - временной ряд
* g_odata - выходной массив кумулятивных сумм
* g_blocksum - выходной массив для конечной суммы блока
* size - длина временного ряда
*/
template<class T>
__global__ void scan_sqr(T* g_idata, T* g_odata, T* g_blocksum, int size) {
	extern __shared__ T s_data[];
	int tid = threadIdx.x;
	unsigned int i = blockDim.x * 2 * blockIdx.x + tid;

	if (i < size) {
		int x = g_idata[i];
		s_data[tid] = x * x;
	}
	else {
		s_data[tid] = 0;
	}
	if (i + blockDim.x < size) {
		int x = g_idata[i + blockDim.x];
		s_data[tid + blockDim.x] = x * x;
	}
	else {
		s_data[tid + blockDim.x] = 0;
	}
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

template __global__ void scan_sqr<int>(int* g_idata, int* g_odata, int* g_blocksum, int size);

/*
* Кумулятивная сумма элементов временного ряда и их квадратов
*
* Парметры шаблона
* T - числовой тип данных временного ряда
* Аргументы
* g_idata - временной ряд
* g_cumsum - выходной массив кумулятивных сумм
* g_cumsum_sqr - выходной массив кумулятивных сумм квадратов
* g_blocksum - выходной массив для конечной суммы блока
* size - длина временного ряда
*/
template<class T>
__global__ void scan_full(T* g_idata, T* g_cumsum, T* g_cumsum_sqr, T* g_blocksum, int size) {
	extern __shared__ T s_data[];
	int tid = threadIdx.x;
	unsigned int i = blockDim.x * 2 * blockIdx.x + tid;
	if (i < size) {
		int x = g_idata[i];
		s_data[tid] = x;
		s_data[tid + blockDim.x] = x * x;
	}
	else {
		s_data[tid] = 0;
		s_data[tid + blockDim.x] = 0;
	}
	__syncthreads();

	int offset = 1;

	//Up-sweep phase
	for (int d = blockDim.x / 2; d > 0; d >>= 1) {
		if (tid < d * 2)
		{
			int ai;
			int bi;
			if (tid < blockDim.x / 2) {
				ai = offset * (2 * tid + 1) - 1;
				bi = offset * (2 * tid + 2) - 1;
			}
			else {
				ai = offset * (2 * tid + 1) - 1 + blockDim.x;
				bi = offset * (2 * tid + 2) - 1 + blockDim.x;
			}
			s_data[bi] += s_data[ai];
		}
		offset *= 2;
		__syncthreads();
	}
	g_blocksum[blockIdx.x] = s_data[blockDim.x * 2 - 1];
	if (tid == 0) { s_data[blockDim.x * 2 - 1] = 0; }

	//Down-sweep phase
	for (int d = 1; d < blockDim.x; d <<= 1) {
		offset >>= 1;
		if (tid < d * 2) {
			int ai;
			int bi;
			if (tid < blockDim.x / 2) {
				ai = offset * (2 * tid + 1) - 1;
				bi = offset * (2 * tid + 2) - 1;
			}
			else {
				ai = offset * (2 * tid + 1) - 1 + blockDim.x;
				bi = offset * (2 * tid + 2) - 1 + blockDim.x;
			}
			T t = s_data[ai];
			s_data[ai] = s_data[bi];
			s_data[bi] += t;
		}
		__syncthreads();
	}

	g_cumsum[i] = s_data[tid];
	g_cumsum_sqr[i] = s_data[tid + blockDim.x];
}

template __global__ void scan_full<int>(int* g_idata, int* g_cumsum, int* g_cumsum_sqr, int* g_blocksum, int size);

/*
* Добавление конечных сумм каждого блока к результатам суммирования
*
* Парметры шаблона
* T - числовой тип данных временного ряда
* Аргументы
* g_idata - массив кумулятивных сумм временного ряда
* g_odata - выходной массив кумулятивных сумм
* g_blocksscan - массив с конечными суммами блоков
*/
template<class T>
__global__ void add(T* g_idata, T* g_odata, T* g_blocksscan) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	g_odata[i] = g_idata[i] + g_blocksscan[blockIdx.x];
}

template __global__ void add<int>(int* g_idata, int* g_odata, int* g_blocksscan);