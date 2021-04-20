/*
* mean_std.cu
*
* Файл содержит реализацию следующих функций:
* mean_std - вычисление средних значений и среднеквадратических отклонений подпоследовательностей ряда
* 
* Автор: Гоглачев Андрей Игоревич, ЮУрГУ, 2021 год
*/

#include "mean_std.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* Вычисление средних значений и среднеквадратических отклонений подпоследовательностей ряда
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
__global__ void mean_std(T* g_cumsum, T* g_cumsum_sqr, float* g_mean, float* g_std, int m, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size - m) {
		float mean = (g_cumsum[i + m] - g_cumsum[i]) / float(m);
		float std = sqrt((g_cumsum_sqr[i + m] - g_cumsum_sqr[i]) / float(m) - mean * mean);
		g_mean[i] = mean;
		g_std[i] = std;
	}
}

template __global__ void mean_std<int>(int* g_cumsum, int* g_cumsum_sqr, float* g_mean, float* g_std, int m, int size);