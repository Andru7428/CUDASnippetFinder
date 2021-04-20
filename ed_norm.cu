/*
* ed_norm.cu
* 
* Файл содержит реализацию следующих функций:
* ed_norm - вычисление матрицы евклидовых расстояний между подпоследовательностями временного ряда
*
* Автор: Гоглачев Андрей Игоревич, ЮУрГУ, 2021 год
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template<class T>
__global__ void ed_norm(T* g_idata, float* g_mean, float* g_std, float* g_ed_norm, int m, int size) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < (size - m) && y < (size - m)) {
		int dot = 0;
		for (int k = 0; k < m; k++) {
			dot += g_idata[x + k] * g_idata[y + k];
		}
		g_ed_norm[x * (size - m) + y] = (2 * m * (1 - (dot - m * g_mean[x] * g_mean[y]) / (m * g_std[x] * g_std[y])));
	}
}

template __global__ void ed_norm<int>(int* g_idata, float* g_mean, float* g_std, float* g_ed_norm, int m, int size);