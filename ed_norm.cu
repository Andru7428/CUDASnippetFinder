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
__global__ void ed_norm(T* g_idata, float* g_mean, float* g_std, float* g_ed_norm, int l, int n) {
	int y = blockIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int i = blockId * blockDim.x + threadIdx.x;

	if (x < (n - l)) {
		int dot = 0;
		for (int k = 0; k < l; k++) {
			dot += g_idata[x + k] * g_idata[y + k];
		}
		g_ed_norm[i] = (2 * l * (1 - (dot - l * g_mean[x] * g_mean[y]) / (l * g_std[x] * g_std[y])));
	}
}

template __global__ void ed_norm<int>(int* g_idata, float* g_mean, float* g_std, float* g_ed_norm, int m, int size);