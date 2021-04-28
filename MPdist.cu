/*
* MPdist.cu
*
* Файл содержит реализацию следующих функций:
* find_all_pab
*
* Автор: Гоглачев Андрей Игоревич, ЮУрГУ, 2021 год
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "MPdist.cuh"

__global__ void find_all_pab(float* g_ed_norm, float* g_all_pab, int m, int l, int n) {
	int y = blockIdx.y * m;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int i = blockId * blockDim.x + threadIdx.x;

	if (x < (n - l) && y < (n / m) + 1) {
		int min = g_ed_norm[i];

		for (int k = 1; k < (m - l); k++) {
			float a = g_ed_norm[x + y * n];
			if (min < a) min = a;
		}
		g_all_pab[i] = min;
	}
}
