#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "MPdist.h"
#include "common.h"
#define IDX2F(i, j, n) (i * n + j)
#include <limits>

void computePba(float* d_distance_matrix, int* d_Pba, int n, int m, int l) {
	cublasHandle_t handle;
	cublasCreate(&handle);
	//cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	for (int i = 0; i < n - l; i++) {
		cublasIsamin(handle, m - l + 1, &d_distance_matrix[i * (n - l)], 1, &d_Pba[i]);
		//d_Pba[i] += IDX2F(i, 0, n - l) - 1;
	}
	cublasDestroy(handle);
}

void computePab(float* d_distance_matrix, int* d_Pab, int n, int m, int l) {
	cublasHandle_t handle;
	cublasCreate(&handle);
	for (int i = 0; i < n - l; i++) {
		for (int j = 0; j < m - l; j++) {
			cublasIsamin(handle, m - l + 1, &d_distance_matrix[i * (n - l) + j], 1, &d_Pab[i * (m - l) + j]);
		}
	}
	cublasDestroy(handle);
}

void MPdist(float* d_distance_matrix, float* h_distance_matrix, float* d_mpdist, int n, int m, int l) {
	int* h_Pba = (int*)malloc((n - l) * sizeof(int));
	computePba(d_distance_matrix, h_Pba, n, m, l);

	float* Pba = (float*)malloc((n - l) * sizeof(float));
	for (int i = 0; i < n - l; i++) {
		int idx = h_Pba[i] + i * (n - l) - 1;
		Pba[i] = h_distance_matrix[idx];
	}
	int* h_Pab = (int*)malloc((n - l) * (m - l) * sizeof(int));
	computePab(d_distance_matrix, h_Pab, n, m, l);

	float* Pab = (float*)malloc((n - l) * (m - l) * sizeof(float));
	for (int i = 0; i < n - l; i++) {
		for (int j = 0; j < m - l; j++) {
			int idx = h_Pba[i] + i * (n - l) + j - 1;
			Pab[i * (m - l) + j] = h_distance_matrix[idx];
		}
	}

	for (int i = 0; i < (n - m) ; i++) {
		float* Pabba = (float*)malloc(2 * (m - l) * sizeof(float));
		float min = std::numeric_limits<float>::max();
		for (int j = 0; j < m - l; j++) {
			Pabba[j] = Pba[i + j];
			Pabba[j + m - l] = Pab[i * (m - l) + j];
		}

		for (int j = 0; j < 2 * (m - l); j++) {
			if (Pabba[j] < min) min = Pabba[j];
		}
		d_mpdist[i] = min;
	}
}

