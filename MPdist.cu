#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "MPdist.h"
#include "common.h"
#define IDX2F(i, j, n) (i * n + j)
#include <limits>


void MPdist_(float* d_distance_matrix, float* d_profile, int n, int m, int l, int idx) {
	float* d_Pab;
	cudaMalloc(&d_Pab, l * (n - l) * sizeof(float));
	float* d_Pba;
	cudaMalloc(&d_Pba, l * sizeof(float));

	computePab_<<<dim3(n - l, l), 256, l * sizeof(float) / 2>>>(d_distance_matrix, d_Pab, n - l, l);
	computePba_<<<n, 256, l * sizeof(float) / 2 >>>(d_distance_matrix, d_Pba, n - l, l);
	computeMPdist<<<n - l, 256, 2 * l * sizeof(float)>>>(d_Pab, d_Pba, d_profile, n - l, l, idx);
}

__global__ void computeMPdist(float* d_Pab, float* d_Pba, float* d_MPdist, int n, int l, int idx) {
	unsigned int tid = threadIdx.x;
	extern __shared__ float sdata[];
	if (tid < l / 2) {
		sdata[tid] = d_Pab[tid + l * blockIdx.x];
		sdata[tid + l] = d_Pba[tid + blockIdx.x];
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2) {
		if (tid < stride && tid + stride < l) {
			sdata[tid] = min(sdata[tid], sdata[tid + stride]);
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpReduce(sdata, tid, l);
	}

	if (tid == 0) {
		d_MPdist[blockIdx.x + idx * (n - 2 * l)] = sdata[0];
	}
}

__global__ void precompute_min_Pab(float* d_distance_matrix, int* d_Pab, int n, int l) {
	unsigned int tid = threadIdx.x;
	extern __shared__ float sdata[];

	if (tid < l / 2) {
		sdata[tid] = min(d_distance_matrix[blockIdx.x * n + tid], d_distance_matrix[blockIdx.x * n + tid + l / 2]);
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2) {
		if (tid < stride && tid + stride < l / 2) {
			sdata[tid] = min(sdata[tid], sdata[tid + stride]);
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpReduce(sdata, tid, l);
	}

	if (tid == 0) {
		printf("%d: %f\n", blockIdx.x, sdata[0]);
	}
}

__global__ void computePab_(float* d_distance_matrix, float* d_Pab, int n, int l) {
	unsigned int tid = threadIdx.x;
	extern __shared__ float sdata[];

	if (tid < l / 2) {
		sdata[tid] = min(d_distance_matrix[blockIdx.y * n + tid + blockIdx.x], d_distance_matrix[blockIdx.y * n + tid + l / 2 + blockIdx.x]);
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2) {
		if (tid < stride && tid + stride < l / 2) {
			sdata[tid] = min(sdata[tid], sdata[tid + stride]);
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpReduce(sdata, tid, l);
	}

	if (tid == 0) {
		d_Pab[l * blockIdx.x + blockIdx.y] = sdata[0];
	}
}

__global__ void computePba_(float* d_distance_matrix, float* d_Pba, int n, int l) {
	unsigned int tid = threadIdx.x;
	extern __shared__ float sdata[];

	if (tid  < l / 2) {
		sdata[tid] = min(d_distance_matrix[tid * n + blockIdx.x], d_distance_matrix[(tid + l / 2) * n + blockIdx.x]);
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2) {
		if (tid < stride && tid + stride < l / 2) {
			sdata[tid] = min(sdata[tid], sdata[tid + stride]);
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpReduce(sdata, tid, l);
	}

	if (tid == 0) {
		d_Pba[blockIdx.x] = sdata[0];
	}
}

__device__ void warpReduce(volatile float* sdata, unsigned int tid, int l)
{
	sdata[tid] = (tid + 32 < l / 2) ? min(sdata[tid], sdata[tid + 32]) : sdata[tid];
	sdata[tid] = min(sdata[tid], sdata[tid + 16]);
	sdata[tid] = min(sdata[tid], sdata[tid + 8]);
	sdata[tid] = min(sdata[tid], sdata[tid + 4]);
	sdata[tid] = min(sdata[tid], sdata[tid + 2]);
	sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}