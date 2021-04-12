#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

template<class T>
__global__ void scan(T* g_idata, T* g_odata) {
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

int main() {
	int n = 1 << 7;
	int numThreads = 1 << 5;
	int numBlocks = (n + (numThreads * 2 - 1)) / (numThreads * 2);

	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	int smemSize = numThreads * sizeof(int) * 2;

	int bytes = n * sizeof(int);
	int* h_idata = (int*)malloc(bytes);
	for (int i = 0; i < n; i++) {
		h_idata[i] = (rand() & 0xFF);
	}
	int* h_odata = (int*)malloc(bytes);

	int* d_idata = NULL;
	int* d_odata = NULL;

	cudaMalloc((void**)&d_idata, bytes);
	cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_odata, bytes);

	scan<int> << <dimGrid, dimBlock, smemSize >> > (d_idata, d_odata);
	cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost);

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

	cudaFree(d_idata);
	cudaFree(d_odata);
	free(h_idata);
	free(h_odata);
}
