/*
//Github - https://github.com/arer90/Accelerated_Computing.git

Section 1. Understanding about CUDA programming(
phase   2. 2D matrix multiplication (= 2D_matrix.cu) // CPU/GPU/SHARED GPU

*/

#pragma once	
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <iostream>
#include <cstdlib>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define MAX 5
#define RANDOM_NUM 10
#define THREADS 32

void CPU_matrixMUL(float *A, float *B, float *C, int num) {
	for (int i = 0; i < num; i++) {
		for (int j = 0; j < num; j++) {
			for (int k = 0; k < num; k++) {
				C[i*num + j] += A[i*num + k] * B[k*num + j];
			}
		}
	}
}

__global__ void GPU_matrixMUL(float *A, float * B, float * C, int num) {
	int col = threadIdx.x + blockIdx.x*blockDim.x;
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	if (col < num && row < num) {
		for (int i = 0; i < num; i++) {
			C[row*num + col] += A[i + row * num] * B[i*num + col];
		}
	}
}

__global__ void GPU_matrixMUL2(float *A, float * B, float * C, int num) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int aBegin = num * THREADS * by;
	int aEnd = aBegin + num - 1;
	int aStep = THREADS;
	int bBegin = THREADS * bx;
	int bStep = THREADS * num;
	float Csub = 0;
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {
		__shared__ float As[THREADS][THREADS];
		__shared__ float Bs[THREADS][THREADS];
		As[ty][tx] = A[a + num * ty + tx];
		Bs[ty][tx] = B[b + num * ty + tx];
		__syncthreads();
#pragma unroll
		for (int k = 0; k < THREADS; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}
		__syncthreads();
	}
	int c = num * THREADS * by + THREADS * bx;
	C[c + num * ty + tx] = Csub;

}


int main() {
	int num = MAX;
	size_t length = num * num;
	clock_t cpustart, cpuend;
	float cpums;
	
	cout << "MatrixMultiply = [" << num << "],[" << length <<"]"<< endl;
	float *matrixA = new float[length];
	float *matrixB = new float[length];
	float *CpuRes = new float[length];

	srand((unsigned)time(NULL));
	for (int i = 0; i < length; i++) {
		
		matrixA[i] = rand() % RANDOM_NUM + 1;
		matrixB[i] = rand() % RANDOM_NUM + 1;
		CpuRes[i] = NULL;
	}

	printf("----MatrixA Values----");
	for (int i = 0; i < length; i++) {
		if (i%num == 0) printf("\n");
		printf("%3.1f ", matrixA[i]);
	}
	printf("\n");
	printf("----MatrixB Values----");
	for (int i = 0; i < length; i++) {
		if (i%num == 0) printf("\n");
		printf("%3.1f ", matrixB[i]);
	}
	printf("\n");

	cpustart = clock();
	CPU_matrixMUL(matrixA, matrixB, CpuRes, num);
	cpuend = clock();

	cpums = (float)((float)cpuend - cpustart / CLOCKS_PER_SEC);
	printf("----CPU_result Values----");
	for (int i = 0; i < length; i++) {
		if (i%num == 0) printf("\n");
		printf("%3.1f ", CpuRes[i]);
	}
	printf("\n");

	cudaEvent_t gstart, gend, g2start, g2end;
	cudaError(cudaEventCreate(&gstart));
	cudaError(cudaEventCreate(&gend));
	cudaError(cudaEventCreate(&g2start));
	cudaError(cudaEventCreate(&g2end));

	size_t deviceQuan = sizeof(float)*length;
	float *dev_matA, *dev_matB, *dev_res, *dev_res2;
	float *writing = new float[length];
	float *writing2 = new float[length];
	float gpums1, gpums2;
	cudaError(cudaMalloc((void**)&dev_matA, deviceQuan));
	cudaError(cudaMalloc((void**)&dev_matB, deviceQuan));
	cudaError(cudaMalloc((void**)&dev_res, deviceQuan));
	cudaError(cudaMalloc((void**)&dev_res2, deviceQuan));

	cudaError(cudaMemcpy(dev_matA, matrixA, deviceQuan, cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(dev_matB, matrixB, deviceQuan, cudaMemcpyHostToDevice));

	dim3 Threads(THREADS,THREADS);
	dim3 Grids((num + Threads.x-1) / Threads.x, (num + Threads.y - 1) / Threads.y);
	//dim3 Grids((num + Threads.x) / Threads.x, (num + Threads.y) / Threads.y);
	//dim3 Grids(32, 32);

	cudaError(cudaEventRecord(gstart,0));
	GPU_matrixMUL << <Grids, Threads >> > (dev_matA, dev_matB, dev_res, num);
	cudaError(cudaEventRecord(gend,0));
	cudaError(cudaEventSynchronize(gend));
	cudaError(cudaEventElapsedTime(&gpums1, gstart, gend));

	cudaError(cudaEventRecord(g2start, 0));
	GPU_matrixMUL2<< <Grids, Threads >> >(dev_matA, dev_matB, dev_res2, num);
	cudaError(cudaEventRecord(g2end, 0));
	cudaError(cudaEventSynchronize(g2end));
	cudaError(cudaEventElapsedTime(&gpums2, g2start, g2end));


	cudaError(cudaDeviceSynchronize());
	cudaError(cudaMemcpy(writing, dev_res, deviceQuan, cudaMemcpyDeviceToHost));
	cudaError(cudaMemcpy(writing2, dev_res2, deviceQuan, cudaMemcpyDeviceToHost));
	
	printf("----GPU_result1 Values----");
	for (int i = 0; i < length; i++) {
		if (i%num == 0) printf("\n");
		printf("%3.1f ", writing[i]);
	}
	printf("\n");
	printf("----GPU_result2 Values----");
	for (int i = 0; i < length; i++) {
		if (i%num == 0) printf("\n");
		printf("%3.1f ", writing2[i]);
	}
	printf("\n");
	printf("\n");

	printf("CPU Matrix(Original) [time duration : %3.7f ] final number : %f\n", cpums, CpuRes[length-1]);
	printf("GPU Matrix(GLOBAL)   [time duration : %3.7f ] final number : %f\n", gpums1, writing[length-1]);
	printf("GPU Matrix(SHARED)   [time duration : %3.7f ] final number : %f\n", gpums2, writing2[length-1]);


	cudaError(cudaFree(dev_matA));
	cudaError(cudaFree(dev_matB));
	cudaError(cudaFree(dev_res));
	cudaError(cudaFree(dev_res2));
	delete[] matrixA;
	delete[] matrixB;
	delete[] CpuRes;
	delete[] writing;		
	delete[] writing2;
	return 0;
}

/*	if you have a question, please email to me "arer90@naver.com" or "arer90@gmail.com".	*/
