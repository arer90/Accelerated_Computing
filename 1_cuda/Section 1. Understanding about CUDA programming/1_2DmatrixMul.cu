/*

https://github.com/skyquasar/Become_professional_in_CUDA

Section 1. Understanding about CUDA programming(Lv1)
phase   1. 2D matrix multiplication (= 2DmatrixMul.cu)  // CPU/GPU/SHARED GPU/cuLBAS (cuBLAS Not Finished.)

*/

#include <iostream>
#include <chrono>
#include <cstdlib>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>  // you must add a library(=cublas.lib) to input part from your project linker.

#pragma once  // This is __syncthreads() function.
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

using namespace std;

#define MAX 5
#define RNUM 10
#define THREADS 32

void CPU_matrixMul(float* matrixA, float* matrixB, float* cpures, int phase) {
	for (int i = 0; i < phase; i++) {
		for (int j = 0; j < phase; j++) {
			for (int k = 0; k < phase; k++) {
				cpures[i*phase + j] += matrixA[i*phase + k] * matrixB[k*phase + j];
			}
		}
	}
}

__global__ void GPU_matrixMul(float *matrixA, float *matirxB, float*res, int phase) {
	int row = threadIdx.y + blockDim.y*blockIdx.y;
	int col = threadIdx.x + blockDim.x*blockIdx.x;
	if (row < phase && col < phase) {
		for (int i = 0; i < phase; i++) {
			res[row*phase + col] += matrixA[i + phase * row] * matirxB[i*phase + col];
		}
	}
}

__global__ void GPU_shared_MMUL(float *matrixA, float * matrixB, float * res, int phase) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int aBegin = phase * THREADS * by;
	int aEnd = aBegin + phase - 1;
	int aStep = THREADS;
	int bBegin = THREADS * bx;
	int bStep = THREADS * phase;
	float Csub = 0;
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep) {
		__shared__ float As[THREADS][THREADS];
		__shared__ float Bs[THREADS][THREADS];
		As[ty][tx] = matrixA[a + phase * ty + tx];
		Bs[ty][tx] = matrixB[b + phase * ty + tx];
		__syncthreads();
#pragma unroll
		for (int k = 0; k < THREADS; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}
		__syncthreads();
	}
	int c = phase * THREADS * by + THREADS * bx;
	res[c + phase * ty + tx] = Csub;

}

int main() {
	int phase = MAX;
	int maxLen = phase * phase;
	srand((unsigned)time(NULL));
	clock_t cpustart, cpuend;

	cout << fixed;
	cout.precision(2);

	float *matrixA, *matrixB, *cpures;
	matrixA = new float[maxLen];
	matrixB = new float[maxLen];
	cpures = new float[maxLen];
	for (int i = 0; i < maxLen; i++) {
		matrixA[i] = rand() % RNUM + 1;
		matrixB[i] = rand() % RNUM + 1;
		cpures[i] = NULL;
	}
	//==========================================
	cout << "_______________MatrixA" << endl;
	for (int i = 0; i < maxLen; i++) {
		if (i%phase == 0) cout << endl;
		cout << matrixA[i] << "	";
	}
	cout << endl;
	cout << "_______________MatrixB" << endl;
	for (int i = 0; i < maxLen; i++) {
		if (i%phase == 0) cout << endl;
		cout << matrixB[i] << "	";
	}
	cout << endl;
	cout << "======================================================" << endl;
	//==========================================
	cpustart = clock();

	CPU_matrixMul(matrixA, matrixB, cpures, phase);

	cpuend = clock();
	double cpums = (double)((double)cpuend - cpustart / CLOCKS_PER_SEC);


	cout << "_______________CPU, 2D Matrix Multiplication." << endl;
	for (int i = 0; i < maxLen; i++) {
		if (i%phase == 0) cout << endl;
		cout << cpures[i] << " ";
	}
	cout << endl << endl;
	//==========================================
	float *dev_matrixA, *dev_matrixB, *dev_res, *dev_res2;
	float *gpures, *gpures2;
	gpures = new float[maxLen];
	gpures2 = new float[maxLen];

	cudaEvent_t start, stop, start2, stop2;
	float gpums, gpums2;
	cudaError(cudaEventCreate(&start));
	cudaError(cudaEventCreate(&stop));
	cudaError(cudaEventCreate(&start2));
	cudaError(cudaEventCreate(&stop2));

	cudaError(cudaMalloc((void**)&dev_matrixA, sizeof(float)*maxLen));
	cudaError(cudaMalloc((void**)&dev_matrixB, sizeof(float)*maxLen));
	cudaError(cudaMalloc((void**)&dev_res, sizeof(float)*maxLen));
	cudaError(cudaMalloc((void**)&dev_res2, sizeof(float)*maxLen));

	cudaError(cudaMemcpy(dev_matrixA, matrixA, sizeof(float)*maxLen, cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(dev_matrixB, matrixB, sizeof(float)*maxLen, cudaMemcpyHostToDevice));

	dim3 thread(THREADS, THREADS);
	dim3 grids((phase + thread.x) / thread.x, (phase + thread.x) / thread.x);

	// global Memory
	cudaEventRecord(start, 0);

	GPU_matrixMul << <grids, thread >> > (dev_matrixA, dev_matrixB, dev_res, phase);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpums, start, stop);

	// Shared Memory
	cudaEventRecord(start2, 0);

	GPU_shared_MMUL << <grids, thread >> > (dev_matrixA, dev_matrixB, dev_res2, phase);

	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&gpums2, start2, stop2);

	cudaError(cudaMemcpy(gpures, dev_res, sizeof(float)*maxLen, cudaMemcpyDeviceToHost));
	cudaError(cudaMemcpy(gpures2, dev_res2, sizeof(float)*maxLen, cudaMemcpyDeviceToHost));

	cout << "_______________GPU, 2D Matrix Multiplication." << endl;
	for (int i = 0; i < maxLen; i++) {
		if (i%phase == 0) cout << endl;
		cout << gpures[i] << " ";
	}
	cout << endl << endl;

	cout << "_______________GPU, 2D Shared Matrix Multiply." << endl;
	for (int i = 0; i < maxLen; i++) {
		if (i%phase == 0) cout << endl;
		cout << gpures2[i] << " ";
	}
	cout << endl << endl;

	//==========================================
	cudaEvent_t custart, custop;
	float cublasms;
	cudaEventCreate(&custart);
	cudaEventCreate(&custop);

	float *cua, *cub, *cures;
	float *cuget = new float[maxLen];
	cudaMalloc((void**)&cua, sizeof(float)*maxLen);
	cudaMalloc((void**)&cub, sizeof(float)*maxLen);
	cudaMalloc((void**)&cures, sizeof(float)*maxLen);

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSetMatrix(phase, phase, sizeof(float), matrixA, phase, cua, phase);
	cublasSetMatrix(phase, phase, sizeof(float), matrixB, phase, cub, phase);

	float al = 1;
	float bl = 1;

	cudaEventRecord(custart, 0);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		phase, phase, phase,
		&al, cua, phase,
		cub, phase,
		&bl, cures, phase);

	cudaEventRecord(custop, 0);
	cudaEventSynchronize(custop);
	cudaEventElapsedTime(&cublasms, custart, custop);

	cublasGetMatrix(phase, phase, sizeof(float), cures, phase, cuget, phase);

	cout << "_______________cuBLAS, 2D Matrix Multiplication." << endl;
	for (int i = 0; i < maxLen; i++) {
		if (i%phase == 0) cout << endl;
		cout << cuget[i] << " ";
	}
	cout << endl << endl;
	cout << endl;
	//==========================================

	cout << fixed;
	cout.precision(7);
	cout << "CPU time duration             : " << cpums << " (ms)." << endl;
	cout << "GPU (GLOBAL) time duration    : " << gpums << " (ms)." << endl;
	cout << "GPU (SHARED) time duration    : " << gpums2 << " (ms)." << endl;
	cout << "cuBLAS time duration          : " << cublasms << " (ms)." << endl;


	cublasDestroy(handle);
	cudaError(cudaFree(cua));
	cudaError(cudaFree(cub));
	cudaError(cudaFree(cures));
	cudaError(cudaFree(dev_matrixA));
	cudaError(cudaFree(dev_matrixB));
	cudaError(cudaFree(dev_res));
	cudaError(cudaFree(dev_res2));
	delete[] gpures;
	delete[] gpures2;
	delete[] matrixA;
	delete[] matrixB;
	delete[] cpures;

	return 0;
}

/*	If you have questions, please email me to 'arer90@gmail.com' or 'arer90@naver.com'	*/
