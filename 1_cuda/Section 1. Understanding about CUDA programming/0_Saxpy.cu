//Github - https://github.com/arer90/Accelerated_Computing.git

/*
	Section 1. Understanding about CUDA programming
	phase   0. Saxpy (= Saxpy.cu)	

	This is made by 'arer90'
*/
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>

using namespace std;

#define MAX 100
#define RANDOMNUM 100

void CPU_Saxpy(float*x, float*y, float alpha) {
	for (int i = 0; i < MAX; i++) {
		y[i] = alpha * x[i] + y[i];
	}
}

__global__ void GPU_Saxpy(float *x, float *y, float alpha) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < MAX) {
		y[idx] = alpha * x[idx] + y[idx];
	}
}

void CUBLAS_saxpy(float *alpha, float *beta, float a, float &time) {
	float *cb_alpha, *cb_beta;
	cudaEvent_t init, fin;
	cudaEventCreate(&init);
	cudaEventCreate(&fin);

	cublasHandle_t handle;
	cublasCreate(&handle);

	cudaError(cudaMalloc((void**)&cb_alpha, sizeof(float)*MAX));
	cudaError(cudaMalloc((void**)&cb_beta, sizeof(float)*MAX));
	cublasSetVector(MAX, sizeof(float), alpha, 1, cb_alpha, 1);
	cublasSetVector(MAX, sizeof(float), beta, 1, cb_beta, 1);

	cudaEventRecord(init, 0);

	cublasSaxpy(handle, MAX, &a, cb_alpha, 1, cb_beta, 1);

	cudaEventRecord(fin, 0);
	cudaEventSynchronize(fin);
	cudaEventElapsedTime(&time, init, fin);

	cublasGetVector(MAX, sizeof(float), cb_beta, 1, beta, 1);

	cublasDestroy(handle);
	cudaFree(cb_alpha);
	cudaFree(cb_beta);

}

int main() {
	clock_t cpustart, cpuend;
	double cpums;
	float *line1, *line2, *cpuline, *gpuline, *cublasline;
	srand((unsigned)time(NULL));
	float rnum = rand() % RANDOMNUM + 1;

	line1 = new float[MAX];
	line2 = new float[MAX];
	cpuline = new float[MAX];
	gpuline = new float[MAX];
	cublasline = new float[MAX];
	for (int i = 0; i < MAX; i++) {
		line1[i] = rand() % RANDOMNUM + 1;
		line2[i] = rand() % RANDOMNUM + 1;
		cpuline[i] = line2[i];
		gpuline[i] = line2[i];
		cublasline[i] = line2[i];
	}
	//====================================== Checking
	cout << "Linear 1 values." << endl;
	for (int i = 0; i < MAX; i++) {
		cout << line1[i] << " ";
	}
	cout << endl;
	
	cout << "Linear 2 values." << endl;
	for (int i = 0; i < MAX; i++) {
		cout << line2[i] << " ";
	}
	cout << endl;
	//======================================
	cpustart = clock();

	CPU_Saxpy(line1, cpuline, rnum);

	cpuend = clock();
	cpums = (double)((double)cpuend-cpustart/CLOCKS_PER_SEC);
	
	cout << endl;
	cout << "CPU result........" << endl;
	for (int i = 0; i < MAX; i++) {
		cout << cpuline[i] << " ";
	}
	cout << endl;

	//======================================

	cudaEvent_t start, stop;
	float gpums, cublasms;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

	float *dev_line1, *dev_line2;

	cudaError(cudaMalloc((void**)&dev_line1, sizeof(float)*MAX));
	cudaError(cudaMalloc((void**)&dev_line2, sizeof(float)*MAX));
	cudaError(cudaMemcpy(dev_line1, line1, sizeof(float)*MAX, cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(dev_line2, line2, sizeof(float)*MAX, cudaMemcpyHostToDevice));

	dim3 thread(32);
	dim3 grids((MAX+32)/MAX);
	
	cudaEventRecord(start, 0);
	
	GPU_Saxpy<<<grids, thread>>>(dev_line1, dev_line2, rnum);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpums, start, stop);

	cudaError(cudaMemcpy(gpuline, dev_line2, sizeof(float)*MAX, cudaMemcpyDeviceToHost));

	cout << endl;
	cout << "GPU result........" << endl;
	for (int i = 0; i < MAX; i++) {
		cout << gpuline[i] << " ";
	}
	cout << endl;

	//======================================
	
	CUBLAS_saxpy(line1, cublasline, rnum, cublasms);

	cout << endl;
	cout << "cuBLAS result........" << endl;
	for (int i = 0; i < MAX; i++) {
		cout << cublasline[i] << " ";
	}
	cout << endl;
	cout << endl;
	//======================================
	cout << fixed;
	cout.precision(7);
	cout << "CPU time duration for Saxpy function : " << cpums << " (ms)." << endl;
	cout << "GPU time duration for Saxpy function : " << gpums << " (ms)." << endl;
	cout << "cuBLAS time duration for Saxpy function : " << cublasms << " (ms)." << endl;

	cudaError(cudaFree(dev_line1));
	cudaError(cudaFree(dev_line2));

	delete[] line1;
	delete[] line2;
	delete[] cpuline;
	delete[] gpuline;
	delete[] cublasline;

	return 0;
}
