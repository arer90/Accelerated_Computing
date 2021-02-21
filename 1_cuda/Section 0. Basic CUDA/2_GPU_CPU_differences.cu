//Github - https://github.com/arer90/Accelerated_Computing.git

/*
	Section 0. Basic CUDA.
	phase   2. GPU_CPU_differences
	
	I made this code for checking the difference between CPU(=Host) and GPU(=Device).
	Moreover, we can see the simple example such as Vector addition but I want to describe understandable details.
	Following code is called "Saxpy Operation."
	equation will be "   y = alpha*x + y,  (alpha is random nuber, x and y are inputs)"

	if you want to learn about this more, following links will help you understand.
	https://devblogs.nvidia.com/six-ways-saxpy/
	https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_1

	This is made by 'arer90'
*/
#include <iostream>		// if you are using a C language, you can use '<stdio.h>'
#include <cstdlib>		// if you are using a C language, you can use '<stdlib.h>'
#include <chrono>		// if you are using a C language, you can use '<time.h>'

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;	// if you are using a C language, you can ignore or delete this section.

#define MAX 10			// you can see if you change this number

/*
	Following function(=CPU_Saxpy(int*x, int*y,int alpha)) is basic function format usually people made of.
*/
void CPU_Saxpy(int*x, int*y,int alpha) {
	for (int i = 0; i < MAX; i++) {
		y[i] = alpha * x[i] + y[i];
	}
}

/*
	However, Next function(=GPU_Saxpy(int*x, int*y, int alpha)) is different than CPU_Saxpy.
	There is no "for" function because of thanks to the ability of GPU's parallel calculation.
	For more detail, CPU is sequence calculation like step-by-step but GPU is starting diverse calculation in same time.
	So we have to make a code to this right.

	"idx" is similar to "int i" in the CPU_Saxpy's for operation.
	However, we have to find out what is meaning of "threadIdx.x", "blockDIm.x" and "blockIdx.x".
		"threadIdx" is smallest size of operation in GPU and this value have a 3 dimentional ways such as (x,y,z).
		"blockDim" is showing how many threads in one block from user's input. you can change the number but there is limited from hardware ability.
		This is also have a 3 dimentional ways such as (x,y,z)
		"blockIdx" is number of block and like above description this is also having a 3 dimentional ways such as (x,y,z).

		For example, if we calculate 10 times with "<<<2,5>>>".  <---- simply think like this <<<[blockDim],[threadIdx]>>> in CPU code.
		In GPU operation, calculations will be like this.
		==============================================================================
		
		<<< 2(=blockDim.x), 5(=threadIdx.x)>>> ---in CPU code
		
		Calculation times : 10
		[times]		=	[threadIdx.x] + [blockDim.x]*[blockIdx.x];
		================================================================ BLOCK 0	
			0		=		0		  +			5	*	0
			1		=		1		  +			5	*	0
			2		=		2		  +			5	*	0
			3		=		3		  +			5	*	0
			4		=		4		  +			5	*	0
		================================================================ BLOCK 1
			5		=		0		  +			5	*	1
			6		=		1		  +			5	*	1
			7		=		2		  +			5	*	1
			8		=		3		  +			5	*	1
			9		=		4		  +			5	*	1
		================================================================ BLOCK 1

	Back to description, "if(idx<MAX)" is similar to "i < MAX" in CPU_Saxpy function in for operation.
*/

__global__ void GPU_Saxpy(int*x, int*y, int alpha) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < MAX) {
		y[idx] = alpha * x[idx] + y[idx];
	}
}

void func() {
	//================================================================ time setup
	clock_t cpustart, cpuend;
	double cpums;
	float gpums;

	//================================================================ declare values
	int *array1, *array2, *gpuarray;
	int rnum = rand()%100+1;
	srand((unsigned)time(NULL));

	cout << "This is GPU and CPU operating time test" << endl << endl;;
	cout << "Count = " << MAX << ", Random number = " << rnum << endl<<endl;
	cout << "=================================================" << endl;

	//================================================================ Memory Alloc
	array1 = new int[MAX];
	array2 = new int[MAX];
	gpuarray = new int[MAX];

	//Following codes for C languages.
	/*array1 = (int*)malloc(sizeof(int)*MAX);
	array2 = (int*)malloc(sizeof(int)*MAX);
	gpuarray = (int*)malloc(sizeof(int)*MAX);*/

	for (int i = 0; i < MAX; i++) {
		array1[i] = rand() % 100 + 1;
		array2[i] = rand() % 100 + 1;
		gpuarray[i] = array2[i];
	}

	//================================================================ CPU Operations
	cpustart = clock();
	
	CPU_Saxpy(array1, array2, rnum);

	cpuend = clock();
	cpums = (double)((double)cpuend - cpustart / CLOCKS_PER_SEC);

	cout << "After CPU Saxpy operation..." << endl;
	for (int i = 0;i<MAX; i++) {
		cout << array2[i] << " ";
	}
	cout << endl;
	cout << fixed;
	cout.precision(7);
	cout << "		*****CPU Saxpy time duration = " << cpums << " (ms)." << endl<<endl;

	//================================================================ GPU Operations
	cudaEvent_t start, stop;			// this GPU time values
	cudaEventCreate(&start);			// you have to declare as cudaEvent for making a time duration.
	cudaEventCreate(&stop);		
	
	/*
	If you want to use GPU, you have to make a values for GPU like following sequences.
	Making a similar type of value with new name and using "cudaMalloc" and "cudaMemcpy".
	"cudaMalloc" is similar to malloc in C and decleartion is simple like
			     = cudaMalloc((void**)&[name of new value], 
				                       [size of this value]);

	"cudaMemcpy" is similar to memcpy in C and decleartion is simple like
			     = cudaMemcpy([new value from cudaMalloc],
							  [value from original one],
							  [size of value],
							  [cudaMemcpyHostToDeivce or cudaMemcpyDeviceToHost]);
				 ps. "cudaMemcpyHostToDevice" is transfering value from cudaMemcpy to GPU's global memory.
					 "cudaMemcpyDeviceToHost" is transfering value from global memory to CPU's memory.
	*/

	int *dev_array1, *dev_array2;
	cudaError(cudaMalloc((void**)&dev_array1, sizeof(int)*MAX));			
	cudaError(cudaMalloc((void**)&dev_array2, sizeof(int)*MAX));
	cudaError(cudaMemcpy(dev_array1, array1, sizeof(int)*MAX, cudaMemcpyHostToDevice));
	cudaError(cudaMemcpy(dev_array2, gpuarray, sizeof(int)*MAX, cudaMemcpyHostToDevice));

	// you can make a thread and block easily by following value type "dim3".
	dim3 threads(32);
	dim3 grids( (MAX+32)/32 );

	cudaEventRecord(start, 0);

	GPU_Saxpy<<<grids,threads>>>(dev_array1, dev_array2, rnum);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpums, start, stop);

	cudaError(cudaMemcpy(gpuarray, dev_array2, sizeof(int)*MAX, cudaMemcpyDeviceToHost));

	cout << "After GPU Saxpy operation..." << endl;
	for (int i = 0; i < MAX; i++) {
		cout << gpuarray[i] << " ";
	}
	cout << endl;
	cout << "		*****GPU Saxpy time duration = " << gpums << " (ms)." << endl << endl;

	cudaError(cudaDeviceReset());
	cudaError(cudaFree(dev_array1));
	cudaError(cudaFree(dev_array2));

	delete[] array1;
	delete[] array2;
	delete[] gpuarray;
	/*free(array1);
	free(array2);
	free(gpuarray);*/

	cout << "RESULT : " << endl;
	cout << "GPU time : " << cpums << " (ms)." << endl;
	cout << "GPU time : " << gpums << " (ms)." << endl << endl;
}

int main() {
	clock_t start, end;
	start = clock();
	func();
	end = clock();
	printf("=======================================================\nAll function time duration = %.7lf ms.\n", (double)((double)end - start / CLOCKS_PER_SEC));
	return 0;
}
