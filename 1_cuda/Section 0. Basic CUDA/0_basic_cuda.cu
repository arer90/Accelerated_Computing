//Github - https://github.com/skyquasar/Become_professional_in_CUDA

/*
	Section 0. Basic CUDA.
	phase   0. Checking by viusal studio
	This is test for your device(GPU) can access it with a simple sentence.
	Moreover, it is like a simple test like "Hello world" in C and C++ language.
	By the way, CUDA with Visual Studio is basically made of C and C++ language only.
	Python, matlab, or any other languages are possilbe to compile for using a GPU,
	but for easy understanding of CUDA, I will use only C and C++ language.
	This is made by 'arer90'
*/

// C or C++ header included.
#include <iostream>

// This is basic CUDA header file. Please remember this like
// C -> <stdio.h> or C++ -> <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// namespace for C++ (-iostream header file)
// if you are not similar to C++ language, please follow this sequences.
using namespace std;

// Device(GPU) function : This is function for Device of GPU.
/*
__global__ : this is a main memory in GPU that every CPU values copied to this memory
             and GPU will use this memory for fastest calculation or any progress.
*/

__global__ void print_example() {
	printf("This is test for CUDA language with GPU.\n");
}

int main() {
	
	/*
	This is Device Call while you're using a CUDA.
	Basically, we are using a function like following examples.
	Ex.) 	void         print_example ( int              value           ,.....);
	    function type    function name ( [parameter type] [parameter name],.....);
	
	However, for calling a GPU device, we need a extra words like '<<< >>>'.
	First sampel is <<<1,1>>>
	but you can change number of <<<N,N>>>
	However, if you want to change setup, please click the 'Clean All' from [build(B)] top bar.
	*/
	print_example<<<1,1>>>();

	return 0;
}
