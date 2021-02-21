//Github - https://github.com/skyquasar/Become_professional_in_CUDA

/*
	Section 0. Basic CUDA.
	phase   1. Device Check
	This code is for checking your device properties such as basic hardware setting.
	If you want to know about more or learn about deatil, please open the sample of CUDA from Nvidia.
	This is made by 'arer90'
*/

// C or C++ header included.
#include <iostream>
// This is basic CUDA header file.
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

int main() {
	int deviceCount = 0;
	// CudaError() function is checking your error from CUDA functions or values
	/*
	cudaGetDeviceCount(&(int value)) is showing your GPU devices.
	if you are using a SLI for making a 2 or more GPU devices to operate for powerful result,
	this function will help you to visualize your device count.
	*/
	cudaError(cudaGetDeviceCount(&deviceCount));

	if (deviceCount == 0) {
		cout << "There is no GPU device in this hardware." << endl;
	}
	else {
		cout << "	There is ["<< deviceCount<<"] GPU device in this hardware." << endl;

		for (int dev = 0; dev < deviceCount; dev++) {
			/*
				'Dev' is showing the GPU device hardware pin number
				'cudaSetDevice' is setting up the device from Dev.
				'cudaDeviceProp' is structs for GPU hardware instruction.
				'cudaGetDeviceProperties(&struct cudaDeviceProp, Dev)' 
				   is copying the data from GPU info to this function
			*/			
			cudaSetDevice(dev);
			cudaDeviceProp devinfo;
			cudaGetDeviceProperties(&devinfo, dev);

			cout << "		Device["<<dev<<"] :  "<< devinfo.name << endl<<endl;
			cout << "		Device[" << dev << "], Clock_Rate = " << devinfo.clockRate << endl;
			cout << "		Device[" << dev << "], L2 Cache Size = " << devinfo.l2CacheSize << endl;
			cout << "		Device[" << dev << "], Max Grid Size = " << devinfo.maxGridSize << endl;
			cout << "		Device[" << dev << "], Max Threads Dim = " << devinfo.maxThreadsDim << endl;
			cout << "		Device[" << dev << "], Max ThreadsPerBlock = " << devinfo.maxThreadsPerBlock << endl;
			cout << "		Device[" << dev << "], Max ThreadsPerMultiProcessor = " << devinfo.maxThreadsPerMultiProcessor << endl;
			cout << "		Device[" << dev << "], Memory BusWidth = " << devinfo.memoryBusWidth << endl;
			cout << "		Device[" << dev << "], Memory ClockRate = " << devinfo.memoryClockRate << endl;
			cout << "		Device[" << dev << "], MemPitch = " << devinfo.memPitch << endl;
			cout << "		Device[" << dev << "], Registers Per Block = " << devinfo.regsPerBlock << endl;
			cout << "		Device[" << dev << "], Shared Memory Per Block = " << devinfo.sharedMemPerBlock << endl;
			cout << "		Device[" << dev << "], warpSize = " << devinfo.warpSize << endl;


			//This is for C language. if you want to delete C language annotation, drag the C language parts and press the key 'Ctrl+K+U'.
			/*printf("		Device[%d] :  %s\n\n", dev, devinfo.name);
			printf("		Device[%d], Clock_Rate = %d\n", dev, devinfo.clockRate);
			printf("		Device[%d], L2 Cache Size = %d\n", dev, devinfo.maxGridSize);
			printf("		Device[%d], Max Grid Size = %d\n", dev, devinfo.maxThreadsDim);
			printf("		Device[%d], Max Threads Dim = %d\n", dev, devinfo.maxThreadsPerBlock);
			printf("		Device[%d], Max ThreadsPerBlock = %d\n", dev, devinfo.maxThreadsPerMultiProcessor);
			printf("		Device[%d], Memory BusWidth = %d\n", dev, devinfo.memoryBusWidth);
			printf("		Device[%d], Memory ClockRate = %d\n", dev, devinfo.memoryClockRate);
			printf("		Device[%d], MemPitch = %d\n", dev, devinfo.memPitch);
			printf("		Device[%d], Registers Per Block = %d\n", dev, devinfo.regsPerBlock);
			printf("		Device[%d], Shared Memory Per Block = %d\n", dev, devinfo.sharedMemPerBlock);
			printf("		Device[%d], warpSize = %d\n", dev, devinfo.warpSize);*/
		}
	}
	return 0;
}
