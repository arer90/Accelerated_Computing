# Accelerated_Computing

## 1_Programming with CUDA
Using examples of CUDA from basic to master level.
Section 0. Basic CUDA
    phase   0. Checking by Visual Sutdio (= 0_basic cuda.cu)
    phase   1. Device Check (= 1_Device_info.cu)
    phase   2. GPU_CPU_differences (= 2_GPU_CPU_differences.cu)   
    ...continue.   
Section 1. Understanding about CUDA programming(Lv1)
    phase   0. Saxpy (= 0_Saxpy.cu)
    phase   1. 2D matrix multiplication (= 1_2DmatrixMul.cu)  // CPU/GPU/SHARED GPU/cuLBAS (Not Finished.)
    phase   2. 2D matrix multiplication (= 2_2D_matrix.cu) // CPU/GPU/SHARED GPU
    ...continue.
if you have a question, please email to me "arer90@naver.com" or "arer90@gmail.com".


-DEVICE INFOMATION-
=====================================================================
Laptop    : GS 63VR 7RF Stealth Pro
Processor : Intel(R) Core(TM) i7-7700HQ CPU @ 2.80 GHz
Memory    : 8.00GB
System    : Windows 10(OS), 64-bit
--------------------------------------------------------------------

Nvidia GPU : "GeForce GTX 1060"
======================================================================
  CUDA Driver Version / Runtime Version          10.1 / 10.0  
  CUDA Capability Major/Minor version number:    6.1  
  Total amount of global memory:                 6144 MBytes (6442450944 bytes)  
  (10) Multiprocessors, (128) CUDA Cores/MP:     1280 CUDA Cores  
  GPU Max Clock rate:                            1671 MHz (1.67 GHz)  
  Memory Clock rate:                             4004 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 1572864 bytes
--------------------------------------------------------------------
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384) 
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536 
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
-------------------------------------------------------------------
  Concurrent copy and kernel execution:          Yes with 5 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            No
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
--------------------------------------------------------------------
if you have a question, please email to me "arer90@naver.com" or "arer90@gmail.com".
