
#ifndef BASIC_KERNEL_H
#define BASIC_KERNEL_H
extern __device__ int cnt_atomic_op;
extern __device__ int* lock_atomic_op;
__global__ void BasicMemcpyKernel(bool parallel, int gpurate, int* d_iarray, int d_iarray_size, int batchsize);

__global__ void BasicAtomicOpKernel(bool parallel, int gpurate, int limit);

__global__ void BasicLockKernel(bool parallel, int gpurate, int limit);

__global__ void BasicUnBalanceLaneKernel(bool parallel, int gpurate, int* arr, int limit);

#endif

