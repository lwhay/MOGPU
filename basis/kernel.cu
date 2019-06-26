#include <iostream>
#include <string.h>
#include <stdio.h>
#include "kernel.h"

__device__ int cnt_atomic_op = 0;
__device__ int *lock_atomic_op;


__global__ void BasicMemcpyKernel(bool parallel, int gpurate, int *d_iarray, int d_iarray_size, int batchsize) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_in_blk = threadIdx.x;
    int iarr_size = d_iarray_size / 2;
    int *iarr_A = d_iarray;
    int *iarr_B = d_iarray + iarr_size;
    long long int start = -1, end;
    if (tid == 0) printf("	125 : %d\n", *(iarr_A + 125));

    start = clock64();
    if (parallel) {
        int anchor = tid;
        while (anchor < iarr_size * sizeof(int) / batchsize) {
            cudaMemcpyAsync(iarr_A + (anchor * batchsize) / sizeof(int), iarr_B + (anchor * batchsize) / sizeof(int),
                            batchsize, cudaMemcpyDeviceToDevice);
            anchor += blockDim.x * gridDim.x;
        }
    } else {
        if (tid == 0) {
            int anchor = 0;
            while (anchor < iarr_size * sizeof(int) / batchsize) {
                cudaMemcpyAsync(iarr_A + (anchor * batchsize) / sizeof(int),
                                iarr_B + (anchor * batchsize) / sizeof(int), batchsize, cudaMemcpyDeviceToDevice);
                anchor++;
            }
        }
    }
    __syncthreads();
    cudaDeviceSynchronize();
    end = clock64();
    if (tid == 0) {
        printf("	125 : %d\n", *(iarr_A + 125));
        printf("	clock: %f ms\n", ((double) (end - start)) / gpurate);
    }
}

__global__ void BasicAtomicOpKernel(bool parallel, int gpurate, int limit) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_in_blk = threadIdx.x;
    long long int start = -1, end;

    start = clock64();
    if (tid == 0) {
        atomicExch(&cnt_atomic_op, 0);
        printf("	cnt_atomic_op : %d\n", cnt_atomic_op);
    }
    __syncthreads();

    int anchor = tid;
    float temp;
    while (anchor < limit) {
        atomicAdd(&cnt_atomic_op, 1);
        //temp = anchor/0.23232f;
        anchor += blockDim.x * gridDim.x;
    }
    __syncthreads();
    cudaDeviceSynchronize();
    end = clock64();

    if (tid == 0) {
        printf("	cnt_atomic_op : %d\n", cnt_atomic_op);
        printf("	clock: %f ms\n", ((double) (end - start)) / gpurate);
    }
}


__global__ void BasicLockKernel(bool parallel, int gpurate, int limit) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_in_blk = threadIdx.x;
    const int tid_w = tid % 32;
    long long int start = -1, end;

    start = clock64();
    if (tid == 0) {
        atomicExch(&cnt_atomic_op, 0);
        lock_atomic_op = new int[32];
        printf("	cnt_atomic_op : %d\n", cnt_atomic_op);
    }
    cudaDeviceSynchronize();
    __syncthreads();

    int anchor = tid;
    int flag;
    while (anchor < limit) {
        while (tid_w == 0) {
            flag = atomicCAS(&lock_atomic_op[tid_w], 0, 1);
            if (flag == 0) {
                atomicAdd(&cnt_atomic_op, 1);
                atomicCAS(&lock_atomic_op[tid_w], 1, 0);
            }
            __syncthreads();
            if (flag == 0)
                break;
        }
        __syncthreads();
        anchor += blockDim.x * gridDim.x;
    }
    __syncthreads();
    cudaDeviceSynchronize();
    end = clock64();

    if (tid == 0) {
        printf("	cnt_atomic_op : %d\n", cnt_atomic_op);
        printf("	clock: %f ms\n", ((double) (end - start)) / gpurate);
    }
}


__global__ void BasicUnBalanceLaneKernel(bool parallel, int gpurate, int *arr, int limit) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_in_blk = threadIdx.x;
    const int tid_w = tid % 32;
    long long int start = -1, end;

    start = clock64();
    if (tid == 0) {
        atomicExch(&cnt_atomic_op, 0);
        printf("	cnt_atomic_op : %d\n", cnt_atomic_op);
    }
    cudaDeviceSynchronize();
    __syncthreads();

    int anchor = tid;
    int tmp;
    while (anchor < limit) {
        tmp = arr[anchor];
        for (int i = 0; i < arr[anchor]; i++) {
            //atomicAdd(&cnt_atomic_op, 1);
            arr[anchor] = arr[anchor] * 23829382 * arr[anchor] / 3333;
        }
        anchor += blockDim.x * gridDim.x;
    }
    __syncthreads();
    cudaDeviceSynchronize();
    end = clock64();

    if (tid == 0) {
        printf("	cnt_atomic_op : %d\n", cnt_atomic_op);
        printf("	clock: %f ms\n", ((double) (end - start)) / gpurate);
    }
}





