#include <iostream>
#include <string.h>
#include <stdio.h>
#include "kernel.h"
using namespace std;

#define MEMCPY_TEST 0
#define ATOMICOP_TEST 1
#define LOCK_TEST 0
#define UNBALANCE_TEST 0

int get_GPU_Rate(void)
{
	cudaDeviceProp deviceProp;//CUDA定义的存储GPU属性的结构体
	cudaGetDeviceProperties(&deviceProp,0);//CUDA定义函数
	cout<< "wrap size: " << deviceProp.warpSize<< deviceProp.major <<endl;
	return deviceProp.clockRate;
}


int main(void)
{
	cout << "==== Basic Experiments! ====" << endl;
	cudaEvent_t start0, stop0, start1, stop1, start2, stop2;
	float elapsed0, elapsed1, elapsed2;
	int res;
	cudaEventCreate(&start0);
	cudaEventCreate(&stop0);
	int gpurate = get_GPU_Rate();


#if MEMCPY_TEST == 1

	cout << "==== Memcpy Experiments! ====" << endl;
	int* d_iarray;
 	int d_iarray_size = 1024*1024*1024;
	res = cudaMalloc((void **)&d_iarray, d_iarray_size*sizeof(int));
	if (res != 0)
	{
		cout << "Error Point 0!" << endl;
		return 0;
	}


	int* h_iarray = new int[d_iarray_size];
	memset(h_iarray, 0, d_iarray_size*sizeof(int)/2);
	memset(h_iarray + d_iarray_size/2, 1, d_iarray_size*sizeof(int)/2);
	cudaEventRecord(start0);
	cudaMemcpy(h_iarray, h_iarray + d_iarray_size/2, d_iarray_size*sizeof(int)/2, cudaMemcpyHostToHost);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("hostMemcpy %d, time(event record) : %.4f ms\n\n", d_iarray_size/2, elapsed0);
	delete h_iarray;

/*	cudaMemset(d_iarray, 0, d_iarray_size*sizeof(int)/2);
	cudaMemset(d_iarray + d_iarray_size/2, 1, d_iarray_size*sizeof(int)/2);
	cudaEventRecord(start0);
	cudaMemcpy(d_iarray, d_iarray + d_iarray_size/2, d_iarray_size*sizeof(int)/2, cudaMemcpyDeviceToDevice);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("deviceMemcpy %d, time(event record) : %.4f ms\n\n", d_iarray_size/2, elapsed0);
*/


	cudaMemset(d_iarray, 0, d_iarray_size*sizeof(int)/2);
	cudaMemset(d_iarray + d_iarray_size/2, 1, d_iarray_size*sizeof(int)/2);
	cudaEventRecord(start0);
	BasicMemcpyKernel<<<1,1>>>(false, gpurate, d_iarray, d_iarray_size, d_iarray_size/2);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicMemcpyKernel %d, time(event record) : %.4f ms\n\n", d_iarray_size/2, elapsed0);

	
	cudaMemset(d_iarray, 0, d_iarray_size*sizeof(int)/2);
	cudaMemset(d_iarray + d_iarray_size/2, 1, d_iarray_size*sizeof(int)/2);
	cudaEventRecord(start0);
	BasicMemcpyKernel<<<1,1>>>(false, gpurate, d_iarray, d_iarray_size, 1024);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicMemcpyKernel %d, time(event record) : %.4f ms\n\n", 1024, elapsed0);

	cudaMemset(d_iarray, 0, d_iarray_size*sizeof(int)/2);
	cudaMemset(d_iarray + d_iarray_size/2, 1, d_iarray_size*sizeof(int)/2);
	cudaEventRecord(start0);
	BasicMemcpyKernel<<<1,1>>>(false, gpurate, d_iarray, d_iarray_size, 10240);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicMemcpyKernel %d, time(event record) : %.4f ms\n\n", 10240, elapsed0);

	cudaMemset(d_iarray, 0, d_iarray_size*sizeof(int)/2);
	cudaMemset(d_iarray + d_iarray_size/2, 1, d_iarray_size*sizeof(int)/2);
	cudaEventRecord(start0);
	BasicMemcpyKernel<<<1,1>>>(false, gpurate, d_iarray, d_iarray_size, 102400);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicMemcpyKernel %d, time(event record) : %.4f ms\n\n", 102400, elapsed0);

	cudaMemset(d_iarray, 0, d_iarray_size*sizeof(int)/2);
	cudaMemset(d_iarray + d_iarray_size/2, 1, d_iarray_size*sizeof(int)/2);
	cudaEventRecord(start0);
	BasicMemcpyKernel<<<1,1>>>(false, gpurate, d_iarray, d_iarray_size, 1024000);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicMemcpyKernel %d, time(event record) : %.4f ms\n\n", 1024000, elapsed0);


	cudaMemset(d_iarray, 0, d_iarray_size*sizeof(int)/2);
	cudaMemset(d_iarray + d_iarray_size/2, 1, d_iarray_size*sizeof(int)/2);
	cudaEventRecord(start0);
	BasicMemcpyKernel<<<1,1024>>>(true, gpurate, d_iarray, d_iarray_size, 102400);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicMemcpyKernel %d parallel, time(event record) : %.4f ms\n\n", 102400, elapsed0);


	cudaMemset(d_iarray, 0, d_iarray_size*sizeof(int)/2);
	cudaMemset(d_iarray + d_iarray_size/2, 1, d_iarray_size*sizeof(int)/2);
	cudaEventRecord(start0);
	BasicMemcpyKernel<<<1,1024>>>(true, gpurate, d_iarray, d_iarray_size, 1024000);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicMemcpyKernel %d parallel, time(event record) : %.4f ms\n", 1024000, elapsed0);

	cudaFree(d_iarray);
	
#endif

#if ATOMICOP_TEST == 1

	cout << endl << "==== AtomicAdd Experiments! ====" << endl;
	cudaEventRecord(start0);
	BasicAtomicOpKernel<<<1,1024>>>(true, gpurate, 10000000);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicAtomicOpKernel %d parallel, time(event record) : %.4f ms\n", 10000000, elapsed0);

	cout << endl << "==== AtomicAdd Experiments! ====" << endl;
	cudaEventRecord(start0);
	BasicAtomicOpKernel<<<1,1024>>>(true, gpurate, 100000000);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicAtomicOpKernel %d parallel, time(event record) : %.4f ms\n", 100000000, elapsed0);

	cout << endl << "==== AtomicAdd Experiments! ====" << endl;
	cudaEventRecord(start0);
	BasicAtomicOpKernel<<<1,1024>>>(true, gpurate, 1000000000);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicAtomicOpKernel %d parallel, time(event record) : %.4f ms\n", 1000000000, elapsed0);
	
#endif

#if LOCK_TEST == 1

	cout << endl << "==== Busy lock Experiments! ====" << endl;
	cudaEventRecord(start0);
	BasicLockKernel<<<1,1024>>>(true, gpurate, 256);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicLockKernel %d parallel, time(event record) : %.4f ms\n", 256, elapsed0);

#endif

#if UNBALANCE_TEST == 1

	cout << endl << "==== Unbalance lane Experiments! ====" << endl;
 	int arr_size = 1024*1000;
	int* h_arr = new int[arr_size];
	int* d_arr;
	res = cudaMalloc((void **)&d_arr, arr_size*sizeof(int));
	if (res != 0)
	{
		cout << "Error Point 0!" << endl;
		return 0;
	}
	
	for(int i = 0; i<arr_size; i+=32)
	{
		for(int j = i; j<i+32; j++)
		{
			h_arr[j] = 10*(j%32) + 1;
		}
	}
 	cudaMemcpyAsync(d_arr, h_arr, arr_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start0);
	BasicUnBalanceLaneKernel<<<1,1024>>>(true, gpurate, d_arr, arr_size);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicUnBalanceLaneKernel %d INNER-LANE, time(event record) : %.4f ms\n", arr_size, elapsed0);

	for(int i = 0; i<arr_size; i+=arr_size/32)
	{
		for(int j = i; j<i+arr_size/32; j++)
		{
			h_arr[j] = 10*(i/(arr_size/32)) + 1;
		}
	}
 	cudaMemcpyAsync(d_arr, h_arr, arr_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start0);
	BasicUnBalanceLaneKernel<<<1,1024>>>(true, gpurate, d_arr, arr_size);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicUnBalanceLaneKernel %d CROSSE-DLANE wid-unrelated, time(event record) : %.4f ms\n", arr_size, elapsed0);

	
	for(int j = 0; j<1024/32; j++)
	{
		for(int k = 0; k<arr_size/1024; k++)
		{
			for(int i = 0; i<32; i++)
			{
				h_arr[k*1024+j*32+i] = 10*((j*(arr_size/1024)*32+k*32+i)/(arr_size/32)) + 1;
			}
		}
	}
	cudaMemcpyAsync(d_arr, h_arr, arr_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start0);
	BasicUnBalanceLaneKernel<<<1,1024>>>(true, gpurate, d_arr, arr_size);
	cudaEventRecord(stop0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsed0, start0, stop0);
	printf("BasicUnBalanceLaneKernel %d CROSSE-DLANE wid-related, time(event record) : %.4f ms\n", arr_size, elapsed0);

	cudaFree(d_arr);
	delete h_arr;
#endif

	cudaEventDestroy(start0);
	cudaEventDestroy(stop0);
	return 0;
}

