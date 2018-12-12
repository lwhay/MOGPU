/**********************************************************************
* SyncFuncGPU.cu
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Date: Feb 11, 2015 | 3:29:27 PM
* Description:*  
* Licence:*
**********************************************************************/

#include "SyncFuncGPU.cuh"
#include "device/DeviceGlobalVar.cuh"



__device__ void sync_func_query(void)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_b = threadIdx.x;
	__syncthreads();
	if (tid_b == 0)
	{
		atomicAdd(&barrier_query, 1);
		while (barrier_query != gridDim.x)
		{

		}
	}
	if (tid == 0)
	{
		atomicExch(&barrier_query, 0);
	}
	__syncthreads();
}

__device__ void sync_func_update(void)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_b = threadIdx.x;
	__syncthreads();
	if (tid_b == 0)
	{
		atomicAdd(&barrier_update, 1);
		while (barrier_update != gridDim.x)
		{

		}
	}
	if (tid == 0)
	{
		atomicExch(&barrier_update, 0);
	}
	__syncthreads();
}

__device__ void sync_func_dist(void)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_b = threadIdx.x;
	__syncthreads();
	if (tid_b == 0)
	{
		atomicAdd(&barrier_dist, 1);
		while (barrier_dist != gridDim.x)
		{
			;
		}
	}
	if (tid == 0)
	{
		atomicExch(&barrier_dist, 0);
	}
	__syncthreads();
}


