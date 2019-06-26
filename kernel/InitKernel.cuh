/*************************************************************************
	> File Name: BusyKernel.h
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: Sun 09 Jul 2017 03:28:01 PM CST
 ************************************************************************/

#include<iostream>
#include "device/DeviceGlobalVar.cuh"

__global__ void InitKernel(GConfig * dev_p_gconfig, \
        ObjBox * d_obs_pool_A, ObjBox * d_obs_pool_B, SIEntry * d_sie_array, Grid * d_index_A, Grid * d_index_B, \
        UpdateCacheArea * d_req_cache_update, QueryCacheArea * d_req_cache_query, \
        CircularQueue * d_queue_bkts_free, int * d_place_holder, ManagedMemory * d_mm, int * d_map);
