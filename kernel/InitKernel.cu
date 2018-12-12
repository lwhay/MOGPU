#include<iostream>
#include "device/DeviceGlobalVar.cuh"

__global__ void InitKernel(GConfig *dev_p_gconfig, \
 		ObjBox *d_obs_pool_A, ObjBox *d_obs_pool_B, SIEntry *d_sie_array, Grid *d_index_A, Grid *d_index_B, \
		UpdateCacheArea *d_req_cache_update, QueryCacheArea *d_req_cache_query, \
		CircularQueue *d_queue_bkts_free,  int* d_place_holder , ManagedMemory* d_mm, int* d_map\
	){
	
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid == 0)
	{
		atomicExch(&barrier_dist, 0);
		atomicExch(&launch_signal, 0);

		gp_config = dev_p_gconfig;
		atomicExch(&gp_config_ready, 1);
		
		//printf("%d\n", gp_config->block_analysis_num);
		//printf("Evaluate value of dev_buffer_update: %d %ef %ef %ef %ef %ef\n", dev_buffer_update[0].oid, dev_buffer_update[1].x, \
		//		dev_buffer_update[2].y, dev_buffer_update[3].vx, dev_buffer_update[4].vy, dev_buffer_update[5].time);

		counter_bucket = 0;
		obs_pool_A = d_obs_pool_A;
		obs_pool_B = d_obs_pool_B;
		sie_array = d_sie_array;
		DMM = d_mm;
		
		index_A = d_index_A;
		index_B = d_index_B;
		index_A->block_read = 0;
		index_A->block_write = 0;
		index_B->block_read = 0;
		index_B->block_write = 0;

		index_A->adjustPointer();
		index_B->adjustPointer();

		sec_index_A = new SecIndex();
		sec_index_B = new SecIndex();
		sec_index_A->index = sie_array;
		sec_index_B->index = sie_array + DMM->mms[1].len;
		flag_switch_dist = 0;
		flag_switch_update = 0;
		flag_switch_query = 1;
		flag_switch_version = 0;
		//seg_switch_version = dev_p_gconfig->max_obj_num/dev_p_gconfig->buffer_block_size;
		seg_switch_version = 1;

	//	printf("Evaluate value of obs_pool: %d %f %f %f\n", obs_pool[0].oid, obs_pool[1].x, obs_pool[2].y, \
	//			obs_pool[5].time);

		len_seg_cache_update = dev_p_gconfig->len_seg_cache_update;
		req_cache_update = d_req_cache_update;
		update_map = d_map;
		
		//cudaMemcpyToSymbol(req_cache_update, d_req_cache_update, sizeof(UpdateCacheArea)*size_t(1), size_t(0), cudaMemcpyDeviceToDevice);
		req_cache_query = d_req_cache_query;
		cnt_enqueue_update = 0;
		cnt_enqueue_query = 0;
		exp_new_cell_null = 0;
		exp_old_cell_null = 0;
		exp_update_in_spec_cell = 0;
		atomicExch(&buffer_exhausted, 0);
		atomicExch(&check_tot_covered, 0);
		queue_bkts_free = d_queue_bkts_free;

		place_holder_update = new int[1024*21];//d_place_holder;//need 1024*21; 
		//sync_holder_update = d_place_holder + 1024;
		place_holder_query_dispatch= d_place_holder + 1024*23;
		sync_holder_query_dispatch= d_place_holder + 1024*24;
		cache_memory_idx_query_dispatch = d_place_holder + 1024*25;
		place_holder_query = d_place_holder + 1024*26;
		place_holder_update_dispatch = d_place_holder+1024*27;//need 1024*5;
		
//#if (USE_MULTIQUEUE == 1)
		multiqueue = d_place_holder + 1024*35;
//#endif

	}
	__threadfence_system();
	__syncthreads();
}
