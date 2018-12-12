/**********************************************************************
* UpdateKernel.cu
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Oct 24, 2014 | 09:48:11 AM
* Description:*  
* Licence:*
**********************************************************************/

#include <stdio.h>
#include <math.h>
#include "device/DeviceGlobalVar.cuh"
#include "UpdateKernel.cuh"
#include "misc/SyncFuncGPU.cuh"

__global__ void UpdateKernel(GConfig *dev_p_gconfig)
{
	//printf("update kernel inside");
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid_b = threadIdx.x;
	const unsigned int barrier_step = gridDim.x;
	
	unsigned int barrier_fence = 0;
	//__shared__ int place_holder[512];
	//__shared__ int sync_holder[512];

	int bkts_per_cell = 0;
	int obs_per_cell = 0;
	long long int start = -1, end;
	long long int start2 = -1, end2;
	//long long int sumtime = 0;
	//long long int copytime = 0;

	
	GConfig *p_config = dev_p_gconfig;//gp_config;
	const int QUERY_SKIP_NUM = p_config->query_skip_round_num * p_config->max_obj_num/p_config->buffer_block_size;
	const int len_bkt = p_config->max_bucket_len;
	//int len_bkt = 800;
	if (tid == 0)
	{
		//atomicExch(&exp_cell_full, 0);
		//atomicExch(&exp_cell_empty, 0);
		//atomicExch(&exp_anchor_x_cell, 0);
	}

	Grid *p_grid_local = NULL;
	SecIndex *p_seci_local = NULL;
	ObjBox *obs_pool_local;
	int anchor = 0,anchor2 = 0;

	while (launch_signal == 0);
	

	int tot_cells = p_config->edge_cell_num * p_config->edge_cell_num;
	//int tot_cells= 16384;
	int tot_vgroup_update = p_config->side_len_vgroup * p_config->side_len_vgroup;
	int i_temp;
	//int tot_vgroup_update = 16384;
	int len_vgroup = tot_cells / tot_vgroup_update;

	//lwh32->1
	int WRAPSIZE =32;
	if(p_config->buffer_block_size/(double)tot_cells < 2.0f)
		WRAPSIZE = 1;
	else if(p_config->buffer_block_size/(double)tot_cells < 3.0f)
		WRAPSIZE = 2;
	else if(p_config->buffer_block_size/(double)tot_cells < 4.0f)
		WRAPSIZE = 4;
	else if(p_config->buffer_block_size/(double)tot_cells < 7.0f)
		WRAPSIZE = 8;
	else if(p_config->buffer_block_size/(double)tot_cells < 15.0f)
		WRAPSIZE = 16;
	//WRAPSIZE = 32;
	
	
	if(tid == 0)
		printf("u_wrapsize: %d \n", WRAPSIZE);	
	int realWrapSize = 0;
	const int wid = tid / WRAPSIZE;
	const int tid_w = tid % WRAPSIZE;
	
	if (tid == 0)
	{
		node_dequeue_update = NULL;
		//cnt_dequeue_update = 0;
	}

	int *place_holder_update_local = &place_holder_update[wid];
	int place_holder_update_fence = 0;

	
	int len_seg_cache_update_local = len_seg_cache_update;
	Cell *p_cell_local = NULL;

	SIEntry *p_sie_med;
	SIEntry *p_sie_tar;
	int idx_cell_med = -1, idx_bkt_med = -1, idx_obj_med = -1, oid_med = -1;
	int idx_cell_tar = -1, idx_bkt_tar = -1, idx_obj_tar = -1, oid_tar = -1;
	ObjBox *p_obj_tar = NULL, *p_obj_med = NULL;
	int tmp_cnt;

	int *arr_delete = NULL;
	UpdateType *arr_i, *update_i;
	UpdateType *arr_f, *update_f;

	int cnt_elem;
	int cnt_bkts;
	int idx_in_queue;

	__syncthreads();
	barrier_fence += barrier_step;
	if (tid_b == 0)
	{
		atomicAdd(&barrier_update, 1);
		while (barrier_update < barrier_fence);
	}
	__syncthreads();


	while (true)
	{
		if(tid == 0)
		{
			atomicExch(&exp_hunger_update_old, exp_hunger_update);
			while ((req_cache_update->token0 == 0) && (req_cache_update->token1 == 0) &&
				   (buffer_exhausted == 0 || req_cache_update->token0 == 1 || req_cache_update->token0 == 1)) {
				exp_hunger_update++;
			}

			if (rebalance == 1 || (buffer_exhausted == 1 && req_cache_update->token0 == 0 && req_cache_update->token1 == 0))
			{
				atomicExch(&exit_update, 1);
			}
		}
		__syncthreads();
        barrier_fence += barrier_step;
	    if (tid_b == 0)
       	{
            atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
 		__syncthreads();
		if(exit_update == 1)
			break;

		if (tid == 0)
		{
			if ((req_cache_update->token0 == 1) && (req_cache_update->cnt0 == cnt_dequeue_update))
			{
				node_dequeue_update = &req_cache_update->array[0];
				flag_switch_update = 0;
			}
			else if ((req_cache_update->token1 == 1) && (req_cache_update->cnt1 == cnt_dequeue_update))
			{
				node_dequeue_update = &req_cache_update->array[1];
				flag_switch_update = 1;
			}
			
		}
		if(start == -1){
			start = clock64();
		}		

		__threadfence_system();
		__syncthreads();
        barrier_fence += barrier_step;
	    if (tid_b == 0)
       	{
            atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
 		__syncthreads();

		if (flag_switch_update == 0)
		{
			p_grid_local = index_A;
			p_seci_local = sec_index_A;
			obs_pool_local = obs_pool_A;
		}
		else if(flag_switch_update == 1)
		{
			p_grid_local = index_B;
			p_seci_local = sec_index_B;
			obs_pool_local = obs_pool_B;
		}

		if(tid == 0)
		{
			atomicExch(&p_grid_local->cursor_update_wrap, 0);
			atomicExch(&cnt_update_d, 0);
			atomicExch(&cnt_update_i, 0);
			atomicExch(&cnt_update_f, 0);
		}
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
		__syncthreads();


		MemElementCollection<int>* mec_delete = node_dequeue_update->mtx_delete_nodes;
#if (DW == 1)
		while(true)
		{	
			if(tid_w == 0)
			{
				*place_holder_update_local = atomicAdd(&p_grid_local->cursor_update_wrap, 1);
			}

			if(*place_holder_update_local >= tot_vgroup_update)
			{
				break;
			}
			int tempcache2;	
			int qn_cursor_idx = node_dequeue_update->mtx_delete_idx[*place_holder_update_local];
			int tempcache = *place_holder_update_local;
			//the step will go wrong when WRAPSIZE>32
#else 
		anchor2 = tid;
		//lwh32->WRAPSIZE
		while(anchor2 < tot_vgroup_update*WRAPSIZE)
		{
			int tempcache2;	
			int qn_cursor_idx = node_dequeue_update->mtx_delete_idx[anchor2/WRAPSIZE];
			int tempcache = anchor2/WRAPSIZE;
#endif
			do{
				anchor = tid_w;
				//MemItem<int>* qn_cursor = &node_dequeue_update->mtx_delete_node[qn_cursor_idx];
				MemElement* me_cursor = &(mec_delete->mes[qn_cursor_idx]);
				tmp_cnt = mec_delete->cnt[qn_cursor_idx];
				while(anchor < tmp_cnt)
				{
					realWrapSize = tmp_cnt -  \
						((int)(anchor/WRAPSIZE))*WRAPSIZE;
					if(realWrapSize > WRAPSIZE) 
						realWrapSize = WRAPSIZE;	
#if IGNORE_CNT==0
					atomicAdd(&cnt_update_d, 1);
#endif
				//	if(p_cell_local->tot_obs_top + len_bkt*(p_cell_local->cnt_bkts-1) \
				//		!= p_cell_local->tot_obs)
				//		printf(" error. %d", p_cell_local->idx);
					if(tid_w == 0)
						*place_holder_update_local = realWrapSize;
					
					oid_med = mec_delete->pool[qn_cursor_idx*mec_delete->LEN + anchor];
					p_sie_med = &p_seci_local->index[oid_med];
					idx_cell_med = p_sie_med->idx_cell;
					idx_bkt_med = p_sie_med->idx_bkt;
					idx_obj_med = p_sie_med->idx_obj;
					p_cell_local = &p_grid_local->arr_cell[idx_cell_med];
				//	if(p_cell_local->tot_obs_top + len_bkt*(p_cell_local->cnt_bkts-1) \
                //                != p_cell_local->tot_obs)
                //                printf(" error: %d ", p_cell_local->idx);
					if (tempcache != p_cell_local->subgrid)
					{
						atomicAdd(place_holder_update_local, -1);
					}
					if (tempcache != p_cell_local->subgrid)
					{
						atomicAdd(&exp_anchor_x_cell, 1);
						break;
					}
					realWrapSize = *place_holder_update_local;
					tempcache2 = p_cell_local->tot_obs;
					p_obj_med = &(p_grid_local->getBkt(p_cell_local->idx, idx_bkt_med)[idx_obj_med]);
					if(p_obj_med->x == 0)//p_obj_med->oid != oid_med)
					{
						atomicAdd(&exp_anchor_x_subgrid, 1);
                        break;
					}
					if(idx_bkt_med*len_bkt+idx_obj_med >= p_cell_local->tot_obs-realWrapSize)
					{
						atomicExch(&p_obj_med->oid, -1);
					}
					if(tid_w == 0)
						*place_holder_update_local = p_cell_local->tot_obs_top;
					if(idx_bkt_med * len_bkt + idx_obj_med < p_cell_local->tot_obs - realWrapSize)
					{
						while(true)
						{
							p_obj_tar = NULL;
							idx_obj_tar = atomicAdd(&p_cell_local->tot_obs_top, -1) - 1;
							if(idx_obj_tar < *place_holder_update_local - realWrapSize)
								break;
							idx_bkt_tar = p_cell_local->cnt_bkts - 1;
							if(idx_obj_tar < 0){
								idx_obj_tar = idx_obj_tar + len_bkt;
								idx_bkt_tar = idx_bkt_tar -1;
							}
							if(idx_obj_tar < 0 || idx_bkt_tar < 0)
							{
								break;
							}
							p_obj_tar = &(p_grid_local->getBkt(p_cell_local->idx, idx_bkt_tar)[idx_obj_tar]);
							if(p_obj_tar->oid >= 0) 
								break;
						}		
#if IGNORE_CNT==0
						if(p_obj_tar != NULL && p_obj_tar->x == 0 && p_obj_tar->y == 0)
						{
							printf("copy version fail???");
						}
#endif
						if(p_obj_tar == NULL)
						{	
							atomicAdd(&p_cell_local->tot_obs_top, 1);
					    }
						if(p_obj_tar == NULL)
						{        
							atomicAdd(&exp_cell_empty, 1);
							//atomicAdd(place_holder_update_local, 1);
							break;//error exit;
						}
						
						p_sie_tar = &p_seci_local->index[p_obj_tar->oid];
						
						p_obj_med->x = p_obj_tar->x;
						p_obj_med->y = p_obj_tar->y;
						p_obj_med->time = p_obj_tar->time;
						p_obj_med->oid = p_obj_tar->oid;
				
						p_sie_tar->idx_bkt = p_sie_med->idx_bkt;
						p_sie_tar->idx_obj = p_sie_med->idx_obj;
						p_obj_tar->oid = -1;
					}
					atomicAdd(place_holder_update_local, -1);
					if(tid_w == 0)
						atomicExch(&p_cell_local->tot_obs_top, *place_holder_update_local);
					atomicAdd(&p_cell_local->tot_obs, -1);
					if (p_cell_local->tot_obs_top  <= 0 && p_cell_local->cnt_bkts > 1)
					{
						if(tid_w == 0)
						{
							cnt_bkts = p_cell_local->cnt_bkts - 1;
							idx_in_queue = atomicAdd(&queue_bkts_free->rear, 1);
							queue_bkts_free->avail_idx_bkt[idx_in_queue % queue_bkts_free->capacity] = p_grid_local->getArrIdxBkt(p_cell_local->idx)[cnt_bkts];
							cnt_elem = atomicAdd(&queue_bkts_free->cnt_elem, 1);					
							p_grid_local->getArrIdxBkt(p_cell_local->idx)[cnt_bkts] = -1;
							p_cell_local->tot_obs_top += len_bkt;
							p_cell_local->cnt_bkts = cnt_bkts;			
#if IGNORE_CNT==0
							atomicAdd(&cnt_free_obj_pool, 1);
#endif
						}
					}
						
					p_sie_med->idx_cell = -1;
					p_sie_med->idx_bkt = -1;
					p_sie_med->idx_obj = -1;
					
#if IGNORE_CNT==0
					if(p_cell_local->tot_obs_top + len_bkt*(p_cell_local->cnt_bkts-1) \
						!= p_cell_local->tot_obs)
						printf(" error,%d ", tempcache2);
#endif
					anchor += WRAPSIZE;
				}
				qn_cursor_idx = me_cursor->next;
				tmp_cnt = mec_delete->cnt[qn_cursor_idx];
			}while(qn_cursor_idx != -1);
			__threadfence_system();	
#if (DW==1)
#else
			anchor2 += blockDim.x*gridDim.x;
#endif
		}

		__threadfence_system();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
		__syncthreads();
		
		if(tid == 0)
			atomicExch(&p_grid_local->cursor_update_wrap, 0);
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
		__syncthreads();


		MemElementCollection<UpdateType>* mec_insert = node_dequeue_update->mtx_insert_nodes;
#if (DW==1)
		while(true)
		{	
			if(tid_w == 0)
			{
				*place_holder_update_local = atomicAdd(&p_grid_local->cursor_update_wrap, 1);
			}

			if(*place_holder_update_local >= tot_vgroup_update)
			{
				break;//exit;
			}
			int qn_cursor_idx = node_dequeue_update->mtx_insert_idx[*place_holder_update_local];
			int tempcache = *place_holder_update_local;
#else 
		anchor2 = tid;
		//lwh32->WRAPSIZE
		while(anchor2 < tot_vgroup_update*WRAPSIZE)
		{
			int qn_cursor_idx = node_dequeue_update->mtx_insert_idx[anchor2/WRAPSIZE];
			int tempcache = anchor2/WRAPSIZE;
#endif
			do{	
				anchor = tid_w;
				MemElement* me_cursor = &(mec_insert->mes[qn_cursor_idx]);
				tmp_cnt = mec_insert->cnt[qn_cursor_idx];
				while(anchor < tmp_cnt)
				{
					realWrapSize = tmp_cnt -  \
						((int)(anchor/WRAPSIZE))*WRAPSIZE;
					if(realWrapSize > WRAPSIZE) 
						realWrapSize = WRAPSIZE;
#if IGNORE_CNT==0
					atomicAdd(&cnt_update_i, 1);
#endif
					update_i = &mec_insert->pool[qn_cursor_idx*mec_insert->LEN + anchor];
					oid_tar = update_i->oid;
					p_sie_tar = &p_seci_local->index[oid_tar];
					p_cell_local = p_grid_local->getCellByXY(update_i->x, update_i->y);
					if(p_cell_local == NULL){
						anchor += WRAPSIZE;
						continue;
					}
					if (p_cell_local->subgrid != tempcache)
					{
						atomicAdd(&exp_anchor_x_subgrid, 1);
					}
					idx_bkt_tar = 0;
					idx_obj_tar = (p_cell_local->tot_obs_top + tid_w) % len_bkt;
					if (p_cell_local->tot_obs_top > len_bkt - realWrapSize)
					{
						if(tid_w < len_bkt -  p_cell_local->tot_obs_top)
						{
							idx_bkt_tar = -1;
						}
						if(p_cell_local->cnt_bkts >= p_cell_local->len_arr_bkts)
						{
							//printf("%d cell over bkts", p_cell_local->idx);
							atomicAdd(&cnt_over_bkt_update, 1);
							break;
						}
						if(tid_w == 0)
						{
							cnt_elem = atomicAdd(&queue_bkts_free->cnt_elem, -1);

							idx_in_queue = atomicAdd(&queue_bkts_free->head, 1);
							int idx_bkt_in_pool = queue_bkts_free->avail_idx_bkt[idx_in_queue % queue_bkts_free->capacity];

							queue_bkts_free->avail_idx_bkt[idx_in_queue] = -1;
							cnt_bkts = p_cell_local->cnt_bkts;

							p_grid_local->getArrIdxBkt(p_cell_local->idx)[cnt_bkts] = idx_bkt_in_pool;

							p_cell_local->tot_obs_top = p_cell_local->tot_obs_top - len_bkt;

							p_cell_local->cnt_bkts++;
#if IGNORE_CNT==0
							atomicAdd(&cnt_malloc_obj_pool, 1);
#endif
							__threadfence_system();
						}
						
					}
					atomicAdd(&p_cell_local->tot_obs_top, 1);
					
					idx_cell_tar = p_cell_local->idx;
					idx_bkt_tar += p_cell_local->cnt_bkts - 1;
					
					p_obj_tar = &(p_grid_local->getBkt(p_cell_local->idx, idx_bkt_tar)[idx_obj_tar]);
					p_obj_tar->y = update_i->y;
					p_obj_tar->oid = update_i->oid;  //add a Object
					p_obj_tar->x = update_i->x;
					p_obj_tar->time = update_i->time;	
#if IGNORE_CNT==0
					if(p_obj_tar->x == 0)
						printf("???");
#endif
					p_sie_tar->idx_cell = idx_cell_tar;    //update the SIEntry
					p_sie_tar->idx_bkt = idx_bkt_tar;
					p_sie_tar->idx_obj = idx_obj_tar;

					atomicAdd(&p_cell_local->tot_obs, 1);
					anchor += WRAPSIZE;
				}
				qn_cursor_idx = me_cursor->next;
				tmp_cnt = mec_insert->cnt[qn_cursor_idx];
			}while(qn_cursor_idx!= -1);
			__threadfence_system();
#if(DW==1)
#else
			anchor2 += blockDim.x*gridDim.x;
#endif
		}


		__threadfence_system();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
		__syncthreads();

		if(tid == 0)
			atomicExch(&p_grid_local->cursor_update_wrap, 0);
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
		__syncthreads();

		MemElementCollection<UpdateType>* mec_fresh = node_dequeue_update->mtx_insert_nodes;
#if (DW==1)
		while(true)
		{	
			if(tid_w == 0)
			{
				*place_holder_update_local = atomicAdd(&p_grid_local->cursor_update_wrap, 1);
			}
			
			if(*place_holder_update_local >= tot_vgroup_update)
			{
				break;//exit;
			}
			int qn_cursor_idx = node_dequeue_update->mtx_fresh_idx[*place_holder_update_local];
#else 
		anchor2 = tid;
		//lwh32->WRAPSIZE
		while(anchor2 < tot_vgroup_update*WRAPSIZE)
		{
			int qn_cursor_idx = node_dequeue_update->mtx_fresh_idx[anchor2/WRAPSIZE];
#endif
			do{	
				anchor = tid_w;
				//MemItem<UpdateType>* qn_cursor = &node_dequeue_update->mtx_insert_node[qn_cursor_idx];
				MemElement* me_cursor = &(mec_fresh->mes[qn_cursor_idx]);
				tmp_cnt = mec_fresh->cnt[qn_cursor_idx];
				while(anchor < tmp_cnt)
				{
#if IGNORE_CNT==0
					atomicAdd(&cnt_update_f, 1);
#endif
					update_f = &mec_fresh->pool[qn_cursor_idx*mec_fresh->LEN + anchor];
					oid_tar = update_f->oid;
					p_sie_tar = &p_seci_local->index[oid_tar];

					idx_bkt_tar = p_sie_tar->idx_bkt;
					idx_obj_tar = p_sie_tar->idx_obj;

					p_cell_local = &p_grid_local->arr_cell[p_sie_tar->idx_cell];
					p_obj_tar = &(p_grid_local->getBkt(p_cell_local->idx, idx_bkt_tar)[idx_obj_tar]);
					p_obj_tar->x = update_f->x; 
					p_obj_tar->y = update_f->y;
					p_obj_tar->time = update_f->time;
					
					anchor += WRAPSIZE;
				} __threadfence_system();
				qn_cursor_idx = me_cursor->next;
				tmp_cnt = mec_fresh->cnt[qn_cursor_idx];
			}while(qn_cursor_idx!= -1);
			__threadfence_system();
#if (DW==1)
#else
			anchor2 += blockDim.x*gridDim.x;
#endif
		}
		
#if CHECK_SI==1
		if(tid == 0)
		{
			atomicExch(&(p_seci_local->index[p_config->max_obj_num].idx_cell), -1);
		}
#endif
		
		anchor = tid;
        	while(anchor < tot_vgroup_update)
		{
			p_grid_local->arr_cell[anchor].memfencedelay = 1;
			anchor += gridDim.x*blockDim.x;
		}
		__threadfence_system();
		cudaDeviceSynchronize();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
		__syncthreads();


		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
		__syncthreads();

		end = clock64();
		
	// 	including:
	//	MMItem mm_d_obs_pool
	//	MMItem mm_d_sie_array
	//	MMItem mm_d_arr_bkts
	//	MMItem mm_d_arr_idx_bkt
	//	MMItem mm_d_cell
	//	MMItem mm_d_arr_cell
		if(tid == 0)
		{
			while(cnt_dequeue_query < cnt_dequeue_update);
			start2 = clock64();	
			if(flag_switch_update == 0)
			{		
				for(int i = 0 ; i < DMM->mmsLen; i++){
					atomicExch(DMM->mms[i].toCheckpoint, 0);
					atomicExch(DMM->mms[i].checkpoint, 1);
	 				cudaMemcpyAsync(DMM->mms[i].toPtr, DMM->mms[i].ptr,\
	                    DMM->mms[i].bsize, cudaMemcpyDeviceToDevice);
				}
				cudaDeviceSynchronize();
				for(int i = 0 ; i < DMM->mmsLen; i++){
					while(*(DMM->mms[i].toCheckpoint) != 1);
					atomicExch(DMM->mms[i].toCheckpoint, 0);
					atomicExch(DMM->mms[i].checkpoint, 0);
				}
				atomicExch(&flag_switch_version, 1);
			}else if(flag_switch_update == 1){
				for(int i = 0 ; i < DMM->mmsLen; i++){
					atomicExch(DMM->mms[i].checkpoint, 0);
					atomicExch(DMM->mms[i].toCheckpoint, 1);
	 				cudaMemcpyAsync(DMM->mms[i].ptr, DMM->mms[i].toPtr,\
	                    DMM->mms[i].bsize, cudaMemcpyDeviceToDevice);
				}
				cudaDeviceSynchronize();
				for(int i = 0 ; i < DMM->mmsLen; i++){
					while(*(DMM->mms[i].checkpoint) != 1);
					atomicExch(DMM->mms[i].toCheckpoint, 0);
					atomicExch(DMM->mms[i].checkpoint, 0);
				}
				atomicExch(&flag_switch_version, 0);
			}
			end2 = clock64();
			copytime += end2-start2;
		}
		cudaDeviceSynchronize();
		//end = clock64();
                int forlimit = 10000;
                int forresult = 0;
                for(int i = 0; i<forlimit; i++){
                        forresult = (forresult+tid)*tid/((double)tid+12);
                }
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
		__syncthreads();	

		if (tid == 0)
		{
			atomicExch(&update_time_per_period, (int)(end-start));
			if(cnt_dequeue_update>=QUERY_SKIP_NUM)
				update_sumtime += (end-start);
			printf("\n\nu%d %.4fms i%d, d%d, f%d\n", cnt_dequeue_update, \
				((double)(end-start))/p_config->clockRate, \
				cnt_update_i, cnt_update_d, cnt_update_f);
			start = -1;
			if (flag_switch_update == 0)
			{
				atomicExch(&req_cache_update->token0, 0);
			}
			else if (flag_switch_update == 1)
			{
				atomicExch(&req_cache_update->token1, 0);
			}
			node_dequeue_update = NULL;
			atomicAdd(&cnt_dequeue_update, 1);
		}

	/*	if (tid == 40)
                {
                        printf("\nCell obj num check:\n");
                        for(int i = gp_config->edge_cell_num*gp_config->edge_cell_num - 1; i>=46000 ;i-=50){
                                //p_cell_local = &arr_cell[idx_cells_covered[i]];
                                p_cell_local = &p_grid_local->arr_cell[i];
                                printf("%d|%d ", p_cell_local->idx, p_cell_local->tot_obs);
                        }
                        printf("\n\n");
                }
	*/
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_update, 1);
			while (barrier_update < barrier_fence);
		}
		__syncthreads();


	}

	__syncthreads();
	barrier_fence += barrier_step;
	if (tid_b == 0)
	{
		atomicAdd(&barrier_update, 1);
		while (barrier_update < barrier_fence);

	}
	__syncthreads();


	if (tid == 0)
	{
		
		if(rebalance == 0){
			printf("\n");
			printf("\nUpdateKernel clock: %f ms\n",((double)update_sumtime)/p_config->clockRate);
			printf("\nVersion Copy clock: %f ms\n",((double)copytime)/p_config->clockRate);
			printf("Version Copy size: %.4f MB\n", (double)(DMM->mms[0].bsize + DMM->mms[1].bsize\
				+ DMM->mms[2].bsize + DMM->mms[3].bsize + DMM->mms[4].bsize + DMM->mms[5].bsize \
				)/(1024.0f*1024.0f));
			printf("len_seg_cache_update_local: %d\n", len_seg_cache_update_local);
			printf("cnt_dequeue_update: %d\n", cnt_dequeue_update);
			printf("exp_hunger_update: %d\n", exp_hunger_update);
			printf("exp_cell_full: %d\n", exp_cell_full);
			printf("exp_cell_empty: %d\n", exp_cell_empty);
			printf("exp_anchor_x_cell: %d\n", exp_anchor_x_cell);
			//printf("Size of Shared Memory: %f\n", sizeof(int) * 12288 / (1024.0));
			printf("exp_idx_x_subgrid_a: %d\n", exp_idx_x_subgrid_a);
			printf("exp_idx_x_subgrid_b: %d\n", exp_idx_x_subgrid_b);
			printf("exp_anchor_x_subgrid: %d\n", exp_anchor_x_subgrid);
			printf("exp_idx_bkt_dequeue_error: %d\n", exp_idx_bkt_dequeue_error);
			bkts_per_cell = 0;
			obs_per_cell= 0;
			for (int i = 0; i < tot_cells; i++)
			{
				bkts_per_cell += index_A->arr_cell[i].cnt_bkts;
				obs_per_cell += index_A->arr_cell[i].tot_obs;
			}
			printf("A:bkts_per_cell: %d %d\n", bkts_per_cell / tot_cells, tot_cells);
			printf("A:obs_per_cell: %d %d\n", obs_per_cell / tot_cells, tot_cells);
			
			bkts_per_cell = 0;
			obs_per_cell= 0;
			for (int i = 0; i < tot_cells; i++)
			{
				bkts_per_cell += index_B->arr_cell[i].cnt_bkts;
				obs_per_cell += index_B->arr_cell[i].tot_obs;
			}
			printf("B:bkts_per_cell: %d %d\n", bkts_per_cell / tot_cells, tot_cells);
			printf("B:obs_per_cell: %d %d\n", obs_per_cell / tot_cells, tot_cells);

	//		printf("bkts_per_cell: %d %d\n", index_A->arr_cell[100].cnt_bkts, tot_cells);
	//		printf("obs_per_cell: %d %d\n", index_A->arr_cell[100].tot_obs, tot_cells);

			printf("queue_bkts_free->capacity: %d \n", queue_bkts_free->capacity);
			printf("queue_bkts_free->head: %d \n", queue_bkts_free->head);
			printf("queue_bkts_free->rear: %d \n", queue_bkts_free->rear);
			printf("queue_bkts_free->cnt_elem: %d \n", queue_bkts_free->cnt_elem);
			printf("cnt_free_obj_pool: %d \n", cnt_free_obj_pool);
			printf("cnt_malloc_obj_pool: %d \n", cnt_malloc_obj_pool);
			printf("cnt_over_bkt_update: %d \n", cnt_over_bkt_update);

			
		}
	}
	

	__syncthreads();
	barrier_fence += barrier_step;
	if (tid_b == 0)
	{
		atomicAdd(&barrier_update, 1);
		while (barrier_update < barrier_fence);

	}
	__syncthreads();

	if(tid == 0)
	{
		atomicExch(&barrier_update, 0);
		atomicExch(&launch_signal, 0);
		atomicExch(&update_time_per_period, 0);
	}
}

