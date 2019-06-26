/**********************************************************************
 * DistributorKernel.cu
 * Copyright @ Cloud Computing Lab, CS, Wuhan University
 * Author: Chundan Wei
 * Email: danuno@qq.com
 * Version: 1.0
 * Date: Oct 22, 2014 | 10:36:45 AM
 * Description:*
 * Licence:*
 **********************************************************************/

#include <memory.h>
#include <string.h>
#include <stdio.h>

#include "config/GConfig.h"
#include "misc/BaseStruct.h"
#include "misc/Buffer.h"
#include "device/DeviceGlobalVar.cuh"
#include "DistributorKernel.cuh"

#include "misc/Cell.cuh"
#include "misc/ObjBox.cuh"

#include "misc/UpdateQNode.cuh"
#include "misc/UpdateCacheArea.cuh"
#include "misc/QueryCacheArea.h"
#include "misc/SyncFuncGPU.cuh"

__global__ void DpProcess(int* place_holder_query_dispatch_local,
		int anchor_idx, int row_start, int row_end, int col_start, int col_end,
		GConfig * p_config, QueryQNode *node_enqueue_query_local)
{
	int k = threadIdx.x;
	int i = row_start + k / (col_end - col_start + 1);
	int j = col_start + k % (col_end - col_start + 1);
	const int EDGE_CELL_NUM = p_config->edge_cell_num;
	const int LEN_SEG_CACHE_QUERY = p_config->len_seg_cache_query;
	int* cnt_queries_per_cell = node_enqueue_query_local->cnt_queries_per_cell;
	int* flag_cells_covered = node_enqueue_query_local->flag_cells_covered;
	volatile int* volatile idx_cells_covered =
			node_enqueue_query_local->idx_cells_covered;
	int* queries_per_cell = node_enqueue_query_local->queries_per_cell;
	int idx_cell = i * EDGE_CELL_NUM + j;
	int idx_tmp_0;
	int idx_tmp = atomicAdd(&cnt_queries_per_cell[idx_cell], 1);
	atomicAdd(&check_tot_covered, 1);
	if (idx_tmp >= LEN_SEG_CACHE_QUERY)
	{
		atomicAdd(&cnt_queries_per_cell[idx_cell], -1);
		atomicAdd(&cnt_over_seg_query, 1);
		atomicAdd(place_holder_query_dispatch_local, 1);
		return;
	}
	if (idx_tmp < 0)
	{
		atomicAdd(place_holder_query_dispatch_local, 1);
		return;
	}
	if (atomicCAS(&flag_cells_covered[idx_cell], 0, 1) == 0)
	{
		atomicExch(&flag_cells_covered[idx_cell], 1);
		idx_tmp_0 = atomicAdd(&node_enqueue_query_local->tot_cells_covered, 1);
		idx_cells_covered[idx_tmp_0] = idx_cell;
	}
	queries_per_cell[idx_cell * LEN_SEG_CACHE_QUERY + idx_tmp] = anchor_idx;
	atomicAdd(place_holder_query_dispatch_local, 1);
	return;
}

__global__ void DistributorKernel(GConfig *dev_p_gconfig,
		UpdateType *dev_buffer_update, int *dev_cnt_update,
		QueryType *dev_buffer_query, int *dev_cnt_query,
		UpdateCacheArea *d_req_cache_update, QueryCacheArea *d_req_cache_query,
		Grid *d_index_A, Grid *d_index_B, CircularQueue *d_queue_bkts_free,
		MemItem<QueryType>* d_qd_obj_pool,
		CircularQueue* d_queue_idx_anchor_free, QueryType* d_qd_query_type_pool,
		int* d_qd_anchor_pool, int* d_place_holder, ManagedMemory* d_mm)
{

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid_b = threadIdx.x;
	const unsigned int barrier_step = gridDim.x;
	volatile unsigned int barrier_fence = 0;
	int anchor = 0, anchor_i = 0;
	//__shared__ int occupation[12288];
	//extern __shared__  int s[];
	const int WRAPSIZE = 32;
	int realWrapSize = 0;
	int realBlockSize = 0;
	const int wid = tid / WRAPSIZE;
	const int tid_w = tid % WRAPSIZE;

	GConfig *p_config = dev_p_gconfig;

	//for debug
	int check_dist_query_local = 0;
	//int check_tot_covered_local = 0;
	int check_cnt_enqueue = 0;
	int check_lb_null = 0;
	int check_rt_null = 0;
	long long int start = -1, end;
	//long long int copytime = 0;
	//double sum_time = 0;
	// printf("%d\n", gp_config->block_analysis_num);

	// Initialize global variables
	if (tid == 0)
	{
		atomicExch(&buffer_exhausted, 0);
		atomicExch(&rebalance, 0);
		atomicExch(&launch_signal, 0);
		atomicExch(&exit_query, 0);
		atomicExch(&exit_update, 0);
	}

	__threadfence_system();
	__syncthreads();
	barrier_fence += barrier_step;
	if (tid_b == 0)
	{
		atomicAdd(&barrier_dist, 1);
		while (barrier_dist < barrier_fence)
			;
		printf("sync-1 over\n");

	}
	__syncthreads();

	// Initialize local variables
	const int EDGE_CELL_NUM = p_config->edge_cell_num;
	const int TOT_CELLS = p_config->edge_cell_num * p_config->edge_cell_num;
	const int TOT_VGROUP_UPDATE = p_config->side_len_vgroup
			* p_config->side_len_vgroup;
	//int query_dispatch_fence = 0;
	//int *sync_holder_query_dispatch_local = &sync_holder_query_dispatch[wid];
	int *place_holder_query_dispatch_local = &place_holder_query_dispatch[wid];
	int *cache_memory_idx_query_dispatch_local =
			&cache_memory_idx_query_dispatch[wid];
	int *place_holder_update_dispatch_local = &place_holder_update_dispatch[wid
			* UPDATE_DISPATCH_SEG];

	ManagedMemory lmm = *d_mm;
	const int QUERY_DISPATCH_WRAP_SIZE = 512;

	// Initialize local update variables
	Grid *local_index = NULL;
	SecIndex *local_seci = NULL;
	UpdateType *buffer_update = dev_buffer_update;
	int offset_buffer_update = offset_buffer_update_rec;
	const int TOTAL_UPDATE = p_config->max_obj_num * p_config->round_num;
	int buffer_block_size_update = p_config->buffer_block_size;
	int len_seg_cache_update_local = len_seg_cache_update;

	int *local_d;
	int *local_d_pool;
	int *local_d_cnt;
	int *local_i;
	UpdateType *local_i_pool;
	int *local_i_cnt;
	int *local_f;
	UpdateType *local_f_pool;
	int *local_f_cnt;
	CircularQueue *local_d_fqueue, *local_i_fqueue, *local_f_fqueue;
	int idx_start, idx_last;
	MemElement* tmp_me;
	int d_seg, i_seg, f_seg;

	Cell *p_cell_new = NULL;
	Cell *p_cell_old = NULL;
	//Cell *p_cell_marked = NULL;

	int oid = -1;
	SIEntry *p_sie = NULL;

	UpdateType* ins_update;

	// Initialize local query variables
	QueryType *buffer_query = dev_buffer_query, *p_buffer_block_query,
			req_query;
	int offset_buffer_query = 0;
	const int TOTAL_QUERY = p_config->max_query_num;
	const int LEFT_UPDATE_AFTERSKIP = TOTAL_UPDATE
			- p_config->query_skip_round_num * p_config->max_obj_num;
	int buffer_block_size_query = offset_buffer_query_rec;
	if (LEFT_UPDATE_AFTERSKIP > p_config->buffer_block_size)
	{
		buffer_block_size_query = (int) ((double) p_config->buffer_block_size
				* (double) p_config->max_query_num
				/ (double) LEFT_UPDATE_AFTERSKIP);
	}
	else
	{
		buffer_block_size_query = p_config->max_query_num;
	}
	const int LEN_SEG_CACHE_QUERY = p_config->len_seg_cache_query;
	QueryType* d_qd_query_type_local;
	int* d_qd_anchor_local;
	const int QUERY_TYPE_POOL_SIZE = lmm.mm_qd_query_type_pool.len;
	const int QUERY_SKIP_NUM = p_config->query_skip_round_num
			* p_config->max_obj_num;
	const int QT_SIZE = p_config->qt_size;
	const int QUEUE_SEG_LEN = p_config->len_seg_multiqueue;
	const int MQUEUE_SIZE = p_config->len_multiqueue;

	//int idx_query = 0;
	int idx_cell = 0, idx_tmp = 0, idx_tmp_0 = 0;
	Grid *p_grid = NULL;
	Cell *p_cell_rt = NULL, *p_cell_lb = NULL;

	int left_bottom;
	int right_top;
	int cell_num;
	int row_start;
	int col_end;
	int row_end;
	int col_start;

	float xmin, ymin, xmax, ymax;

	QueryQNode *node_enqueue_query_local = NULL;
	//int tot_cells_covered = 0;
	int *flag_cells_covered = NULL;
	volatile int * volatile idx_cells_covered = NULL;
	int *cnt_queries_per_cell = NULL;
	int *queries_per_cell = NULL;

	QueryType *buffer_block_query = NULL;

	if (tid == 0)
	{
		atomicExch(&launch_signal, 1);
	}

	__syncthreads();
	barrier_fence += barrier_step;
	if (tid_b == 0)
	{
		atomicAdd(&barrier_dist, 1);
		while (barrier_dist < barrier_fence)
			;
		printf("sync-2 over\n");

	}
	__syncthreads();

	if (tid == 0)
	{
		atomicExch(&req_cache_update->token0, 0);
		atomicExch(&req_cache_update->token1, 0);
		atomicExch(&req_cache_query->token0, 0);
		atomicExch(&req_cache_query->token1, 0);
	}

	/*	for(int k = 0; k<2; k++){
	 CircularQueue* fqueue_local;
	 for (int i = 0; i<3 ;i++)
	 {
	 if (i == 0)
	 fqueue_local = req_cache_update->array[k].fqueue_delete;
	 else if (i == 1)
	 fqueue_local = req_cache_update->array[k].fqueue_insert;
	 else if (i == 2)
	 fqueue_local = req_cache_update->array[k].fqueue_fresh;
	 anchor = tid;
	 while(anchor < fqueue_local->capacity){
	 if(anchor<TOT_VGROUP_UPDATE)
	 {

	 }
	 atomicExch(&fqueue_local->avail_idx_bkt[anchor], anchor);
	 anchor += blockDim.x*gridDim.x;
	 }
	 if(tid == 0){
	 atomicExch(&fqueue_local->cnt_elem, fqueue_local->capacity);
	 atomicExch(&fqueue_local->head, 0);
	 atomicExch(&fqueue_local->rear, 0);
	 }

	 }
	 }*/

	__syncthreads();
	barrier_fence += barrier_step;
	if (tid_b == 0)
	{
		atomicAdd(&barrier_dist, 1);
		while (barrier_dist < barrier_fence)
			;
	}
	__syncthreads();

	while (offset_buffer_update < TOTAL_UPDATE)
	{
		// Distribute Updates
		if (offset_buffer_update + buffer_block_size_update >= TOTAL_UPDATE)
		{
			buffer_block_size_update = TOTAL_UPDATE - offset_buffer_update;
		}

		while ((flag_switch_dist == 1
				&& (req_cache_update->token0 == 1
						|| req_cache_query->token0 == 1))
				|| (flag_switch_dist == 0
						&& (req_cache_update->token1 == 1
								|| req_cache_query->token1 == 1)))
		{
			if (tid == 0)
				exp_hunger_dist0++;
		}

		if (query_time_per_period != 0 && update_time_per_period != 0)
		{
			double Tdis = dis_time_per_period;
			double Tupd = update_time_per_period;
			double Tque = query_time_per_period;
			double Dis = p_config->block_analysis_num;
			double Upd = p_config->block_update_num;
			double Que = p_config->block_query_num;
			double TdisDis = Tdis * (double) Dis;
			double TupdUpd = Tupd * (double) Upd;
			double TqueQue = Tque * (double) Que;
			double x_double =
					(TdisDis * (Upd + Que) - Dis * (TupdUpd + TqueQue))\

							/ (TdisDis + TupdUpd + TqueQue);
			double y_double = TupdUpd / TdisDis * x_double + TupdUpd / Tdis
					- Upd;

			if (tid == 0)
				printf("relance, x is %.2f, y is %.2f \n", x_double, y_double);
			int x_int = (int) (x_double);
			int y_int = (int) (y_double);
			if (offset_buffer_update
					<= (p_config->query_skip_round_num + 2)
							* p_config->max_obj_num)
			{
				x_double = .0f;
				y_double = .0f;
				x_int = 0;
				y_int = 0;
			}

#if REBALANCE == 0
			x_double = 0;
			y_int = 0;
			x_int = 0;
			y_double = 0;
#endif

			if (abs(x_int) >= 1 || abs(y_int) >= 1)
			{
				int newDis = Dis + x_int;
				int newQue = Que - x_int;
				if (abs(newDis) < 1)
				{
					x_int = 1 - Dis;
					newDis = Dis + x_int;
					newQue = Que - x_int;
				}
				if (abs(newQue) < 1)
				{
					x_int = Que - 1;
					newDis = Dis + x_int;
					newQue = Que - x_int;
				}

				int newUpd = Upd + y_int;
				int newQue2 = newQue - y_int;
				if (abs(newUpd) < 1)
				{
					y_int = 1 - Upd;
					newUpd = Upd + y_int;
					newQue2 = newQue - y_int;
				}
				if (abs(newQue2) < 1)
				{
					y_int = newQue - 1;
					newUpd = Upd + y_int;
					newQue2 = newQue - y_int;
				}
				p_config->block_analysis_num = newDis;
				p_config->block_update_num = newUpd;
				p_config->block_query_num = newQue2;
				atomicExch(&buffer_exhausted, 1);
				atomicExch(&rebalance, 1);
				if (tid == 0)
					printf("new kernels' rate is %d : %d : %d\n",
							p_config->block_analysis_num,
							p_config->block_update_num,
							p_config->block_query_num);
				break;
			}
		}

		buffer_update = dev_buffer_update
				+ (offset_buffer_update_rec
						% (p_config->buffer_block_size
								* p_config->buffer_update_round));
		while (update_map[(int) (offset_buffer_update_rec
				/ p_config->buffer_block_size)\
 % p_config->buffer_update_round]
				== 0)
			__threadfence_system();

		if (tid == 0)
		{
			if (flag_switch_dist == 1 && req_cache_update->token0 == 0
					&& req_cache_query->token0 == 0)
			{
				node_enqueue_update = &req_cache_update->array[0];
				flag_switch_dist = 0;
			}
			else if (flag_switch_dist == 0 && req_cache_update->token1 == 0
					&& req_cache_query->token1 == 0)
			{
				node_enqueue_update = &req_cache_update->array[1];
				flag_switch_dist = 1;
			}
		}

		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		start = clock64();

		if (flag_switch_version == 0)
		{
			local_index = index_A;
			local_seci = sec_index_A;
		}
		else if (flag_switch_version == 1)
		{
			local_index = index_B;
			local_seci = sec_index_B;
		}

		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;

		}
		__syncthreads();

		local_d = node_enqueue_update->mtx_delete_idx; //local_d refers to Delete Cache Area(oid1, oid2,...,oidn), local_d_cnt refers to its  oid num;
		local_d_cnt = node_enqueue_update->sum_d;
		//local_d_pool = node_enqueue_update->mtx_delete_pool;
		local_d_fqueue = node_enqueue_update->fqueue_delete;
		//d_seg = node_dequeue_update->d_size/TOT_CELLS;

		local_i = node_enqueue_update->mtx_insert_idx; //local_i refers to Insert Cache Area(req1, req2,..., reqn), local_i_cnt refers to its request num;
		local_i_cnt = node_enqueue_update->sum_i;
		//local_i_pool = node_enqueue_update->mtx_insert_pool;
		local_i_fqueue = node_enqueue_update->fqueue_insert;
		//i_seg = node_dequeue_update->i_size/TOT_CELLS;

		local_f = node_enqueue_update->mtx_fresh_idx; //local_f refers to Fresh Cache Area(req1, req2,..., reqn), local_f_cnt refers to its request num;
		local_f_cnt = node_enqueue_update->sum_f;
		//local_f_pool = node_enqueue_update->mtx_fresh_pool;
		local_f_fqueue = node_enqueue_update->fqueue_fresh;
		//f_seg = node_dequeue_update->f_size/TOT_CELLS;

		CircularQueue* fqueue_local;
		for (int i = 0; i < 2; i++)
		{
			if (i == 0)
				fqueue_local = node_enqueue_update->fqueue_delete;
			else if (i == 1)
				fqueue_local = node_enqueue_update->fqueue_insert;
			anchor = tid;
			while (anchor < fqueue_local->capacity)
			{
				fqueue_local->avail_idx_bkt[anchor] = anchor;
				anchor += blockDim.x * gridDim.x;
			}
			if (tid == 0)
			{
				atomicExch(&fqueue_local->cnt_elem, fqueue_local->capacity);
				atomicExch(&fqueue_local->head, 0);
				atomicExch(&fqueue_local->rear, 0);
			}
		}
#if CHECK_SI==1
		if (tid_b == 0)
			while (local_seci->index[p_config->max_obj_num].idx_cell != -1)
				;
#endif
		__threadfence_system();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

#if CHECK_SI==1
		atomicExch(&(local_seci->index[p_config->max_obj_num].idx_cell),
				p_config->max_obj_num);
#endif
		int* mtx_idx;
		fqueue_local = node_enqueue_update->fqueue_delete;
		mtx_idx = node_enqueue_update->mtx_delete_idx;
		anchor = tid;
		idx_tmp = fqueue_local->head;
		while (anchor < TOT_VGROUP_UPDATE)
		{
			fqueue_local->avail_idx_bkt[(idx_tmp + anchor)
					% fqueue_local->capacity] = -1;
			mtx_idx[anchor] = idx_tmp + anchor;
			node_enqueue_update->mtx_delete_nodes->last[anchor] = idx_tmp
					+ anchor;
			anchor += blockDim.x * gridDim.x;
		}
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		if (tid == 0)
		{
			atomicAdd(&fqueue_local->cnt_elem, -TOT_VGROUP_UPDATE);
			atomicAdd(&fqueue_local->head, TOT_VGROUP_UPDATE);
			cudaMemcpyAsync(node_enqueue_update->mtx_delete_nodes->mes,
					node_enqueue_update->mtx_delete_nodes_bak,
					sizeof(MemElement) * node_enqueue_update->d_size,
					cudaMemcpyDeviceToDevice);
			cudaMemsetAsync(node_enqueue_update->mtx_delete_nodes->cnt, 0,
					sizeof(int) * node_enqueue_update->d_size);
		}
		cudaDeviceSynchronize();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		fqueue_local = node_enqueue_update->fqueue_insert;
		mtx_idx = node_enqueue_update->mtx_insert_idx;
		anchor = tid;
		idx_tmp = fqueue_local->head;
		while (anchor < TOT_VGROUP_UPDATE)
		{
			fqueue_local->avail_idx_bkt[(idx_tmp + anchor)
					% fqueue_local->capacity] = -1;
			mtx_idx[anchor] = idx_tmp + anchor;
			node_enqueue_update->mtx_insert_nodes->last[anchor] = idx_tmp
					+ anchor;
			anchor += blockDim.x * gridDim.x;
		}

		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();
		if (tid == 0)
		{
			atomicAdd(&fqueue_local->cnt_elem, -TOT_VGROUP_UPDATE);
			atomicAdd(&fqueue_local->head, TOT_VGROUP_UPDATE);
			cudaMemcpyAsync(node_enqueue_update->mtx_insert_nodes->mes,
					node_enqueue_update->mtx_insert_nodes_bak,
					sizeof(MemElement) * node_enqueue_update->i_size,
					cudaMemcpyDeviceToDevice);
			cudaMemsetAsync(node_enqueue_update->mtx_insert_nodes->cnt, 0,
					sizeof(int) * node_enqueue_update->i_size);
		}
		cudaDeviceSynchronize();

		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		fqueue_local = node_enqueue_update->fqueue_insert;
		mtx_idx = node_enqueue_update->mtx_fresh_idx;
		anchor = tid;
		idx_tmp = fqueue_local->head;
		while (anchor < TOT_VGROUP_UPDATE)
		{
			fqueue_local->avail_idx_bkt[(anchor + idx_tmp)
					% fqueue_local->capacity] = -1;
			mtx_idx[anchor] = idx_tmp + anchor;
			node_enqueue_update->mtx_insert_nodes->last[TOT_VGROUP_UPDATE
					+ anchor] = idx_tmp + anchor;
			anchor += blockDim.x * gridDim.x;
		}
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		if (tid == 0)
		{
			atomicAdd(&fqueue_local->cnt_elem, -TOT_VGROUP_UPDATE);
			atomicAdd(&fqueue_local->head, TOT_VGROUP_UPDATE);
		}
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		anchor = tid;
		while (anchor < TOT_VGROUP_UPDATE)
		{
			local_d_cnt[anchor] = 0;
			local_i_cnt[anchor] = 0;
			local_f_cnt[anchor] = 0;
			anchor += blockDim.x * gridDim.x;
		}

		__threadfence_system();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		/*	if(tid == 0){
		 end = clock64();
		 double temp_sum_time = ((double)(end - start))/p_config->clockRate;
		 sum_time += temp_sum_time;
		 printf("d1-0 %.4f ms\n", temp_sum_time);
		 }
		 start = clock64();
		 */
		if (tid_w == 0)
			place_holder_update_dispatch_local[0] = 0;
		MemElementCollection<UpdateType>* local_node_i =
				node_enqueue_update->mtx_insert_nodes;
		MemElementCollection<int>* local_node_d =
				node_enqueue_update->mtx_delete_nodes;
		CircularQueue* local_fqueue_u;   //= local_i_fqueue;
		anchor = tid;

		//anchor = buffer_block_size_update;

		while (anchor < buffer_block_size_update)
		{
#if SEG_CACHE == 0
			if (place_holder_update_dispatch_local[0] < 0)
				place_holder_update_dispatch_local[0] = 0;
			if (place_holder_update_dispatch_local[0] < UPDATE_DISPATCH_SEG - 1)
			{
				if (tid_w
						< UPDATE_DISPATCH_SEG - 1
								- place_holder_update_dispatch_local[0])
				{
					int cnt_elem = atomicAdd(&local_i_fqueue->cnt_elem, -1);
					int idx_in_queue = atomicAdd(&local_i_fqueue->head, 1);
					int idx_anchor_in_pool =
							local_i_fqueue->avail_idx_bkt[idx_in_queue
									% local_i_fqueue->capacity];
					if (idx_anchor_in_pool == -1)
					{
						printf("local_f_fqueue empty error!");
						atomicAdd(&cnt_over_seg_update, 1);
						break;
					}
					else
					{
						atomicExch(
								&local_i_fqueue->avail_idx_bkt[idx_in_queue
										% local_i_fqueue->capacity], -1);
					}
					place_holder_update_dispatch_local[place_holder_update_dispatch_local[0]
							+ 1 + tid_w] = idx_anchor_in_pool;
				}
				if (tid_w == 0)
				{
					place_holder_update_dispatch_local[0] = UPDATE_DISPATCH_SEG
							- 1;
				}
			}
#endif

			ins_update = &buffer_update[anchor];
			if (ins_update->oid > p_config->max_obj_num)
			{
				anchor += gridDim.x * blockDim.x;
				continue;
			}
			if (ins_update->x == 0 && ins_update->y == 0)
				printf("load data error\n");

			//ins_update = &buffer_update[anchor];
			oid = ins_update->oid;
			p_sie = &(local_seci->index[oid]); // p_sie refers to a SIEntry object; local_seci refers to a SecIndex object;

			if (p_sie->idx_cell >= 0)
			{
				p_cell_old = &local_index->arr_cell[p_sie->idx_cell]; //local_index refers to the Grid
			}
			else
			{
#if IGNORE_CNT==0
				atomicAdd(&exp_old_cell_null, 1);
#endif
				p_cell_old = NULL;
			}

			p_cell_new = local_index->getCellByXY(ins_update->x, ins_update->y);

			if (p_cell_new == NULL)
			{
				atomicAdd(&exp_new_cell_null, 1);
			}
			// (p_cell_new != NULL) && (p_cell_new != p_cell_old) insert
			// (p_cell_old != NULL) && (p_cell_old == p_cell_new) refresh

			if (p_cell_new != NULL)
			{
				local_fqueue_u = local_i_fqueue;
				//int *local_cnt;
				if (p_cell_new != p_cell_old)
				{
					//local_cnt = local_i_cnt;
					idx_tmp_0 = p_cell_new->subgrid;
					mtx_idx = local_i;
				}
				else
				{
					//local_cnt = local_f_cnt;
					idx_tmp_0 = p_cell_new->subgrid + TOT_VGROUP_UPDATE;
					mtx_idx = local_f;
				}
				//atomicAdd(&local_cnt[p_cell_new->subgrid], 1);    //p_cell_new->subgrid refers to the Cell ID
				idx_last = 0;							//last
				while (true)
				{
					idx_last = local_node_i->last[idx_tmp_0];
					if (local_node_i->cnt[idx_last] < local_node_i->LEN)
					{
						idx_tmp = atomicAdd(&(local_node_i->cnt[idx_last]), 1);
						//idx_tmp = atomicAdd(&(local_node_i->cnt[idx_last]), -1);
						if (idx_tmp + 1 >= local_node_i->LEN)
						{
							atomicAdd(&(local_node_i->cnt[idx_last]), -1);
#if SEG_CACHE == 0
							tmp_me = &(local_node_i->mes[idx_last]);
							if (tmp_me->next == -1)
							{
								if (atomicCAS(&tmp_me->lock, 0, tid + 1) == 0)
								{
									if (atomicCAS(&tmp_me->next, -1, -1) == -1)
									{
										int idx_anchor_in_pool;
										int local_cache_page_idx =
												atomicAdd(
														place_holder_update_dispatch_local,
														-1);
										if (local_cache_page_idx <= 0)
										{
											int cnt_elem = atomicAdd(
													&local_fqueue_u->cnt_elem,
													-1);
											int idx_in_queue = atomicAdd(
													&local_fqueue_u->head, 1);
											idx_anchor_in_pool =
													local_fqueue_u->avail_idx_bkt[idx_in_queue
															% local_fqueue_u->capacity];
											if (idx_anchor_in_pool == -1)
											{
												printf(
														"local_i&f_fqueue empty error!\t");
												atomicAdd(&cnt_over_seg_update,
														1);
												atomicCAS(&tmp_me->lock,
														tid + 1, 0);
												break;
											}
											else
											{
												atomicExch(
														&local_fqueue_u->avail_idx_bkt[idx_in_queue
																% local_fqueue_u->capacity],
														-1);
											}
										}
										else
										{
											//atomicAdd(&exp_new_cell_null, 1);
											idx_anchor_in_pool =
													place_holder_update_dispatch_local[local_cache_page_idx];
										}
										atomicExch(&tmp_me->next,
												idx_anchor_in_pool);
										atomicExch(
												&(local_node_i->last[idx_tmp_0]),
												idx_anchor_in_pool);
									}
									atomicCAS(&tmp_me->lock, tid + 1, 0);
								}
							}
#else
							atomicAdd(&cnt_over_seg_update, 1);
							break;
#endif
						}
						else
						{
							local_node_i->pool[idx_last * local_node_i->LEN
									+ idx_tmp] = *ins_update;
							break;
						}
						//break;
					}
#if SEG_CACHE == 1
					break;
#endif
				}
			}

			if ((p_cell_new != NULL) && (p_cell_old != NULL)
					&& (p_cell_old != p_cell_new))
			{
				local_fqueue_u = local_d_fqueue;
				while (true)
				{
					idx_last = local_node_d->last[p_cell_old->subgrid];
					if (local_node_d->cnt[idx_last] < local_node_d->LEN)
					{
						idx_tmp = atomicAdd(&(local_node_d->cnt[idx_last]), 1);
						//idx_tmp = atomicAdd(&(local_node_d->cnt[idx_last]), -1);
						if (idx_tmp + 1 >= local_node_d->LEN)
						{
							atomicAdd(&(local_node_d->cnt[idx_last]), -1);
#if SEG_CACHE == 0
							tmp_me = &(local_node_d->mes[idx_last]);
							if (tmp_me->next == -1)
							{
								if (atomicCAS(&tmp_me->lock, 0, tid + 1) == 0)
								{
									if (atomicCAS(&tmp_me->next, -1, -1) == -1)
									{
										int idx_anchor_in_pool;
										int cnt_elem = atomicAdd(
												&local_fqueue_u->cnt_elem, -1);
										int idx_in_queue = atomicAdd(
												&local_fqueue_u->head, 1);
										idx_anchor_in_pool =
												local_fqueue_u->avail_idx_bkt[idx_in_queue
														% local_fqueue_u->capacity];
										if (idx_anchor_in_pool == -1)
										{
											printf(
													"local_d_fqueue empty error!");
											atomicAdd(&cnt_over_seg_update, 1);
											atomicCAS(&tmp_me->lock, tid + 1,
													0);
											break;
										}
										else
										{
											atomicExch(
													&local_fqueue_u->avail_idx_bkt[idx_in_queue
															% local_fqueue_u->capacity],
													-1);
										}
										atomicExch(&tmp_me->next,
												idx_anchor_in_pool);
										atomicExch(
												&(local_node_d->last[p_cell_old->subgrid]),
												idx_anchor_in_pool);
									}
									atomicCAS(&tmp_me->lock, tid + 1, 0);
								}
							}
#else
							atomicAdd(&cnt_over_seg_update, 1);
							break;
#endif
						}
						else
						{
							if (idx_tmp > local_node_d->LEN)
								printf("idx_tmp error\n");
							local_node_d->pool[idx_last * local_node_d->LEN
									+ idx_tmp] = ins_update->oid;
							break;
						}
						//break;
					}
#if SEG_CACHE == 1
					break;
#endif
				}
			}
			anchor += blockDim.x * gridDim.x;
		}

		offset_buffer_update += buffer_block_size_update;
		__threadfence_system();
		cudaDeviceSynchronize();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();
		if (tid == 0)
			atomicExch(
					update_map
							+ (int) (offset_buffer_update_rec
									/ p_config->buffer_block_size)
									% p_config->buffer_update_round, 0);

		//if(tid == 0) printf("local_i_fqueue used %d \n", local_i_fqueue->capacity-local_fqueue_u->cnt_elem);

#if CHECK_UPDATE_MEMPOOL==1
		anchor = tid;
		while(anchor < TOT_VGROUP_UPDATE)
		{
			int tmp_idx = node_enqueue_update->mtx_insert_idx[anchor];
			int tmp_cnt = 0;
			do
			{
				MemItem<UpdateType>* qn_cursor = &node_enqueue_update->mtx_insert_node[tmp_idx];
				tmp_cnt += qn_cursor->cnt;
				tmp_idx = qn_cursor->next;
			}while(tmp_idx != -1);
			atomicAdd(&cnt_over_seg_update, tmp_cnt);
			atomicAdd(&cnt_over_seg_query, local_i_cnt[anchor]);
			anchor+=gridDim.x*blockDim.x;
		}
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence);

		}
		__syncthreads();
#endif

		if (tid == 0)
		{
#if CHECK_UPDATE_MEMPOOL==1
			printf("\nInsert:\n");
			for(int i = 0; i<TOT_VGROUP_UPDATE; i+=100)
			{
				printf("%d ",local_i_cnt[i]);
			}
			printf("\n\n");

			printf("\nInsert check:\n");
			for(int i = 0; i<TOT_VGROUP_UPDATE; i+=100)
			{
				int tmp_idx = node_enqueue_update->mtx_insert_idx[i];
				int tmp_cnt = 0;
				do
				{
					MemItem<UpdateType>* qn_cursor = &node_enqueue_update->mtx_insert_node[tmp_idx];
					tmp_cnt += qn_cursor->cnt;
					tmp_idx = qn_cursor->next;
				}while(tmp_idx != -1);
				printf("%d ",tmp_cnt);
			}
			printf("\n\n");
#endif
			end = clock64();
			long long int temp_sum_time = ((double) (end - start));
			atomicExch(&dis_time_per_period, (int) (end - start));
			if (offset_buffer_update > QUERY_SKIP_NUM)
				distribute_sumtime += temp_sum_time;
			printf("d1 %.4f ms\n",
					(double) temp_sum_time / (double) p_config->clockRate);

			atomicExch(&offset_buffer_update_rec, offset_buffer_update);

			if (flag_switch_dist == 0)
			{
				atomicExch(&req_cache_update->cnt0, cnt_enqueue_update);
				atomicExch(&req_cache_update->token0, 1);
			}
			if (flag_switch_dist == 1)
			{
				atomicExch(&req_cache_update->cnt1, cnt_enqueue_update);
				atomicExch(&req_cache_update->token1, 1);
			}
			node_enqueue_update = NULL;
			atomicAdd(&cnt_enqueue_update, 1);
		}

		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		if (offset_buffer_update <= QUERY_SKIP_NUM)
		{
			if (tid == 0)
			{
				atomicAdd(&cnt_enqueue_query, 1);
				atomicAdd(&cnt_dequeue_query, 1);
			}
			continue;
		}
		if (offset_buffer_update <= QUERY_SKIP_NUM + buffer_block_size_update)
		{
			if (tid == 0)
			{
				//start = clock64();
			}
		}

		// ------------------------------------------------------------------------------------------------------------
		// Distribute Queries
		if (offset_buffer_query + buffer_block_size_query >= TOTAL_QUERY)
		{
			buffer_block_size_query = TOTAL_QUERY - offset_buffer_query;
		}

		if (flag_switch_dist == 0)
		{
			while (req_cache_query->token0 == 1)
			{
				if (tid == 0)
					exp_hunger_dist1++;
			}
		}
		else if (flag_switch_dist == 1)
		{
			while (req_cache_query->token1 == 1)
			{
				if (tid == 0)
					exp_hunger_dist1++;
			}
		}

		start = clock64();
		if (tid == 0)
		{
			if (flag_switch_dist == 0)
			{
				node_enqueue_query = &req_cache_query->array[0];
			}
			else if (flag_switch_dist == 1)
			{
				node_enqueue_query = &req_cache_query->array[1];
			}
			node_enqueue_query->tot_cells_covered = 0;
			node_enqueue_query->bound_btw_cell[0] = 0;
			node_enqueue_query->buffer_block_size_query =
					buffer_block_size_query;

			atomicExch(&cnt_singular, 0);
		}

		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		node_enqueue_query_local = node_enqueue_query;
		flag_cells_covered = node_enqueue_query->flag_cells_covered;
		idx_cells_covered = node_enqueue_query->idx_cells_covered;
		cnt_queries_per_cell = node_enqueue_query->cnt_queries_per_cell;
		queries_per_cell = node_enqueue_query->queries_per_cell;
		buffer_block_query = node_enqueue_query->buffer_block_query;
		node_enqueue_query->offset_buffer_query = offset_buffer_query;

		if (flag_switch_version == 0)
		{
			p_grid = index_A;
		}
		else if (flag_switch_version == 1)
		{
			p_grid = index_B;
		}

		anchor = tid;
		while (anchor < TOT_CELLS)
		{
			cnt_queries_per_cell[anchor] = 0;
			anchor += blockDim.x * gridDim.x;
		}
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		p_buffer_block_query = buffer_query + offset_buffer_query;

#if (QUERY_PATTERN!=3 && QUERY_PATTERN!=30)

		for (int i = 0; i < 1; i++)
		{
			fqueue_local = node_enqueue_query->fqueue_query;
			anchor = tid;
			while (anchor < fqueue_local->capacity)
			{
				fqueue_local->avail_idx_bkt[anchor] = anchor;
				anchor += blockDim.x * gridDim.x;
			}
			if (tid == 0)
			{
				atomicExch(&fqueue_local->cnt_elem, fqueue_local->capacity);
				atomicExch(&fqueue_local->head, 0);
				atomicExch(&fqueue_local->rear, 0);
			}
		}
		__threadfence_system();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		fqueue_local = node_enqueue_query->fqueue_query;
		mtx_idx = node_enqueue_query->mtx_query_idx;
		anchor = tid;
		idx_tmp = fqueue_local->head;
		while (anchor < TOT_VGROUP_UPDATE)
		{
			fqueue_local->avail_idx_bkt[(idx_tmp + anchor)
					% fqueue_local->capacity] = -1;
			mtx_idx[anchor] = idx_tmp + anchor;
			node_enqueue_query->mtx_query_nodes->last[anchor] = idx_tmp
					+ anchor;
			anchor += blockDim.x * gridDim.x;
		}
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		if (tid == 0)
		{
			atomicAdd(&fqueue_local->cnt_elem, -TOT_VGROUP_UPDATE);
			atomicAdd(&fqueue_local->head, TOT_VGROUP_UPDATE);
			cudaMemcpyAsync(node_enqueue_query->mtx_query_nodes->mes,
					node_enqueue_query->mtx_query_nodes_bak,
					sizeof(MemElement) * node_enqueue_query->q_size,
					cudaMemcpyDeviceToDevice);
			cudaMemsetAsync(node_enqueue_query->mtx_query_nodes->cnt, 0,
					sizeof(int) * node_enqueue_query->q_size);
		}
		cudaDeviceSynchronize();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		anchor = tid;

#if (USE_MULTIQUEUE == 1)

		anchor = tid;
		while (anchor < QUERY_TYPE_POOL_SIZE)
		{
			d_qd_anchor_pool[anchor] = -1;
			anchor += blockDim.x * gridDim.x;
		}

		if (tid == 0)
		{
			atomicExch(&d_queue_idx_anchor_free->cnt_elem,
					d_queue_idx_anchor_free->capacity);
			atomicExch(&d_queue_idx_anchor_free->head, 0);
			atomicExch(&d_queue_idx_anchor_free->rear, 0);
		}
		anchor = tid;
		while (anchor < d_queue_idx_anchor_free->capacity)
		{
			atomicExch(&d_queue_idx_anchor_free->avail_idx_bkt[anchor], anchor);
			anchor += blockDim.x * gridDim.x;
		}

		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		anchor = tid;
		while (anchor < MQUEUE_SIZE)
		{
			atomicAdd(&d_queue_idx_anchor_free->cnt_elem, -1);
			int idx_in_queue = atomicAdd(&d_queue_idx_anchor_free->head, 1);
			int idx_anchor_in_pool =
					d_queue_idx_anchor_free->avail_idx_bkt[idx_in_queue
							% d_queue_idx_anchor_free->capacity];
			d_queue_idx_anchor_free->avail_idx_bkt[idx_in_queue
					% d_queue_idx_anchor_free->capacity] = -1;
			MemItem<QueryType>* qn = &d_qd_obj_pool[idx_anchor_in_pool];
			qn->id = idx_anchor_in_pool;
			qn->cnt = 0;
			qn->queuelen = 1;
			qn->len = QT_SIZE;
			qn->next = -1;
			qn->last = idx_anchor_in_pool;
			qn->lock = 0;
			multiqueue[anchor] = idx_anchor_in_pool;
			anchor += gridDim.x * blockDim.x;
		}
		anchor = tid;
		while (anchor < 512)
		{
			cache_memory_idx_query_dispatch[anchor] = -1;
			anchor += gridDim.x * blockDim.x;
		}
		__threadfence_system();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();
		anchor = tid;
		while (anchor < buffer_block_size_query)
		{
			req_query = p_buffer_block_query[anchor];

			xmin = req_query.minX;
			ymin = req_query.minY;
			xmax = req_query.maxX;
			ymax = req_query.maxY;

			p_cell_lb = p_grid->getNearestCellByXY(xmin, ymin);
			p_cell_rt = p_grid->getNearestCellByXY(xmax, ymax);

			left_bottom = p_cell_lb->idx;
			right_top = p_cell_rt->idx;
			cell_num = p_grid->cell_num;
			row_start = right_top / cell_num;
			col_end = right_top % cell_num;
			row_end = left_bottom / cell_num;
			col_start = left_bottom % cell_num;

			int cellNum = (row_end - row_start + 1) * (col_end - col_start + 1);
			int count = cellNum / QUEUE_SEG_LEN + 1;
			if (count >= MQUEUE_SIZE)
			{
				printf("count over MQUEUE_SIZE\n");
				continue;
			}
			MemItem<QueryType>* qn_start = &d_qd_obj_pool[multiqueue[count]];
			MemItem<QueryType>* qn_last;
			bool errorFlag = false;
			while (true)
			{
				qn_last = &d_qd_obj_pool[qn_start->last];
				if (qn_last->cnt >= qn_last->len)
				{
					while (qn_last->cnt >= qn_last->len)
						ins_update = &buffer_update[anchor];		//donothing
					continue;
				}

				if (*cache_memory_idx_query_dispatch_local == -1)
				{
					if (tid_w == 0)
					{
						int cnt_elem = atomicAdd(
								&d_queue_idx_anchor_free->cnt_elem, -1);
						int idx_in_queue = atomicAdd(
								&d_queue_idx_anchor_free->head, 1);
						int idx_anchor_in_pool =
								d_queue_idx_anchor_free->avail_idx_bkt[idx_in_queue
										% d_queue_idx_anchor_free->capacity];
						if (idx_anchor_in_pool == -1)
						{
							printf("idx_anchor_in_pool empty error!");
							errorFlag = true;
							break;
						}
						else
						{
							atomicExch(
									&d_queue_idx_anchor_free->avail_idx_bkt[idx_in_queue
											% d_queue_idx_anchor_free->capacity],
									-1);
							MemItem<QueryType>* newQn =
									&d_qd_obj_pool[idx_anchor_in_pool];
							atomicExch(&newQn->id, idx_anchor_in_pool);
							atomicExch(&newQn->cnt, 0);
							atomicExch(&newQn->queuelen, 1);
							atomicExch(&newQn->len, QT_SIZE);
							atomicExch(&newQn->next, -1);
							atomicExch(&newQn->last, idx_anchor_in_pool);
							atomicExch(&newQn->lock, 0);
							atomicExch(cache_memory_idx_query_dispatch_local,
									idx_anchor_in_pool);
						}
					}
				}

				idx_tmp = atomicAdd(&qn_last->cnt, 1);
				if (idx_tmp + 1 >= qn_last->len)
				{
					atomicAdd(&qn_last->cnt, -1);
					idx_tmp_0 = atomicExch(
							cache_memory_idx_query_dispatch_local, -1);
					if (idx_tmp_0 > 0)
					{
						if (qn_last->next == -1)
						{
							if (atomicCAS(&qn_last->lock, 0, 1) == 0)
							{
								if (atomicCAS(&qn_last->next, -1, -1) == -1)
								{
									atomicExch(&qn_last->next, idx_tmp_0);
									atomicExch(&qn_start->last, idx_tmp_0);
									atomicAdd(&qn_start->queuelen, 1);
									atomicCAS(&qn_last->lock, 1, 0);
									continue;
								}
								atomicCAS(&qn_last->lock, 1, 0);
							}
						}
						atomicExch(cache_memory_idx_query_dispatch_local,
								idx_tmp_0);
					}
					continue;
				}
				break;
			}
			if (!errorFlag)
			{
				qn_last->pool[idx_tmp] = req_query;
				qn_last->cache_anchor[idx_tmp] = anchor;
			}
#if IGNORE_CNT==0
			atomicAdd(&cnt_queries, 1);
#endif
			anchor += blockDim.x * gridDim.x;
		}

		__threadfence_system();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;

		}
		__syncthreads();

		anchor = tid;
		while (anchor < blockDim.x * gridDim.x)
		{
			if (*cache_memory_idx_query_dispatch_local != -1)
			{
				if (tid_w == 0)
				{
					*cache_memory_idx_query_dispatch_local = -1;
				}
			}
			anchor += blockDim.x * gridDim.x;
		}

#if CHECK_QUERY_MEMPOOL==1
		if(tid == 0)
		{
			atomicExch(&query_sum_formempool, 0);
		}
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence);
		}
		__syncthreads();
		anchor = tid;
		while(anchor < MQUEUE_SIZE)
		{
			int cursor_idx = multiqueue[anchor];
			MemItem<QueryType>* qn_cursor;
			int query_cnt_temp = 0;
			while(cursor_idx != -1)
			{
				qn_cursor = &d_qd_obj_pool[cursor_idx];
				query_cnt_temp += qn_cursor->cnt;
				cursor_idx = qn_cursor->next;
			}
			printf("%d ", query_cnt_temp);
			atomicAdd(&query_sum_formempool, query_cnt_temp);
			anchor += gridDim.x*blockDim.x;
		}
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence);
		}
		__syncthreads();
		if(tid == 0)
		{
			printf("\n");
			printf("SUM : %d\n", query_sum_formempool);
		}

#endif

		if (tid == 0)
		{
			atomicExch(&cursor_distribute_wrap, 0);
			atomicExch(&cnt_distribute_wrap, 0);
		}

		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

		MemElementCollection<int>* local_node_q =
				node_enqueue_query->mtx_query_nodes;
		CircularQueue* local_fqueue_q = node_enqueue_query->fqueue_query;
		while (true)
		{
			if (cursor_distribute_wrap * QUERY_DISPATCH_WRAP_SIZE
					>= QUERY_TYPE_POOL_SIZE)
				break;
			if (tid_w == 0)
			{
				*place_holder_query_dispatch_local = QUERY_DISPATCH_WRAP_SIZE
						* atomicAdd(&cursor_distribute_wrap, 1);
			}

			if (*place_holder_query_dispatch_local >= QUERY_TYPE_POOL_SIZE)
			{
				break;		//exit;
			}
			if (*place_holder_query_dispatch_local < 0)
			{
				break;
			}

			int anchor_end = QUERY_TYPE_POOL_SIZE
					- *place_holder_query_dispatch_local;
			if (anchor_end > QUERY_DISPATCH_WRAP_SIZE)
				anchor_end = QUERY_DISPATCH_WRAP_SIZE;
			d_qd_anchor_local = d_qd_anchor_pool
					+ (*place_holder_query_dispatch_local);
			d_qd_query_type_local = d_qd_query_type_pool
					+ (*place_holder_query_dispatch_local);
			anchor = tid_w;
			int anchor_idx;
			while (anchor < QUERY_DISPATCH_WRAP_SIZE)
			{
				if (anchor >= anchor_end)
					break;
				anchor_idx = d_qd_anchor_local[anchor];
				if (anchor_idx < 0)
				{
					break;
				}

				req_query = d_qd_query_type_local[anchor];
				xmin = req_query.minX;
				ymin = req_query.minY;
				xmax = req_query.maxX;
				ymax = req_query.maxY;
				p_cell_lb = p_grid->getNearestCellByXY(xmin, ymin);
				p_cell_rt = p_grid->getNearestCellByXY(xmax, ymax);

				left_bottom = p_cell_lb->idx;
				right_top = p_cell_rt->idx;
				cell_num = p_grid->cell_num;
				row_start = right_top / cell_num;
				col_end = right_top % cell_num;
				row_end = left_bottom / cell_num;
				col_start = left_bottom % cell_num;

				anchor_idx = atomicAdd(&cnt_distribute_wrap, 1);

				int cellNum = (row_end - row_start + 1)
						* (col_end - col_start + 1);
				for (int k = 0; k < cellNum; k++)
				{
					int i = row_start + k / (col_end - col_start + 1);
					int j = col_start + k % (col_end - col_start + 1);
					idx_cell = i * EDGE_CELL_NUM + j;
					idx_tmp = atomicAdd(&cnt_queries_per_cell[idx_cell], 1);
#if IGNORE_CNT==0
					atomicAdd(&check_tot_covered, 1);
#endif

#if SEG_CACHE == 0
					if (idx_tmp >= local_node_q->LEN)
					{
						atomicAdd(&cnt_queries_per_cell[idx_cell], -1);
						atomicAdd(&cnt_over_seg_query, 1);
						continue;
					}
#endif
					if (idx_tmp < 0)
						continue;
					if (atomicCAS(&flag_cells_covered[idx_cell], 0, 1) == 0)
					{
						//atomicExch(&flag_cells_covered[idx_cell], 1);
						idx_tmp_0 = atomicAdd(
								&node_enqueue_query->tot_cells_covered, 1);
						idx_cells_covered[idx_tmp_0] = idx_cell;
					}
					//queries_per_cell[idx_cell * LEN_SEG_CACHE_QUERY + idx_tmp] = anchor_idx;

					while (true)
					{
						idx_last = local_node_q->last[idx_cell];
						if (local_node_q->cnt[idx_last] < local_node_q->LEN)
						{
							idx_tmp = atomicAdd(&(local_node_q->cnt[idx_last]),
									1);
							if (idx_tmp + 1 >= local_node_q->LEN)
							{
								atomicAdd(&(local_node_q->cnt[idx_last]), -1);
#if SEG_CACHE == 0
								tmp_me = &(local_node_q->mes[idx_last]);
								if (tmp_me->next == -1)
								{
									if (atomicCAS(&tmp_me->lock, 0, tid + 1)
											== 0)
									{
										if (atomicCAS(&tmp_me->next, -1, -1)
												== -1)
										{
											int idx_anchor_in_pool;
											int cnt_elem = atomicAdd(
													&local_fqueue_q->cnt_elem,
													-1);
											int idx_in_queue = atomicAdd(
													&local_fqueue_q->head, 1);
											idx_anchor_in_pool =
													local_fqueue_q->avail_idx_bkt[idx_in_queue
															% local_fqueue_q->capacity];
											if (idx_anchor_in_pool == -1)
											{
												printf(
														"local_fqueue_q empty error!");
												atomicAdd(&cnt_over_seg_query,
														1);
												atomicCAS(&tmp_me->lock,
														tid + 1, 0);
												break;
											}
											else
											{
												atomicExch(
														&local_fqueue_q->avail_idx_bkt[idx_in_queue
																% local_fqueue_q->capacity],
														-1);
											}
											atomicExch(&tmp_me->next,
													idx_anchor_in_pool);
											atomicExch(
													&(local_node_q->last[idx_cell]),
													idx_anchor_in_pool);
										}
										atomicCAS(&tmp_me->lock, tid + 1, 0);
									}
								}
#else
								atomicAdd(&cnt_over_seg_query, 1);
								break;
#endif
							}
							else
							{
								if (idx_tmp > local_node_q->LEN)
									printf("idx_tmp error\n");
								local_node_q->pool[idx_last * local_node_q->LEN
										+ idx_tmp] = anchor_idx;
								break;
							}
						}
					}
				}

				buffer_block_query[anchor_idx] = req_query;
				check_dist_query_local++;
#if IGNORE_CNT==0
				atomicAdd(&cnt_queries, 1);
#endif
				__threadfence_system();
				anchor += WRAPSIZE;
			}
		}

		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

#else
		//lwh
        MemElementCollection<int> *local_node_q =
                node_enqueue_query->mtx_query_nodes;
        /*CircularQueue *local_fqueue_q = node_enqueue_query->fqueue_query;*/
        int anchor_idx;
		anchor = tid;
		while (anchor < buffer_block_size_query)
		{
			req_query = p_buffer_block_query[anchor];
			//lwh
			anchor_idx = d_qd_anchor_local[anchor];

			xmin = req_query.minX;
			ymin = req_query.minY;
			xmax = req_query.maxX;
			ymax = req_query.maxY;

			p_cell_lb = p_grid->getNearestCellByXY(xmin,ymin);
			p_cell_rt = p_grid->getNearestCellByXY(xmax,ymax);

			left_bottom = p_cell_lb->idx;
			right_top = p_cell_rt->idx;
			cell_num = p_grid->cell_num;
			row_start = right_top / cell_num;
			col_end = right_top % cell_num;
			row_end = left_bottom / cell_num;
			col_start = left_bottom % cell_num;

            anchor_idx = atomicAdd(&cnt_distribute_wrap, 1);

			int cellNum = (row_end-row_start+1)*(col_end-col_start+1);
			int count = cellNum/QUEUE_SEG_LEN + 1;
			if(count >= MQUEUE_SIZE)
			{	printf("count over MQUEUE_SIZE\n");continue;}
#if IGNORE_CNT==0
			atomicAdd(multiqueue+count,1);
#endif

#if USE_DPPROCESS==1
			DpProcess<<<1, cellNum>>>(place_holder_query_dispatch_local, anchor, row_start, row_end, col_start, col_end, dev_p_gconfig, node_enqueue_query);
			cudaDeviceSynchronize();
#else
			for(int k = 0; k < cellNum; k++)
			{
				int i = row_start + k/(col_end-col_start+1);
				int j = col_start + k%(col_end-col_start+1);
				idx_cell = i * EDGE_CELL_NUM + j;
				idx_tmp = atomicAdd(&cnt_queries_per_cell[idx_cell], 1);

#if IGNORE_CNT==0
				atomicAdd(&check_tot_covered, 1);
#endif
#if SEG_CACHE == 0
				if(idx_tmp >= local_node_q->LEN)
				{
					atomicAdd(&cnt_queries_per_cell[idx_cell], -1);
					atomicAdd(&cnt_over_seg_query, 1);
					continue;
				}
#endif
				if (atomicCAS(&flag_cells_covered[idx_cell], 0, 1) == 0)
				{
					//atomicExch(&flag_cells_covered[idx_cell], 1);
					idx_tmp_0 = atomicAdd(&node_enqueue_query->tot_cells_covered, 1);
					idx_cells_covered[idx_tmp_0] = idx_cell;
				}
				//queries_per_cell[idx_cell * LEN_SEG_CACHE_QUERY + idx_tmp] = anchor;

				while(true)
				{
					idx_last = local_node_q->last[idx_cell];
					if(local_node_q->cnt[idx_last] < local_node_q->LEN)
					{
						idx_tmp = atomicAdd(&(local_node_q->cnt[idx_last]), 1);
						if(idx_tmp + 1 >= local_node_q->LEN)
						{
							atomicAdd(&(local_node_q->cnt[idx_last]), -1);
#if SEG_CACHE == 0
							tmp_me = &(local_node_q->mes[idx_last]);
							if(tmp_me->next == -1)
							{
								if(atomicCAS(&tmp_me->lock,0,tid+1) == 0)
								{
									if(atomicCAS(&tmp_me->next,-1,-1) == -1)
									{
										int idx_anchor_in_pool;
										int cnt_elem = atomicAdd(&local_fqueue_q->cnt_elem, -1);
										int idx_in_queue = atomicAdd(&local_fqueue_q->head, 1);
										idx_anchor_in_pool = local_fqueue_q->avail_idx_bkt[idx_in_queue % local_fqueue_q->capacity];
										if(idx_anchor_in_pool == -1)
										{
											printf("local_fqueue_q empty error!");
											atomicAdd(&cnt_over_seg_query, 1);
											atomicCAS(&tmp_me->lock,tid+1,0);
											break;
										}
										else
										{
											atomicExch(&local_fqueue_q->avail_idx_bkt[idx_in_queue % local_fqueue_q->capacity], -1);
										}
										atomicExch(&tmp_me->next, idx_anchor_in_pool);
										atomicExch(&(local_node_q->last[idx_cell]), idx_anchor_in_pool);
									}
									atomicCAS(&tmp_me->lock,tid+1,0);
								}
							}
#else
							atomicAdd(&cnt_over_seg_query, 1);
							break;
#endif
						}
						else
						{
							if(idx_tmp > local_node_q->LEN)
							printf("idx_tmp error\n");
							local_node_q->pool[idx_last*local_node_q->LEN + idx_tmp] = anchor_idx;
							break;
						}
					}
				}
			}
#endif
			buffer_block_query[anchor] = req_query;
			check_dist_query_local++;

#if IGNORE_CNT==0
			atomicAdd(&cnt_queries,1);
#endif
			anchor += blockDim.x * gridDim.x;
		}
#endif
//		if(tid == 0)
//		{
//			for(int k = 0; k<MQUEUE_SIZE; k++){
//				printf("%d ", multiqueue[k]);
//			}
//			printf("\n");
//		}

#else //QUERY PATTHERN ELSE
		MemElementCollection<int>* local_node_q = node_enqueue_query->mtx_query_nodes;
		atomicExch(&local_node_q->globalCnt, 0);
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence);
		}
		__syncthreads();
		anchor = tid;
		while (anchor < buffer_block_size_query)
		{
			req_query = p_buffer_block_query[anchor];
			atomicAdd(&local_node_q->globalCnt,1);
			buffer_block_query[anchor] = req_query;
			check_dist_query_local++;
#if IGNORE_CNT==0
			atomicAdd(&cnt_queries,1);
#endif
			anchor += blockDim.x * gridDim.x;
		}
#endif
		check_cnt_enqueue++;

		__threadfence_system();
		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

//		anchor = tid;
//		while(anchor < tot_cells_covered)
//		{
//			flag_cells_covered[ idx_cells_covered[anchor] ] = 0;
//			anchor += blockDim.x * gridDim.x;
//		}
//
//		__syncthreads();
//		barrier_fence += barrier_step;
//		if (tid_b == 0)
//		{
//			atomicAdd(&barrier_dist, 1);
//			while (barrier_dist < barrier_fence);
//		}
//		__syncthreads();

		offset_buffer_query += buffer_block_size_query;
		if (tid == 0)
		{
			end = clock64();
			long long int temp_sum_time = (end - start);
			atomicAdd(&dis_time_per_period, (int) (end - start));
			distribute_sumtime += temp_sum_time;
			printf("d2 %.4f ms\n",
					(double) temp_sum_time / p_config->clockRate);
			atomicExch(&offset_buffer_query_rec, offset_buffer_query);
			if (flag_switch_dist == 0)
			{
				atomicExch(&req_cache_query->cnt0, cnt_enqueue_query);
				atomicExch(&req_cache_query->token0, 1);
			}
			if (flag_switch_dist == 1)
			{
				atomicExch(&req_cache_query->cnt1, cnt_enqueue_query);
				atomicExch(&req_cache_query->token1, 1);
			}
			node_enqueue_query = NULL;
			atomicAdd(&cnt_enqueue_query, 1);
		}

		__syncthreads();
		barrier_fence += barrier_step;
		if (tid_b == 0)
		{
			atomicAdd(&barrier_dist, 1);
			while (barrier_dist < barrier_fence)
				;
		}
		__syncthreads();

	}
	__syncthreads();
	barrier_fence += barrier_step;
	if (tid_b == 0)
	{
		atomicAdd(&barrier_dist, 1);
		while (barrier_dist < barrier_fence)
			;
	}
	__syncthreads();
	//end = clock64();
	if (tid == 0)
	{
		*dev_cnt_update = offset_buffer_update;
		*dev_cnt_query = offset_buffer_query;
	}
	if (tid == 0 && offset_buffer_update >= TOTAL_UPDATE)
	{
		atomicExch(&p_config->terminalFlag, 1);
		atomicExch(&buffer_exhausted, 1);
		if (start != -1)
		{
			printf("\nDistributorKernel clock: %f ms\n",
					((double) distribute_sumtime) / p_config->clockRate);
		}
		printf("\n");
		printf("len_seg_cache_update_local: %d\n", len_seg_cache_update_local);
		printf("cnt_enqueue_update: %d\n", cnt_enqueue_update);
		printf("cnt_enqueue_query: %d\n", cnt_enqueue_query);
		printf("exp_hunger_dist0: %d\n", exp_hunger_dist0);
		printf("exp_hunger_dist1: %d\n", exp_hunger_dist1);
		printf("offset_buffer_update: %d\n", offset_buffer_update);
		printf("dev_cnt_update: %d\n", *dev_cnt_update);
		printf("offset_buffer_query: %d\n", offset_buffer_query);
		printf("dev_cnt_query: %d\n", *dev_cnt_query);
		printf("len_seg_cache_query in Distribute Kernel: %d\n",
				LEN_SEG_CACHE_QUERY);
		printf("buffer_block_size_update: %d\n", buffer_block_size_update);
		printf("buffer_block_size_query: %d\n", buffer_block_size_query);

		printf("exp_new_cell_null: %d\n", exp_new_cell_null);
		printf("exp_update_in_spec_cell: %d\n", exp_update_in_spec_cell);
		printf("exp_old_cell_null: %d\n", exp_old_cell_null);

		printf("check_tot_covered: %d %d\n",
				check_tot_covered / cnt_enqueue_query, cnt_enqueue_query);
		printf("check_dist_query_local: %d\n", check_dist_query_local);
		printf("check_cnt_enqueue: %d\n", check_cnt_enqueue);
		printf("check_lb_null: %d\n", check_lb_null);
		printf("check_rt_null: %d\n", check_rt_null);
		printf("cnt_over_seg_update: %d\n", cnt_over_seg_update);
		printf("cnt_over_seg_query: %d\n", cnt_over_seg_query);
		printf("cnt_queries: %d\n", cnt_queries);

//		printf("Sizeof UpdateType: %d\n", sizeof(UpdateType));
//		printf("Sizeof QueryType: %d\n", sizeof(QueryType));
//		printf("Sizeof ObjBox: %d\n", sizeof(ObjBox));
//		printf("Sizeof QueryQNode: %d\n", sizeof(QueryQNode));
//		printf("Sizeof QueryType *: %d\n", sizeof(QueryType *));
//		printf("Sizeof int *: %d\n", sizeof(int *));
//		printf("Sizeof SIEntry: %d\n", sizeof(SIEntry));

	}

	__syncthreads();
	barrier_fence += barrier_step;
	if (tid_b == 0)
	{
		atomicAdd(&barrier_dist, 1);
		while (barrier_dist < barrier_fence)
			;
	}
	__syncthreads();

	if (tid == 0)
	{
		atomicExch(&barrier_dist, 0);
	}
}

