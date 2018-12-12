/************************************************************
* DeviceGlobalVar.cu
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 16, 2014*  4:11:56 PM* 
* Description:*  
* Licence:*
************************************************************/

#include "DeviceGlobalVar.cuh"


// update request queues & query request queues
//__device__ QueryQueue *query_req_queue;

__device__ UpdateCacheArea *req_cache_update = NULL;
__device__ int *update_map;

__device__ QueryCacheArea *req_cache_query = NULL;
__device__ int cnt_enqueue_update = 0;
__device__ int cnt_dequeue_update = 0;
__device__ int cnt_enqueue_query = 0;
__device__ int cnt_dequeue_query = 0;
__device__ UpdateQNode *node_dequeue_update = NULL;
__device__ UpdateQNode *node_enqueue_update = NULL;
__device__ QueryQNode *node_dequeue_query = NULL;
__device__ QueryQNode *node_enqueue_query = NULL;

//__device__ UpdateTypeForSort* array_i_forsort = NULL;
//__device__ int array_forsort_cnt = 0;
//__device__ int each_anchor_len = 0;
//__device__ int* each_anchor_cnt = NULL;

//__device__ unsigned int progress_distribute;
//__device__ unsigned int progress_update;
//__device__ unsigned int progress_query;

__device__ int gp_config_ready = 0; 

// Global variables for GPGrid
__device__ Grid *index_A = NULL;
__device__ Grid *index_B = NULL;
__device__ SecIndex *sec_index_A = NULL;
__device__ SecIndex *sec_index_B = NULL;
__device__ int flag_switch_version = 0;
__device__ int seg_switch_version = 2;
__device__ int query_switch_version = 0;
__device__ volatile int flag_switch_dist = 0;
__device__ volatile int flag_switch_update = 0;
__device__ volatile int flag_switch_query = 0;
__device__ ManagedMemory* DMM;

// sync signal
__device__ unsigned int barrier_dist = 0;
__device__ unsigned int barrier_update = 0;
__device__ unsigned int barrier_query = 0;

__device__ int cnt_req_query = 0;


__device__ int launch_signal = 0;
__device__ int sync_signal_query = 0;
__device__ int sync_signal_update = 0;

__device__ int exp_hunger_dist0 = 0;
__device__ int exp_hunger_dist1 = 0;
__device__ int exp_hunger_update = 0;
__device__ int exp_hunger_query = 0;
__device__ int exp_cell_full = 0;
__device__ int exp_idx_bkt_dequeue_error = 0;
__device__ int check_init_cells_a = 0;
__device__ int check_init_cells_b = 0;

__device__ int exp_cell_empty = 0;
__device__ int exp_new_cell_null = 0;
__device__ int exp_old_cell_null = 0;
__device__ int exp_update_in_spec_cell = 0;
__device__ int exp_anchor_x_cell = 0;
__device__ int exp_anchor_x_subgrid = 0;
__device__ int chk_flag_cells_covered = 0;
__device__ int chk_obj_covered = 0;


__device__ int update_set_over = 0;
__device__ int exp_hunger_dist_old = 0;
__device__ int exp_hunger_update_old = 0;

// in ObjBox.h
__device__ unsigned int counter_bucket = 0;
//__device__ ObjBox *obs_pool = NULL;

__device__ ObjBox *obs_pool_A = NULL;
__device__ ObjBox *obs_pool_B = NULL;


// in SIEntry.h
__device__ SIEntry *sie_array = NULL;
__device__ int pitch_sie = 0;


// in UpdateQNode.h
__device__ int len_seg_cache_update = 0;

// Grid
__device__ int exp_idx_x_subgrid_a = 0;
__device__ int exp_idx_x_subgrid_b = 0;
__device__ CircularQueue *queue_bkts_free = NULL;


// For debug
__device__ int check_tot_covered = 0;
__device__ int check_area_per_query = 0;
__device__ int cnt_singular = 0;
__device__ int check_dist_query = 0;


__device__ int* place_holder_update;
__device__ int* sync_holder_update;
__device__ int* place_holder_query;

__device__ int* place_holder_query_dispatch;
__device__ int* sync_holder_query_dispatch;
__device__ int* cache_memory_idx_query_dispatch;

__device__ int* place_holder_update_dispatch;

__device__ int cnt_over_seg_update = 0;
__device__ int cnt_over_seg_query = 0;
__device__ int cnt_over_bkt_update = 0;

__device__ int cnt_queries = 0;
__device__ int cnt_update_i = 0;
__device__ int cnt_update_d = 0;
__device__ int cnt_update_f = 0;



__device__ long long int query_sumtime = 0;
__device__ long long int update_sumtime = 0;
__device__ long long int distribute_sumtime = 0;
__device__ long long int copytime = 0;



__device__ int cursor_distribute_wrap = 0;
__device__ int cnt_distribute_wrap = 0;


__device__ int query_sum_formempool = 0;
__device__ int tot_obs_covered_forquery = 0;

__device__ int offset_buffer_query_rec = 0;
__device__ int offset_buffer_update_rec = 0;

__device__ int query_time_per_period = 1;
__device__ int update_time_per_period = 1;
__device__ int dis_time_per_period = 1;



__device__ int rebalance = 0;
__device__ int buffer_exhausted = 0;
__device__ int exit_query = 0;
__device__ int exit_update = 0;;





__device__ int cnt_malloc_obj_pool = 0;
__device__ int cnt_free_obj_pool = 0;


//__device__ int g_len_seg_cache_query = 0;

__device__ int* multiqueue;


