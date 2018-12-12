/************************************************************
* DeviceGlobalVar.h
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 16, 2014*  4:08:30 PM* 
* Description:*  
* Licence:*
************************************************************/

#ifndef DEVICEGLOBALVAR_H_
#define DEVICEGLOBALVAR_H_

#include "misc/SecIndex.cuh"
#include "misc/Grid.cuh"
#include "config/GConfig.h"
#include "misc/BaseStruct.h"
#include "misc/UpdateCacheArea.cuh"
#include "misc/QueryCacheArea.h"
#include "misc/Buffer.h"

//
extern Buffer<UpdateType> *host_buffer_update;
extern Buffer<QueryType> *host_buffer_query;

#ifdef linux
extern __device__ int gp_config_ready;

// buffer on GPU
//extern __device__ UpdateType *buffer_update;
//extern __device__ QueryType *buffer_query;

// update request queues & query request queues
//extern __device__ QueryQueue *query_req_queue;
extern __device__ UpdateCacheArea *req_cache_update;
extern __device__ int *update_map;

extern __device__ QueryCacheArea *req_cache_query;
extern __device__ int cnt_enqueue_update;
extern __device__ int cnt_dequeue_update;
extern __device__ int cnt_enqueue_query;
extern __device__ int cnt_dequeue_query;
extern __device__ UpdateQNode *node_dequeue_update;
extern __device__ UpdateQNode *node_enqueue_update;
extern __device__ QueryQNode *node_dequeue_query;
extern __device__ QueryQNode *node_enqueue_query;

//extern __device__ UpdateTypeForSort* array_i_forsort;
//extern __device__ int array_forsort_cnt;
//extern __device__ int each_anchor_len;
//extern __device__ int* each_anchor_cnt;

//extern __device__ unsigned int progress_distribute;
//extern __device__ unsigned int progress_update;
//extern __device__ unsigned int progress_query;


// Global variables for GPGrid
extern __device__ Grid *index_A;
extern __device__ Grid *index_B;
extern __device__ SecIndex *sec_index_A;
extern __device__ SecIndex *sec_index_B;
extern __device__ int flag_switch_version;
extern __device__ int seg_switch_version;
extern __device__ int query_switch_version;

extern __device__ volatile int flag_switch_dist;
extern __device__ volatile int flag_switch_update;
extern __device__ volatile int flag_switch_query;
extern __device__ ManagedMemory* DMM;


// sync signals
extern __device__ unsigned int barrier_dist;
extern __device__ unsigned int barrier_update;
extern __device__ unsigned int barrier_query;

extern __device__ int cnt_req_query;


extern __device__ int launch_signal;
extern __device__ int sync_signal_query;
extern __device__ int sync_signal_update;

extern __device__ int exp_hunger_dist0;
extern __device__ int exp_hunger_dist1;
extern __device__ int exp_hunger_update;
extern __device__ int exp_hunger_query;

extern __device__ int exp_cell_full;
extern __device__ int exp_cell_empty;

extern __device__ int exp_idx_bkt_dequeue_error;
extern __device__ int check_init_cells_a;
extern __device__ int check_init_cells_b;

extern __device__ int exp_new_cell_null;
extern __device__ int exp_old_cell_null;
extern __device__ int exp_update_in_spec_cell;

extern __device__ int exp_anchor_x_cell;
extern __device__ int exp_anchor_x_subgrid;

extern __device__ int chk_flag_cells_covered;
extern __device__ int chk_obj_covered;


extern __device__ int update_set_over;
extern __device__ int exp_hunger_dist_old;
extern __device__ int exp_hunger_update_old;

// Grid
extern __device__ int exp_idx_x_subgrid_a;
extern __device__ int exp_idx_x_subgrid_b;
extern __device__ CircularQueue *queue_bkts_free;

//extern __device__ int g_len_seg_cache_query;


// For debug
extern __device__ int check_tot_covered;
extern __device__ int check_area_per_query;
extern __device__ int cnt_singular;

extern __device__ int check_dist_query;

extern __device__ int* place_holder_update;
extern __device__ int* sync_holder_update;
extern __device__ int* place_holder_query;

extern __device__ int* place_holder_query_dispatch;
extern __device__ int* sync_holder_query_dispatch;
extern __device__ int* cache_memory_idx_query_dispatch;

extern __device__ int* place_holder_update_dispatch;


extern __device__ int cnt_over_seg_update;
extern __device__ int cnt_over_seg_query;
extern __device__ int cnt_over_bkt_update;


extern __device__ int cnt_queries;
extern __device__ int cnt_update_i;
extern __device__ int cnt_update_d;
extern __device__ int cnt_update_f;

extern __device__ long long int query_sumtime;
extern __device__ long long int update_sumtime;
extern __device__ long long int distribute_sumtime;
extern __device__ long long int copytime;



extern __device__ int cursor_distribute_wrap;
extern __device__ int cnt_distribute_wrap;


extern __device__ int query_sum_formempool;
extern __device__ int tot_obs_covered_forquery;

extern __device__ int offset_buffer_query_rec;
extern __device__ int offset_buffer_update_rec;

extern __device__ int query_time_per_period;
extern __device__ int update_time_per_period;
extern __device__ int dis_time_per_period;


extern __device__ int rebalance;
extern __device__ int buffer_exhausted;
extern __device__ int exit_query;
extern __device__ int exit_update;


extern __device__ int cnt_malloc_obj_pool;
extern __device__ int cnt_free_obj_pool;


extern __device__ int* multiqueue;
#else
extern "C" __device__ int gp_config_ready;

// buffer on GPU
//extern __device__ UpdateType *buffer_update;
//extern __device__ QueryType *buffer_query;

// update request queues & query request queues
//extern __device__ QueryQueue *query_req_queue;
extern "C" __device__ UpdateCacheArea *req_cache_update;
extern "C" __device__ int *update_map;

extern "C" __device__ QueryCacheArea *req_cache_query;
extern "C" __device__ int cnt_enqueue_update;
extern "C" __device__ int cnt_dequeue_update;
extern "C" __device__ int cnt_enqueue_query;
extern "C" __device__ int cnt_dequeue_query;
extern "C" __device__ UpdateQNode *node_dequeue_update;
extern "C" __device__ UpdateQNode *node_enqueue_update;
extern "C" __device__ QueryQNode *node_dequeue_query;
extern "C" __device__ QueryQNode *node_enqueue_query;

//extern __device__ UpdateTypeForSort* array_i_forsort;
//extern __device__ int array_forsort_cnt;
//extern __device__ int each_anchor_len;
//extern __device__ int* each_anchor_cnt;

//extern __device__ unsigned int progress_distribute;
//extern __device__ unsigned int progress_update;
//extern __device__ unsigned int progress_query;


// Global variables for GPGrid
extern "C" __device__ Grid *index_A;
extern "C" __device__ Grid *index_B;
extern "C" __device__ SecIndex *sec_index_A;
extern "C" __device__ SecIndex *sec_index_B;
extern "C" __device__ int flag_switch_version;
extern "C" __device__ int seg_switch_version;
extern "C" __device__ int query_switch_version;

extern "C" __device__ volatile int flag_switch_dist;
extern "C" __device__ volatile int flag_switch_update;
extern "C" __device__ volatile int flag_switch_query;
extern "C" __device__ ManagedMemory* DMM;


// sync signals
extern "C" __device__ unsigned int barrier_dist;
extern "C" __device__ unsigned int barrier_update;
extern "C" __device__ unsigned int barrier_query;

extern "C" __device__ int cnt_req_query;


extern "C" __device__ int launch_signal;
extern "C" __device__ int sync_signal_query;
extern "C" __device__ int sync_signal_update;

extern "C" __device__ int exp_hunger_dist0;
extern "C" __device__ int exp_hunger_dist1;
extern "C" __device__ int exp_hunger_update;
extern "C" __device__ int exp_hunger_query;

extern "C" __device__ int exp_cell_full;
extern "C" __device__ int exp_cell_empty;

extern "C" __device__ int exp_idx_bkt_dequeue_error;
extern "C" __device__ int check_init_cells_a;
extern "C" __device__ int check_init_cells_b;

extern "C" __device__ int exp_new_cell_null;
extern "C" __device__ int exp_old_cell_null;
extern "C" __device__ int exp_update_in_spec_cell;

extern "C" __device__ int exp_anchor_x_cell;
extern "C" __device__ int exp_anchor_x_subgrid;

extern "C" __device__ int chk_flag_cells_covered;
extern "C" __device__ int chk_obj_covered;


extern "C" __device__ int update_set_over;
extern "C" __device__ int exp_hunger_dist_old;
extern "C" __device__ int exp_hunger_update_old;

// Grid
extern "C" __device__ int exp_idx_x_subgrid_a;
extern "C" __device__ int exp_idx_x_subgrid_b;
extern "C" __device__ CircularQueue *queue_bkts_free;

//extern __device__ int g_len_seg_cache_query;


// For debug
extern "C" __device__ int check_tot_covered;
extern "C" __device__ int check_area_per_query;
extern "C" __device__ int cnt_singular;

extern "C" __device__ int check_dist_query;

extern "C" __device__ int* place_holder_update;
extern "C" __device__ int* sync_holder_update;
extern "C" __device__ int* place_holder_query;

extern "C" __device__ int* place_holder_query_dispatch;
extern "C" __device__ int* sync_holder_query_dispatch;
extern "C" __device__ int* cache_memory_idx_query_dispatch;

extern "C" __device__ int* place_holder_update_dispatch;


extern "C" __device__ int cnt_over_seg_update;
extern "C" __device__ int cnt_over_seg_query;
extern "C" __device__ int cnt_over_bkt_update;


extern "C" __device__ int cnt_queries;
extern "C" __device__ int cnt_update_i;
extern "C" __device__ int cnt_update_d;
extern "C" __device__ int cnt_update_f;

extern "C" __device__ long long int query_sumtime;
extern "C" __device__ long long int update_sumtime;
extern "C" __device__ long long int distribute_sumtime;
extern "C" __device__ long long int copytime;



extern "C" __device__ int cursor_distribute_wrap;
extern "C" __device__ int cnt_distribute_wrap;


extern "C" __device__ int query_sum_formempool;
extern "C" __device__ int tot_obs_covered_forquery;

extern "C" __device__ int offset_buffer_query_rec;
extern "C" __device__ int offset_buffer_update_rec;

extern "C" __device__ int query_time_per_period;
extern "C" __device__ int update_time_per_period;
extern "C" __device__ int dis_time_per_period;


extern "C" __device__ int rebalance;
extern "C" __device__ int buffer_exhausted;
extern "C" __device__ int exit_query;
extern "C" __device__ int exit_update;


extern "C" __device__ int cnt_malloc_obj_pool;
extern "C" __device__ int cnt_free_obj_pool;


extern "C" __device__ int* multiqueue;
#endif
#endif /* DEVICEGLOBALVAR_H_ */
