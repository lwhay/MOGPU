/**********************************************************************
* DistributorKernel.h
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Oct 22, 2014 | 10:35:25 AM
* Description:*  
* Licence:*
**********************************************************************/


#ifndef DISTRIBUTORKERNEL_H_
#define DISTRIBUTORKERNEL_H_

#include "misc/UpdateCacheArea.cuh"
#include "misc/QueryCacheArea.h"
//#include "BusyKernel.h"
#include "misc/BaseStruct.h"
#include "misc/SecIndex.cuh"
#include "misc/Grid.cuh"


__global__ void
DpProcess(int *place_holder_query_dispatch_local, int anchor_idx, int row_start, int row_end, int col_start,
          int col_end, GConfig *p_config, \
    QueryQNode *node_enqueue_query_local);

__global__ void DistributorKernel(GConfig * dev_p_gconfig, \
        UpdateType * dev_buffer_update, int * dev_cnt_update, \
        QueryType * dev_buffer_query, int * dev_cnt_query, \
        UpdateCacheArea * d_req_cache_update, QueryCacheArea * d_req_cache_query, \
        Grid * d_index_A, Grid * d_index_B, CircularQueue * d_queue_bkts_free, \
        MemItem<QueryType> * d_qd_obj_pool, CircularQueue * d_queue_idx_anchor_free, QueryType * d_qd_query_type_pool,
                                  int * d_qd_anchor_pool, \
        int * d_place_holder, ManagedMemory * d_mm);


#endif /* DISTRIBUTORKERNEL_H_ */
