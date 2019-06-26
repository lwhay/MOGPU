/**********************************************************************
* QueryKernel.cu
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Oct 28, 2014 | 02:19:19 PM
* Description:*  
* Licence:*
**********************************************************************/
#include <math.h>
#include "QueryKernel.cuh"
#include "device/DeviceGlobalVar.cuh"
#include "misc/SyncFuncGPU.cuh"


//__global__ void QueryKernel(SimObject *d_obs_output, int *d_cnt_obs_per_req, QueryType *dev_buffer_query)
__global__ void QueryKernel(SimObject *d_obs_output, int *d_cnt_obs_per_req) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid_in_blk = threadIdx.x;
//	const int bid = blockIdx.x;
    const unsigned int barrier_step = gridDim.x;
    unsigned int barrier_fence = 0;
    //__shared__ QueryType shared_arr_query[1489];
    unsigned long obs_per_period_query = 0;
    unsigned long cells_per_period_query = 0;
    long long int start = -1, end;
    //long long int sumtime = 0;
    int local_query_switch_version = 0;

    const int WRAPSIZE = 32;
    int realWrapSize = 0;
    const int wid = tid / WRAPSIZE;
    const int tid_w = tid % WRAPSIZE;

    int *cnt_obs_per_req_local = d_cnt_obs_per_req;
    if (tid == 0) {
        node_dequeue_query = NULL;
        //cnt_dequeue_query = 0;
    }

    __syncthreads();
    barrier_fence += barrier_step;
    if (tid_in_blk == 0) {
        atomicAdd(&barrier_query, 1);
        while (barrier_query < barrier_fence && gp_config_ready == 0) { ;
        }
    }
    __syncthreads();

    while (launch_signal == 0);

    int len_bkt = gp_config->max_bucket_len;
//	int total_query = gp_config->max_query_num;
//	int buffer_block_size_update = gp_config->buffer_block_size;
//	int buffer_block_size_query = (int)((double)buffer_block_size_update * (double)total_query / (double)(total_update));

    int *place_holder_query_local = &place_holder_query[wid];

    int buffer_block_size_query = 0;
    const int LEN_SEG_CACHE_QUERY = gp_config->len_seg_cache_query;
//	int len_seg_cache_query = 1000;

    Grid * p_grid_local = NULL;
    volatile Cell *volatile p_cell_local = NULL;
    volatile ObjBox *volatile p_bucket_local = NULL;
    int bucketlen = 0;
    volatile Cell *volatile arr_cell = NULL;
    QueryCacheArea * req_cache_query_local = req_cache_query;

    SimObject *obj_dest = NULL;

    int anchor = tid;
    int edge_cell_num = gp_config->edge_cell_num;
    int tot_cells_query = edge_cell_num * edge_cell_num;

    int tot_cells_covered = 0;
    int *flag_cells_covered = NULL;

    volatile int *volatile idx_cells_covered = NULL;
    //int tot_obs_covered = 0;

    int *cnt_queries_per_cell = NULL;
    int *queries_per_cell = NULL;

    int *bound_btw_cell = NULL;

    QueryType *buffer_block_query = NULL;
    QueryQNode * array_local_0 = &req_cache_query_local->array[0];
    QueryQNode * array_local_1 = &req_cache_query_local->array[1];
    QueryQNode * node_dequeue_query_local = NULL;


    int idx_bkt_in_cell = -1;
    int idx_bkt_in_pool = -1;
    int idx_obj = -1;
    ObjBox * p_obj = NULL;


    int tot_queries_per_obs = 0;
    int qid = -1;
    int cid = -1;
    int oid = -1;
    float x = -1;
    float y = -1;
    int idx_tmp = -1, idx_tmp2 = -1;

    __syncthreads();
    barrier_fence += barrier_step;
    if (tid_in_blk == 0) {
        atomicAdd(&barrier_query, 1);
        while (barrier_query < barrier_fence);
    }
    __syncthreads();


    while (true) {

        if (tid == 0) {
            while ((req_cache_query->token0 == 0 || cnt_dequeue_query >= cnt_dequeue_update) \
 && (req_cache_query->token1 == 0 || cnt_dequeue_query >= cnt_dequeue_update) \
 && (buffer_exhausted == 0 || req_cache_query->token0 == 1 || req_cache_query->token0 == 1)) {
                exp_hunger_query++;
            }
            if (rebalance == 1 ||
                (buffer_exhausted == 1 && req_cache_query->token0 == 0 && req_cache_query->token1 == 0)) {
                atomicExch(&exit_query, 1);
            }
        }

        __syncthreads();
        barrier_fence += barrier_step;
        if (tid_in_blk == 0) {
            atomicAdd(&barrier_query, 1);
            while (barrier_query < barrier_fence);
        }
        __syncthreads();

        if (exit_query == 1)
            break;

        if (tid == 0) {
            if (req_cache_query->token0 == 1 && (req_cache_query->cnt0 == cnt_dequeue_query)) {
                node_dequeue_query = &req_cache_query->array[0];
                flag_switch_query = 0;
            } else if (req_cache_query->token1 == 1 && (req_cache_query->cnt1 == cnt_dequeue_query)) {
                node_dequeue_query = &req_cache_query->array[1];
                flag_switch_query = 1;
            } else {
                printf("\nerror\n");
            }
        }

        if (start == -1) {
            start = clock64();
        }

        __threadfence_system();
        __syncthreads();
        barrier_fence += barrier_step;
        if (tid_in_blk == 0) {
            atomicAdd(&barrier_query, 1);
            while (barrier_query < barrier_fence);
        }
        __syncthreads();

        if (flag_switch_query == 0) {
            p_grid_local = index_A;
        } else if (flag_switch_query == 1) {
            p_grid_local = index_B;
        }

        __syncthreads();
        barrier_fence += barrier_step;
        if (tid_in_blk == 0) {
            atomicAdd(&barrier_query, 1);
            while (barrier_query < barrier_fence);
        }
        __syncthreads();

        arr_cell = p_grid_local->arr_cell;

        node_dequeue_query_local = node_dequeue_query;

        tot_cells_covered = node_dequeue_query->tot_cells_covered;
        flag_cells_covered = node_dequeue_query->flag_cells_covered;
        cnt_queries_per_cell = node_dequeue_query->cnt_queries_per_cell;
        queries_per_cell = node_dequeue_query->queries_per_cell;
        idx_cells_covered = node_dequeue_query->idx_cells_covered;

        bound_btw_cell = node_dequeue_query->bound_btw_cell;


        buffer_block_query = node_dequeue_query->buffer_block_query;
        buffer_block_size_query = node_dequeue_query->buffer_block_size_query;
        cnt_obs_per_req_local = d_cnt_obs_per_req + node_dequeue_query->offset_buffer_query;

        //tot_cells_covered = 0;
        //*****************************************************
        /*	int forlimit = 10;
            int forresult = 0;
            for(int i = 0; i<forlimit; i++){
                forresult = (forresult+tid)*tid/((double)tid+12);
            }
        */
        /*		anchor = tid;
              while(anchor < gp_config->edge_cell_num*gp_config->edge_cell_num)
               {
                   while(p_grid_local->arr_cell[anchor].memfencedelay != 1);
                   p_grid_local->arr_cell[anchor].memfencedelay = 0;
                   anchor += gridDim.x*blockDim.x;
               }
           */    /*
		anchor = tid;
		while(anchor < gp_config->max_obj_num)
		{
			while(sec_index_B->index[anchor].memfence_delay != 1);
			sec_index_B->index[anchor].memfence_delay = 0;
			anchor += ((gridDim.x*blockDim.x)>>memfence_delay_step);
		}
		*/

        __threadfence_system();
        __syncthreads();
        barrier_fence += barrier_step;
        if (tid_in_blk == 0) {
            atomicAdd(&barrier_query, 1);
            while (barrier_query < barrier_fence);
        }
        __syncthreads();

#if CHECK_OBJ_BEFOREQUERY == 1
        if (tid == 0)
        {
            printf("\nCell obj num check:\n");
            for(int i = 10000000-1; i>=0; i-=100000)
            {
                printf("%d|%d ", i, sec_index_B->index[i].memfence_delay);
            }
            printf("\n\n");
        }
#endif

        if (tid == 0)
            atomicExch(&p_grid_local->cursor_query_wrap, 0);

        __syncthreads();
        barrier_fence += barrier_step;
        if (tid_in_blk == 0) {
            atomicAdd(&barrier_query, 1);
            while (barrier_query < barrier_fence);
        }
        __syncthreads();

#if QUERY_PATTERN == 3 //forcell-querybased-forobj-nobalance;
        anchor = tid;
        QueryType req_query;
        Cell *p_cell_lb,*p_cell_rt;
        int left_bottom,right_top,cell_num,row_start,col_end,row_end,col_start;
        while(anchor < node_dequeue_query->mtx_query_nodes->globalCnt)
        {
            req_query = buffer_block_query[anchor];

            float xmin = req_query.minX;
            float ymin = req_query.minY;
            float xmax = req_query.maxX;
            float ymax = req_query.maxY;

            p_cell_lb = p_grid_local->getNearestCellByXY(xmin,ymin);
            p_cell_rt = p_grid_local->getNearestCellByXY(xmax,ymax);

            left_bottom = p_cell_lb->idx;
            right_top = p_cell_rt->idx;
            cell_num = p_grid_local->cell_num;
            row_start = right_top / cell_num;
            col_end = right_top % cell_num;
            row_end = left_bottom / cell_num;
            col_start = left_bottom % cell_num;

            int cellNum = (row_end-row_start+1)*(col_end-col_start+1);
            for(int k = 0; k < cellNum; k++)
            {
                int i = row_start + k/(col_end-col_start+1);
                int j = col_start + k%(col_end-col_start+1);
                int idx_cell = i * gp_config->edge_cell_num + j;
                p_cell_local = &arr_cell[idx_cell];
                for(int m = 0; m<p_cell_local->tot_obs; m++)
                {
                    idx_bkt_in_cell = m/len_bkt;
                    idx_obj = m%len_bkt;
                    p_obj = &(p_grid_local->getBkt(p_cell_local->idx, idx_bkt_in_cell)[idx_obj]);
                    oid = p_obj->oid % gp_config->max_obj_num;
                    x = p_obj->x;
                    y = p_obj->y;
                    obj_dest = &d_obs_output[oid];
                    if (buffer_block_query[anchor].minX <= x && \
                            buffer_block_query[anchor].minY <= y && \
                            buffer_block_query[anchor].maxX >= x && \
                            buffer_block_query[anchor].maxY >= y)
                    {
                        //obj_dest->oid = oid;
                        //obj_dest->x = x;
                        //obj_dest->y = y;
                        //obj_dest->time = -1;
                        atomicAdd(&cnt_obs_per_req_local[anchor], 1);
#if IGNORE_CNT==0
                        atomicAdd(&chk_obj_covered, 1);
#endif
                    }
                }
            }
            anchor += blockDim.x*gridDim.x;
        }

#elif QUERY_PATTERN == 30

        QueryType req_query;
        Cell *p_cell_lb,*p_cell_rt;
        int left_bottom,right_top,cell_num,row_start,col_end,row_end,col_start;
        while(true)
        {
            if(tid_w == 0)
            {
                atomicExch(place_holder_query_local, atomicAdd(&p_grid_local->cursor_query_wrap, 1));
            }
            if(*place_holder_query_local >= node_dequeue_query->mtx_query_nodes->globalCnt)
            {
                break;//exit;
            }

            req_query = buffer_block_query[*place_holder_query_local];

            float xmin = req_query.minX;
            float ymin = req_query.minY;
            float xmax = req_query.maxX;
            float ymax = req_query.maxY;

            p_cell_lb = p_grid_local->getNearestCellByXY(xmin,ymin);
            p_cell_rt = p_grid_local->getNearestCellByXY(xmax,ymax);

            left_bottom = p_cell_lb->idx;
            right_top = p_cell_rt->idx;
            cell_num = p_grid_local->cell_num;
            row_start = right_top / cell_num;
            col_end = right_top % cell_num;
            row_end = left_bottom / cell_num;
            col_start = left_bottom % cell_num;

            int cellNum = (row_end-row_start+1)*(col_end-col_start+1);
            for(int k = 0; k < cellNum; k++)
            {
                int i = row_start + k/(col_end-col_start+1);
                int j = col_start + k%(col_end-col_start+1);
                int idx_cell = i * gp_config->edge_cell_num + j;
                p_cell_local = &arr_cell[idx_cell];
                anchor = tid_w;
                while(anchor < p_cell_local->tot_obs)
                {
                    idx_bkt_in_cell = anchor/len_bkt;
                    idx_obj = anchor%len_bkt;
                    p_obj = &(p_grid_local->getBkt(p_cell_local->idx, idx_bkt_in_cell)[idx_obj]);
                    oid = p_obj->oid % gp_config->max_obj_num;
                    x = p_obj->x;
                    y = p_obj->y;
                    obj_dest = &d_obs_output[oid];
                    if (buffer_block_query[*place_holder_query_local].minX <= x && \
                            buffer_block_query[*place_holder_query_local].minY <= y && \
                            buffer_block_query[*place_holder_query_local].maxX >= x && \
                            buffer_block_query[*place_holder_query_local].maxY >= y)
                    {
                        //obj_dest->oid = oid;
                        //obj_dest->x = x;
                        //obj_dest->y = y;
                        //obj_dest->time = -1;
                        atomicAdd(&cnt_obs_per_req_local[*place_holder_query_local], 1);
#if IGNORE_CNT==0
                        atomicAdd(&chk_obj_covered, 1);
#endif
                    }
                    anchor += WRAPSIZE;
                }
            }
        }

#elif QUERY_PATTERN == 20 //DWPcellbased-forobj-forquery

        while(true)
        {
            if(tid_w == 0)
            {
                atomicExch(place_holder_query_local, atomicAdd(&p_grid_local->cursor_query_wrap, 1));
            }
            if(*place_holder_query_local >= tot_cells_covered)
            {
                break;//exit;
            }
            p_cell_local = &arr_cell[idx_cells_covered[*place_holder_query_local]];
#if IGNORE_CNT==0
            if(tid_w == 0)
                atomicAdd(&tot_obs_covered_forquery, p_cell_local->tot_obs);
#endif
            cid = p_cell_local->idx;
            //tot_queries_per_obs = cnt_queries_per_cell[cid];
            //if(tot_queries_per_obs > LEN_SEG_CACHE_QUERY){
            //	tot_queries_per_obs = LEN_SEG_CACHE_QUERY;
            //}
            int qn_cursor_idx = node_dequeue_query->mtx_query_idx[cid];
            MemElementCollection<int>* mec_query = node_dequeue_query->mtx_query_nodes;
            do{
                MemElement* me_cursor = &(mec_query->mes[qn_cursor_idx]);
                tot_queries_per_obs = mec_query->cnt[qn_cursor_idx];

                anchor = tid_w;
                while(anchor < p_cell_local->tot_obs)
                {
                    idx_bkt_in_cell = anchor/len_bkt;
                    idx_obj = anchor%len_bkt;
                    p_obj = &(p_grid_local->getBkt(cid, idx_bkt_in_cell)[idx_obj]);
                    oid = p_obj->oid;
                    x = p_obj->x;
                    y = p_obj->y;
                    for (int i = 0; i < tot_queries_per_obs; i++)
                    {
                        //idx_tmp = queries_per_cell[];
                        idx_tmp = mec_query->pool[LEN_SEG_CACHE_QUERY * cid + i];
                        qid = buffer_block_query[idx_tmp].qid;
                        obj_dest = &d_obs_output[oid];
                        if (buffer_block_query[idx_tmp].minX <= x && \
                                buffer_block_query[idx_tmp].minY <= y && \
                                buffer_block_query[idx_tmp].maxX >= x && \
                                buffer_block_query[idx_tmp].maxY >= y)
                        {
                            //obj_dest->oid = oid;
                            //obj_dest->x = x;
                            //obj_dest->y = y;
                            //obj_dest->time = -1;
                            atomicAdd(&cnt_obs_per_req_local[idx_tmp], 1);
#if IGNORE_CNT==0
                            atomicAdd(&chk_obj_covered, 1);
#endif
                        }
                    }
                    anchor += WRAPSIZE;
                }

                __threadfence_system();
                qn_cursor_idx = me_cursor->next;
            }while(qn_cursor_idx!= -1);

        }

#elif QUERY_PATTERN == 2000 //DWPbuketbased-forobj-forquery-two-dimension-dispatch

        anchor = tid;
        while (anchor < node_dequeue_query->tot_cells_covered) {
            p_cell_local = &arr_cell[idx_cells_covered[anchor]];
            //count the number of objs of each related cells;
            bound_btw_cell[anchor + 1] = p_cell_local->cnt_bkts;
            anchor += blockDim.x * gridDim.x;
        }
        p_cell_local = NULL;
        __threadfence_system();
        __syncthreads();
        barrier_fence += barrier_step;
        if (tid_in_blk == 0) {
            atomicAdd(&barrier_query, 1);
            while (barrier_query < barrier_fence);

        }
        __syncthreads();

        for (int i = 0; i < node_dequeue_query->tot_cells_covered; i++) {
            while (true) {
                if (bound_btw_cell[i + 1] - 1 < 0) {
                    break;//exit;
                }
                if (tid_w == 0) {
                    atomicExch(place_holder_query_local, atomicAdd(bound_btw_cell + i + 1, -1) - 1);
                }
                if (*place_holder_query_local < 0) {
                    break;//exit;
                }
                p_cell_local = &arr_cell[idx_cells_covered[i]];
                p_bucket_local = p_grid_local->getBkt(p_cell_local->idx, *place_holder_query_local);
                if (*place_holder_query_local == p_cell_local->cnt_bkts - 1)
                    bucketlen = p_cell_local->tot_obs_top;
                else
                    bucketlen = p_cell_local->len_bkt;
#if IGNORE_CNT == 0
                if (tid_w == 0)
                    atomicAdd(&tot_obs_covered_forquery, bucketlen);
#endif
                int qn_cursor_idx = node_dequeue_query->mtx_query_idx[p_cell_local->idx];
                MemElementCollection<int> *mec_query = node_dequeue_query->mtx_query_nodes;
                do {
                    MemElement *me_cursor = &(mec_query->mes[qn_cursor_idx]);
                    tot_queries_per_obs = mec_query->cnt[qn_cursor_idx];

                    anchor = 0;
                    while (tot_queries_per_obs > 0 && anchor * WRAPSIZE < bucketlen) {
                        if ((anchor + 1) * WRAPSIZE <= bucketlen)//|| tot_queries_per_obs < WRAPSIZE)
                        {
                            idx_bkt_in_cell = *place_holder_query_local;
                            idx_obj = anchor * WRAPSIZE + tid_w;
                            p_obj = &(p_grid_local->getBkt(p_cell_local->idx, idx_bkt_in_cell)[idx_obj]);
                            oid = p_obj->oid;
                            x = p_obj->x;
                            y = p_obj->y;
                            for (int k = 0; k < tot_queries_per_obs; k++) {
                                idx_tmp = mec_query->pool[cid * mec_query->LEN + k];
                                qid = buffer_block_query[idx_tmp].qid;
                                //obj_dest = &d_obs_output[oid];
                                if (buffer_block_query[idx_tmp].minX <= x && \
                                                buffer_block_query[idx_tmp].minY <= y && \
                                                buffer_block_query[idx_tmp].maxX >= x && \
                                                buffer_block_query[idx_tmp].maxY >= y) {
                                    atomicAdd(&cnt_obs_per_req_local[idx_tmp], 1);
#if IGNORE_CNT == 0
                                    atomicAdd(&chk_obj_covered, 1);
#endif
                                }
                            }
                        } else if (tot_queries_per_obs >= WRAPSIZE) {
                            int cache_anchor = anchor;
                            anchor = 0;
                            idx_bkt_in_cell = *place_holder_query_local;
                            while (anchor < tot_queries_per_obs) {
                                idx_tmp = mec_query->pool[cid * mec_query->LEN + anchor];
                                qid = buffer_block_query[idx_tmp].qid;
                                for (int k = 0; k < bucketlen - cache_anchor * WRAPSIZE; k++) {
                                    idx_obj = cache_anchor * WRAPSIZE + k;
                                    p_obj = &(p_grid_local->getBkt(p_cell_local->idx, idx_bkt_in_cell)[idx_obj]);
                                    oid = p_obj->oid;
                                    x = p_obj->x;
                                    y = p_obj->y;
                                    if (buffer_block_query[idx_tmp].minX <= x && \
                                                buffer_block_query[idx_tmp].minY <= y && \
                                                buffer_block_query[idx_tmp].maxX >= x && \
                                                buffer_block_query[idx_tmp].maxY >= y) {
                                        atomicAdd(&cnt_obs_per_req_local[idx_tmp], 1);
#if IGNORE_CNT == 0
                                        atomicAdd(&chk_obj_covered, 1);
#endif
                                    }
                                }
                                anchor += WRAPSIZE;
                            }
                            break;
                        } else {
                            int localLen = (int) ceil(tot_queries_per_obs * bucketlen / (double) WRAPSIZE);
                            idx_bkt_in_cell = *place_holder_query_local;
                            idx_obj = -1;
                            for (int k = 0; k < localLen; ++k) {
                                idx_tmp = tid_w * localLen + k;
                                if (idx_tmp > tot_queries_per_obs * bucketlen)
                                    continue;
                                if (idx_obj != idx_tmp / tot_queries_per_obs) {
                                    idx_obj = idx_tmp / tot_queries_per_obs;
                                    p_obj = &(p_grid_local->getBkt(p_cell_local->idx, idx_bkt_in_cell)[idx_obj]);
                                    oid = p_obj->oid;
                                    x = p_obj->x;
                                    y = p_obj->y;
                                }
                                idx_tmp = mec_query->pool[LEN_SEG_CACHE_QUERY * cid + idx_tmp % tot_queries_per_obs];
                                qid = buffer_block_query[idx_tmp].qid;
                                //obj_dest = &d_obs_output[oid];
                                if (buffer_block_query[idx_tmp].minX <= x && \
                                                buffer_block_query[idx_tmp].minY <= y && \
                                                buffer_block_query[idx_tmp].maxX >= x && \
                                                buffer_block_query[idx_tmp].maxY >= y) {
                                    atomicAdd(&cnt_obs_per_req_local[idx_tmp], 1);
#if IGNORE_CNT == 0
                                    atomicAdd(&chk_obj_covered, 1);
#endif
                                }
                            }
                        }
                        anchor += 1;
                    }

                    __threadfence_system();
                    qn_cursor_idx = me_cursor->next;
                } while (qn_cursor_idx != -1);
            }
        }
#endif

        __syncthreads();
        barrier_fence += barrier_step;
        if (tid_in_blk == 0) {
            atomicAdd(&barrier_query, 1);
            while (barrier_query < barrier_fence);
        }
        __syncthreads();


        anchor = tid;
        while (anchor < tot_cells_query) {
            if (flag_cells_covered[anchor] != 0) {
                atomicAdd(&chk_flag_cells_covered, 1);
            }
            flag_cells_covered[anchor] = 0;
            anchor += blockDim.x * gridDim.x;
        }

        __threadfence_system();
        __syncthreads();
        barrier_fence += barrier_step;
        if (tid_in_blk == 0) {
            atomicAdd(&barrier_query, 1);
            while (barrier_query < barrier_fence);
        }
        __syncthreads();

        end = clock64();
        if (tid == 0) {
            atomicExch(&query_time_per_period, (int) (end - start));
            query_sumtime += (end - start);
            printf("\nq%d %.4f ms\n", cnt_dequeue_query, ((double) (end - start)) / gp_config->clockRate);
            start = -1;
            atomicExch(&node_dequeue_query->tot_cells_covered, 0);

            if (flag_switch_query == 0) {
                atomicExch(&req_cache_query->token0, 0);
            } else if (flag_switch_query == 1) {
                atomicExch(&req_cache_query->token1, 0);
            }
            node_dequeue_query = NULL;
            atomicAdd(&cnt_dequeue_query, 1);
        }

        __syncthreads();
        barrier_fence += barrier_step;
        if (tid_in_blk == 0) {
            atomicAdd(&barrier_query, 1);
            while (barrier_query < barrier_fence);
        }
        __syncthreads();


    }

    __syncthreads();
    barrier_fence += barrier_step;
    if (tid_in_blk == 0) {
        atomicAdd(&barrier_query, 1);
        while (barrier_query < barrier_fence)
            if (rebalance == 1)
                break;
    }
    __syncthreads();
    __threadfence_system();

    if (tid == 0) {
        if (rebalance == 0) {
            printf("\n");
            printf("\nQueryKernel clock: %f ms\n", ((double) query_sumtime) / gp_config->clockRate);

            printf("len_seg_cache_query in Query Kernel: %d\n", LEN_SEG_CACHE_QUERY);
            printf("cnt_dequeue_query: %d\n", cnt_dequeue_query);
            printf("exp_hunger_query: %d\n", exp_hunger_query);
            printf("cells_per_period_query: %lld %d\n", cells_per_period_query / cnt_dequeue_query, cnt_dequeue_query);
            printf("obs_per_period_query: %lld %d\n", obs_per_period_query / cnt_dequeue_query * 2, cnt_dequeue_query);
            printf("cnt_obs_per_req_local sample data: %d %d %d %d %d\n", cnt_obs_per_req_local[0], \
                cnt_obs_per_req_local[10], cnt_obs_per_req_local[20], \
                cnt_obs_per_req_local[30], cnt_obs_per_req_local[40] \
);

            printf("tot_obs_covered: %d\n", tot_obs_covered_forquery);

            printf("barrier_query: %u\n", barrier_query);
            printf("barrier_fence: %u\n", barrier_fence);
            printf("chk_flag_cells_covered: %u\n", chk_flag_cells_covered);
            printf("chk_obj_covered: %u\n", chk_obj_covered);
            printf("chk_obj_covered per ms: %.4f\n",
                   (double) chk_obj_covered / (((double) query_sumtime) / gp_config->clockRate));

        }
    }


    __syncthreads();
    barrier_fence += barrier_step;
    if (tid_in_blk == 0) {
        atomicAdd(&barrier_query, 1);
        while (barrier_query < barrier_fence);
    }
    __syncthreads();

    if (tid == 0) {
        atomicExch(&barrier_query, 0);
        atomicExch(&launch_signal, 0);
        atomicExch(&query_time_per_period, 0);
    }
}
