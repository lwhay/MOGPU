/**********************************************************************
* GPGrid.cu
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Oct 23, 2014 | 10:13:10 PM
* Description:*  
* Licence:*
**********************************************************************/

#include <iostream>
#include <string.h>

#ifdef linux

#include <pthread.h>

#else

#include <windows.h>
#include <process.h>

#endif

#include "config/Config.h"
#include "config/GConfig.h"
#include "misc/Buffer.h"
#include "GPGrid.h"
#include "Grid.cuh"
#include "SecIndex.cuh"

#include "kernel/DistributorKernel.cuh"
#include "kernel/UpdateKernel.cuh"
#include "kernel/QueryKernel.cuh"
#include "device/DeviceGlobalVar.cuh"
#include "kernel/InitKernel.cuh"

#ifndef USEMEMPOOL
#define USEMEMPOOL 1;
#endif

using namespace std;
struct Message {
    cudaStream_t *stream_c;
    cudaEvent_t *start3, *stop3;
    float elapsed0;
    int time;

    int *d_map;
    int *h_map;
    int mapsize;
    UpdateType *d_ptr;
    UpdateType *h_ptr;

    long long int bsize;
    long long int cursor;
    long long int limit;
};

#ifdef linux

void *copyData(void *m) {
#else

    DWORD WINAPI copyData(void *m) {
#endif
    Message *message = (Message *) m;
    int *d_map = message->d_map;
    int *h_map = message->h_map;
    int mapsize = message->mapsize;
    UpdateType *d_ptr = message->d_ptr;
    UpdateType *h_ptr = message->h_ptr;
    long long int bsize = message->bsize;
    long long int limit = message->limit;
    cudaStream_t *stream_c = message->stream_c;
    cudaEvent_t *start3 = message->start3;
    cudaEvent_t *stop3 = message->stop3;
    float elapsed;
    while (message->cursor < limit) {
        if (message->cursor + bsize > limit) {
            bsize = limit - (message->cursor);
        }
        int idx = ((int) ((message->cursor) / (message->bsize))) % mapsize;
        do {
            cudaMemcpyAsync(h_map, d_map, sizeof(int) * mapsize, cudaMemcpyDeviceToHost, *stream_c);
            cudaStreamSynchronize(*stream_c);
        } while (h_map[idx] != 0);
        cout << "\nload buffer idx:  " << idx << ", empty: " << (h_map[idx] == 0) << endl;
        cudaEventRecord(*start3, *stream_c);
        cudaMemcpyAsync(d_ptr + ((message->cursor) % (mapsize * message->bsize)), h_ptr + (message->cursor),
                        sizeof(UpdateType) * bsize, cudaMemcpyHostToDevice, *stream_c);
        cudaEventRecord(*stop3, *stream_c);
        cudaStreamSynchronize(*stream_c);
        cudaEventElapsedTime(&elapsed, *start3, *stop3);
        h_map[idx] = 1;
        cudaMemcpyAsync(d_map + idx, h_map + idx, sizeof(int), cudaMemcpyHostToDevice, *stream_c);
        cudaStreamSynchronize(*stream_c);
        message->cursor += bsize;
        message->time++;
        message->elapsed0 += elapsed;
    }
    return 0;
}

void launchGPGrid(void) {
    // Reset all status on the GPU
#ifdef linux
    pthread_t t1;
#else
    HANDLE handle[HOST_COPY_DATA_THREAD_NUM];
#endif
    cudaDeviceReset();
    cudaError_t cudaStatus;
    Config *p_config = Config::getInstance();
    long long unsigned sizecnt = 0;
    initHostGConfig();
    cout << "Finished initHostGConfig()" << endl;

    if (p_config->edge_cell_num < p_config->side_len_vgroup) {
        cout << "\nError parameters: edge_cell_num < side_len_vgroup !!! \n" << endl;
        exit(1);
    }

    // Initialize Buffer
    Buffer<UpdateType> *orig_buffer_update = new Buffer<UpdateType>();
    Buffer<QueryType> *orig_buffer_query = new Buffer<QueryType>();
    orig_buffer_update->initBuffer();
    orig_buffer_query->initBuffer();


    cudaEvent_t start0, stop0, start1, stop1, start2, stop2, start3, stop3, realstart;
    float elapsed0, elapsed1, elapsed2;
    float elapsed0_sum = 0, elapsed1_sum = 0, elapsed2_sum = 0;
    int res;

    cudaEventCreate(&start0);
    cudaEventCreate(&stop0);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventCreate(&realstart);

    cudaStream_t stream0;
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStream_t stream3;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    GConfig * pinned_p_config;
    UpdateType *pinned_buffer_update;
    QueryType *pinned_buffer_query;
    UpdateType *dev_buffer_update;
    QueryType *dev_buffer_query;

    // ------------------------------------------------------------------------------------
    cout << "Sizeof UpdateType: " << sizeof(UpdateType) << endl;
    cout << "Sizeof QueryType: " << sizeof(QueryType) << endl;
    cout << "Sizeof ObjBox: " << sizeof(ObjBox) << endl;
    cout << "Sizeof ObjBox **: " << sizeof(ObjBox **) << endl;
    cout << "Sizeof QueryQNode: " << sizeof(QueryQNode) << endl;
    cout << "Sizeof QueryType *: " << sizeof(QueryType *) << endl;
    cout << "Sizeof int *: " << sizeof(int *) << endl;
    cout << "Sizeof SIEntry: " << sizeof(SIEntry) << endl << endl;

    cout << "Initializing memory for Buffer...  " << endl;


    if (p_config->gaussian_data == 1) {
        cout << "Gaussian Data!!!" << endl;
    } else if (p_config->uniform_data == 1) {
        cout << "Uniform Data!!!" << endl;
    }


    ManagedMemory * d_mm;
    cudaMalloc((void **) &d_mm, sizeof(ManagedMemory) * 1);
    cudaMemset(d_mm, 0, sizeof(ManagedMemory) * 1);
    ManagedMemory * h_mm = new ManagedMemory();
    h_mm->mmsLen = 5;

    //Inspective Variables
    int *complete_cnt_update, *dev_cnt_update;
    int *complete_cnt_query, *dev_cnt_query;
    cudaHostAlloc((void **) &complete_cnt_update, sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void **) &complete_cnt_query, sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void **) &pinned_p_config, sizeof(GConfig), cudaHostAllocMapped);
    //cudaMallocManaged(&pinned_p_config,sizeof(GConfig));
    cudaHostAlloc((void **) &pinned_buffer_update, orig_buffer_update->buffer_len * sizeof(UpdateType),
                  cudaHostAllocMapped);
    cudaHostAlloc((void **) &pinned_buffer_query, orig_buffer_query->buffer_len * sizeof(QueryType),
                  cudaHostAllocMapped);

    cout << "Initializing buffer on host memeory..." << endl;
    int tot_cells = p_config->edge_cell_num * p_config->edge_cell_num;

    memcpy(pinned_p_config, host_p_config, sizeof(GConfig));
    memcpy(pinned_buffer_update, orig_buffer_update->req_buffer, orig_buffer_update->buffer_len * sizeof(UpdateType));
    memcpy(pinned_buffer_query, orig_buffer_query->req_buffer, orig_buffer_query->buffer_len * sizeof(QueryType));

    cout << endl << "Blocks & Threads for Distributor Kernel: " << pinned_p_config->block_analysis_num << "\t" \
 << pinned_p_config->thread_analysis_num << endl;
    cout << "Blocks & Threads for Update Kernel: " << pinned_p_config->block_update_num << "\t" << \
            pinned_p_config->thread_update_num << endl;
    cout << "Blocks & Threads for Query Kernel: " << pinned_p_config->block_query_num << "\t" << \
            pinned_p_config->thread_query_num << endl << endl;

    cout << "Check initializaion of update buffer: " << (pinned_buffer_update)->oid << (pinned_buffer_update + 1)->x
         << "\t" << (pinned_buffer_update + 1)->y << endl;
    cout << "Check initializaion of query buffer: " << (pinned_buffer_query + 1000)->minX << "\t"
         << (pinned_buffer_query + 1000)->minY << endl;


    cout << "Calling cudaMalloc() for buffer..." << endl;
    cudaMalloc((void **) &dev_p_config, sizeof(GConfig));
    //cudaMalloc((void **)&dev_buffer_update, sizeof(UpdateType) * orig_buffer_update->buffer_len);
    cudaMalloc((void **) &dev_buffer_update,
               sizeof(UpdateType) * p_config->buffer_update_round * p_config->buffer_block_size);
    cudaMalloc((void **) &dev_buffer_query, sizeof(QueryType) * orig_buffer_query->buffer_len);
    cudaMalloc((void **) &dev_cnt_update, sizeof(int));
    cudaMalloc((void **) &dev_cnt_query, sizeof(int));

    cout << endl << "Loading config data...  " << endl;
    cudaMemcpyAsync(dev_p_config, pinned_p_config, sizeof(GConfig), \
            cudaMemcpyHostToDevice, stream0);
    //cudaMemcpy(dev_buffer_update, pinned_buffer_update, sizeof(UpdateType) * orig_buffer_update->buffer_len, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_buffer_query, pinned_buffer_query, sizeof(QueryType) * orig_buffer_query->buffer_len,
               cudaMemcpyHostToDevice);
    //cudaHostGetDevicePointer((void **)&dev_buffer_update, (void *)pinned_buffer_update, 0);
    //cudaHostGetDevicePointer((void **)&dev_buffer_query, (void *)pinned_buffer_query, 0);
    cout << "Memory size for orig_buffer_query&update(MiB): " <<
         (sizeof(UpdateType) * p_config->buffer_update_round * p_config->buffer_block_size +
          sizeof(QueryType) * orig_buffer_query->buffer_len) / (1024.0 * 1024) << endl << endl;

    //cudaFreeHost(pinned_buffer_update);
    //cudaFreeHost(pinned_buffer_query);

    // ------------------------------------------------------------------------------------
    cout << endl << "Initializing memory for Objects..." << endl;
    ObjBox * d_obs_pool;
    ObjBox * d_obs_pool_A;
    ObjBox * d_obs_pool_B;
    // bucket_len must be devided by 512 without remainder
    //size_t d_cols_bkt = p_config->max_bucket_len;
    //size_t d_avail_bkt = (p_config->max_obj_num * 2 / d_cols_bkt) + 1;
    //size_t d_rows_bkt =  d_avail_bkt + p_config->edge_cell_num * p_config->edge_cell_num * 2 + 32;
    //size_t obs_num = d_cols_bkt *  d_rows_bkt;
    size_t d_cols_bkt = p_config->max_bucket_len;
    size_t d_avail_bkt = (p_config->max_obj_num / d_cols_bkt) + 1;
    size_t d_rows_bkt = d_avail_bkt + p_config->edge_cell_num * p_config->edge_cell_num + 16;
    size_t obs_num = d_cols_bkt * d_rows_bkt + 1;


    res = cudaMalloc((void **) &d_obs_pool_A, (obs_num * sizeof(ObjBox)));
    if (res != 0) {
        cout << "Objects::Error Point 0A!" << endl;
    }
    res = cudaMalloc((void **) &d_obs_pool_B, (obs_num * sizeof(ObjBox)));
    if (res != 0) {
        cout << "Objects::Error Point 0B!" << endl;
    }

//	ObjBox *h_obs_pool;
//	cudaMallocHost(&h_obs_pool, sizeof(ObjBox) * obs_num);
//	cout << "Check object_pool: " << h_obs_pool[13].x << "\t" << h_obs_pool[14].y << endl;
//	cudaMemcpyAsync(d_obs_pool, h_obs_pool, obs_num * sizeof(ObjBox), \
//				cudaMemcpyHostToDevice, stream0);

    cout << "d_avail_bkt: " << d_avail_bkt << endl;
    cout << "d_rows_bkt: " << d_rows_bkt << endl;
    cout << "Memory size for Objects(MiB): " << (obs_num * 2 * sizeof(ObjBox)) / (1024.0 * 1024) << endl << endl;

    h_mm->mms[0].ptr = d_obs_pool_A;
    h_mm->mms[0].toPtr = d_obs_pool_B;
    h_mm->mms[0].checkpoint = (int *) &(d_obs_pool_A + obs_num - 1)->time;
    h_mm->mms[0].toCheckpoint = (int *) &(d_obs_pool_B + obs_num - 1)->time;
    h_mm->mms[0].len = obs_num;
    h_mm->mms[0].bsize = obs_num * sizeof(ObjBox);

    //------------------------------------------------------------------------------------

    cout << "Initializing memory for Second Index..." << endl;
    size_t d_cols_sie = (p_config->max_obj_num + 1);
    size_t d_rows_sie = 2;
    SIEntry * h_sie_array;
    cudaMallocHost(&h_sie_array, sizeof(SIEntry) * d_cols_sie * d_rows_sie);
    for (int i = 0; i < d_cols_sie * d_rows_sie; ++i) {
        h_sie_array[i].idx_cell = -1;
    }
    SIEntry * d_sie_array;
    res = cudaMalloc((void **) &d_sie_array, d_cols_sie * d_rows_sie * sizeof(SIEntry));

    cout << "Check second index: " << h_sie_array[100].idx_cell << "\t" \
 << h_sie_array[100].idx_bkt << "\t" << h_sie_array[113].idx_obj << endl;

    cout << "Result of cudaMalloc for d_sie_array:\t" << res << "\t" << d_sie_array << endl;
    cudaMemcpyAsync(d_sie_array, h_sie_array, d_cols_sie * d_rows_sie * sizeof(SIEntry), \
                    cudaMemcpyHostToDevice, stream0);
    cout << "Memory size for SIE(MiB): " << (d_cols_sie * d_rows_sie * sizeof(SIEntry)) / (1024.0 * 1024) << endl
         << endl;

    h_mm->mms[1].ptr = d_sie_array;
    h_mm->mms[1].toPtr = d_sie_array + d_cols_sie;
    h_mm->mms[1].checkpoint = &(d_sie_array + d_cols_sie - 1)->idx_obj;
    h_mm->mms[1].toCheckpoint = &(d_sie_array + 2 * d_cols_sie - 1)->idx_obj;
    h_mm->mms[1].len = d_cols_sie;
    h_mm->mms[1].bsize = sizeof(SIEntry) * d_cols_sie;



    // ------------------------------------------------------------------------------------
    sizecnt = 0;
    cout << endl << "Initializing memory for Update Cache..." << endl;
    int capacity_cache_update = 2;
    int tot_vgroup_update = p_config->side_len_vgroup * p_config->side_len_vgroup;
    int h_seg_node_update = p_config->len_seg_cache_update;
    //memnode number per version
#if SEG_CACHE == 0
    int len_node_update = p_config->buffer_block_size / h_seg_node_update + 1 + tot_vgroup_update + \
        26 * (p_config->thread_update_num / 32 + 1) * 5;
#else
    int len_node_update = tot_vgroup_update + 5;
#endif


    // Update cache area: Start--------
    int *d_delete_mtx_pool;
    res = cudaMalloc((void **) &d_delete_mtx_pool,
                     capacity_cache_update * len_node_update * h_seg_node_update * sizeof(int));
    if (res != 0) {
        cout << "Update Cache::Error Point 0!" << endl;
        return;
    }
    cudaMemsetAsync(d_delete_mtx_pool, 0, capacity_cache_update * len_node_update * h_seg_node_update * sizeof(int),
                    stream0);
    sizecnt += capacity_cache_update * len_node_update * h_seg_node_update * sizeof(int);

    MemElement *h_me_delete_pool = new MemElement[(capacity_cache_update + 1) * len_node_update];
    for (int i = 0; i < len_node_update; ++i) {
        h_me_delete_pool[2 * len_node_update + i].lock = 0;
        h_me_delete_pool[2 * len_node_update + i].next = -1;
    }
    MemElement *d_me_delete_pool;
    cudaMalloc((void **) &d_me_delete_pool, sizeof(MemElement) * (capacity_cache_update + 1) * len_node_update);
    sizecnt += sizeof(MemElement) * (capacity_cache_update + 1) * len_node_update;
    cudaMemcpyAsync(d_me_delete_pool, h_me_delete_pool,
                    sizeof(MemElement) * (capacity_cache_update + 1) * len_node_update, cudaMemcpyHostToDevice,
                    stream0);
    MemElementCollection<int> *h_mec_d = new MemElementCollection<int>[2];
    for (int i = 0; i < 2; ++i) {
        int *d_mec_cnts;
        cudaMalloc((void **) &d_mec_cnts, sizeof(int) * len_node_update);
        sizecnt += sizeof(int) * len_node_update;
        h_mec_d[i].cnt = d_mec_cnts;
        int *d_mec_lasts;
        cudaMalloc((void **) &d_mec_lasts, sizeof(int) * tot_cells);
        sizecnt += sizeof(int) * tot_cells;
        h_mec_d[i].last = d_mec_lasts;
        h_mec_d[i].LEN = h_seg_node_update;
        h_mec_d[i].mes = &d_me_delete_pool[i * len_node_update];
        h_mec_d[i].pool = &d_delete_mtx_pool[i * len_node_update * h_seg_node_update];
    }
    MemElementCollection<int> *d_mec_d;
    cudaMalloc((void **) &d_mec_d, sizeof(MemElementCollection<int>) * 2);
    sizecnt += sizeof(MemElementCollection<int>) * 2;
    cudaMemcpyAsync(d_mec_d, h_mec_d, sizeof(MemElementCollection<int>) * 2, cudaMemcpyHostToDevice, stream0);


    UpdateType *d_insert_mtx_pool;
    res = cudaMalloc((void **) &d_insert_mtx_pool,
                     capacity_cache_update * len_node_update * 2 * h_seg_node_update * sizeof(UpdateType));
    if (res != 0) {
        cout << "Update Cache::Error Point 1!" << endl;
        return;
    }
    cudaMemsetAsync(d_insert_mtx_pool, 0,
                    capacity_cache_update * len_node_update * 2 * h_seg_node_update * sizeof(UpdateType), stream0);
    sizecnt += capacity_cache_update * len_node_update * 2 * h_seg_node_update * sizeof(UpdateType);

    MemElement *h_me_insert_pool = new MemElement[(capacity_cache_update + 1) * len_node_update * 2];
    for (int i = 0; i < len_node_update * 2; ++i) {
        h_me_insert_pool[2 * len_node_update * 2 + i].lock = 0;
        h_me_insert_pool[2 * len_node_update * 2 + i].next = -1;
    }
    MemElement *d_me_insert_pool;
    cudaMalloc((void **) &d_me_insert_pool, sizeof(MemElement) * (capacity_cache_update + 1) * len_node_update * 2);
    sizecnt += sizeof(MemElement) * (capacity_cache_update + 1) * len_node_update * 2;
    cudaMemcpyAsync(d_me_insert_pool, h_me_insert_pool,
                    sizeof(MemElement) * (capacity_cache_update + 1) * len_node_update * 2, cudaMemcpyHostToDevice,
                    stream0);
    MemElementCollection<UpdateType> *h_mec_i = new MemElementCollection<UpdateType>[2];
    for (int i = 0; i < 2; ++i) {
        int *d_mec_cnts;
        cudaMalloc((void **) &d_mec_cnts, sizeof(int) * len_node_update * 2);
        sizecnt += sizeof(int) * len_node_update * 2;
        h_mec_i[i].cnt = d_mec_cnts;
        int *d_mec_lasts;
        cudaMalloc((void **) &d_mec_lasts, sizeof(int) * tot_cells * 2);
        sizecnt += sizeof(int) * tot_cells * 2;
        h_mec_i[i].last = d_mec_lasts;
        h_mec_i[i].LEN = h_seg_node_update;
        h_mec_i[i].mes = &d_me_insert_pool[i * len_node_update * 2];
        h_mec_i[i].pool = &d_insert_mtx_pool[i * len_node_update * 2 * h_seg_node_update];
    }
    MemElementCollection<UpdateType> *d_mec_i;
    cudaMalloc((void **) &d_mec_i, sizeof(MemElementCollection<UpdateType>) * 2);
    sizecnt += sizeof(MemElementCollection<UpdateType>) * 2;
    cudaMemcpyAsync(d_mec_i, h_mec_i, sizeof(MemElementCollection<UpdateType>) * 2, cudaMemcpyHostToDevice, stream0);


    // = = = = = = = = = = = = = =
    int *d_mtx_idx;
    res = cudaMalloc((void **) &d_mtx_idx, 3 * capacity_cache_update * tot_vgroup_update * sizeof(int));
    if (res != 0) {
        cout << "Update Cache::Error Point 3.1!" << endl;
        return;
    }
    cudaMemsetAsync(d_mtx_idx, 0, 3 * capacity_cache_update * tot_vgroup_update * sizeof(int), stream0);
    sizecnt += 3 * capacity_cache_update * tot_vgroup_update * sizeof(int);

    int *d_sum_pool;
    res = cudaMalloc((void **) &d_sum_pool, 3 * capacity_cache_update * tot_vgroup_update * sizeof(int));
    if (res != 0) {
        cout << "Update Cache::Error Point 3.2!" << endl;
        return;
    }
    cudaMemsetAsync(d_sum_pool, 0, 3 * capacity_cache_update * tot_vgroup_update * sizeof(int), stream0);
    sizecnt += 3 * capacity_cache_update * tot_vgroup_update * sizeof(int);

    //End------
    UpdateQNode * h_arr_node = (UpdateQNode *) malloc(capacity_cache_update * sizeof(UpdateQNode));
    UpdateQNode * d_arr_node;
    res = cudaMalloc((void **) &d_arr_node, capacity_cache_update * sizeof(UpdateQNode));
    if (res != 0) {
        cout << "Update Cache::Error Point 6!" << endl;
        return;
    }
    for (int i = 0; i < capacity_cache_update; i++) {
        h_arr_node[i].lock = 0;
        h_arr_node[i].seg = h_seg_node_update;
        h_arr_node[i].num_cells = tot_vgroup_update;

        h_arr_node[i].d_size = len_node_update;
        h_arr_node[i].i_size = len_node_update * 2;
        h_arr_node[i].f_size = len_node_update;

        h_arr_node[i].mtx_delete_idx = d_mtx_idx + (3 * i + 0) * tot_vgroup_update;
        h_arr_node[i].mtx_insert_idx = d_mtx_idx + (3 * i + 1) * tot_vgroup_update;
        h_arr_node[i].mtx_fresh_idx = d_mtx_idx + (3 * i + 2) * tot_vgroup_update;
        h_arr_node[i].mtx_delete_nodes = &d_mec_d[i];
        h_arr_node[i].mtx_insert_nodes = &d_mec_i[i];
        h_arr_node[i].mtx_delete_nodes_bak = &d_me_delete_pool[2 * len_node_update];
        h_arr_node[i].mtx_insert_nodes_bak = &d_me_insert_pool[2 * len_node_update * 2];

        int tempsize = 0;
        for (int k = 0; k < 2; k++) {
            if (k == 1) {
                tempsize = len_node_update * 2;
            } else if (k == 0) {
                tempsize = len_node_update;
            }
            int *h_avail_queue = new int[tempsize];//memleak
            for (int j = 0; j < tempsize; j++) {
                h_avail_queue[j] = j;
            }
            int *d_avail_queue;
            res = cudaMalloc((void **) &d_avail_queue, sizeof(int) * tempsize);
            if (res != 0) {
                cout << "Output::Error Point 8.0!" << endl;
            }
            cudaMemcpyAsync(d_avail_queue, h_avail_queue, sizeof(int) * tempsize, cudaMemcpyHostToDevice, stream0);
            sizecnt += sizeof(int) * tempsize;
            CircularQueue * h_queue_free;
            cudaMallocHost(&h_queue_free, sizeof(CircularQueue));
            h_queue_free->capacity = tempsize;
            h_queue_free->avail_idx_bkt = d_avail_queue;
            h_queue_free->cnt_elem = tempsize;
            h_queue_free->head = 0;
            h_queue_free->rear = 0;
            CircularQueue * d_queue_free;
            res = cudaMalloc((void **) &d_queue_free, sizeof(CircularQueue));
            if (res != 0) {
                cout << "Output::Error Point 8.1!" << endl;
            }
            cudaMemcpyAsync(d_queue_free, h_queue_free, sizeof(CircularQueue), cudaMemcpyHostToDevice, stream0);
            sizecnt += sizeof(CircularQueue);
            if (k == 0)
                h_arr_node[i].fqueue_delete = d_queue_free;
            else if (k == 1)
                h_arr_node[i].fqueue_insert = d_queue_free;
        }

        h_arr_node[i].sum_d = d_sum_pool + (3 * i + 0) * tot_vgroup_update;
        h_arr_node[i].sum_i = d_sum_pool + (3 * i + 1) * tot_vgroup_update;
        h_arr_node[i].sum_f = d_sum_pool + (3 * i + 2) * tot_vgroup_update;
    }
    cudaMemcpyAsync(d_arr_node, h_arr_node, capacity_cache_update * sizeof(UpdateQNode), cudaMemcpyHostToDevice,
                    stream0);

    UpdateCacheArea * h_req_cache_update;
    cudaMallocHost(&h_req_cache_update, sizeof(UpdateCacheArea));
    UpdateCacheArea * d_req_cache_update;
    cudaMalloc((void **) &d_req_cache_update, sizeof(UpdateCacheArea));
    sizecnt += sizeof(UpdateCacheArea);
    h_req_cache_update->token0 = 0;
    h_req_cache_update->token1 = 0;
    h_req_cache_update->cnt0 = 0;
    h_req_cache_update->cnt1 = 0;
    h_req_cache_update->array = d_arr_node;
    cudaMemcpyAsync(d_req_cache_update, h_req_cache_update, sizeof(UpdateCacheArea), cudaMemcpyHostToDevice, stream0);

    cout << "Memory size for cache area of updates(MiB): " << sizecnt / (1024 * 1024.0) << endl << endl;


    // ------------------------------------------------------------------------------------
    cout << endl << "Initializing memory for Query Cache..." << endl;
    long long unsigned capacity_cache_query = 2;
    sizecnt = 0;
    long long unsigned tot_cells_query = p_config->edge_cell_num * p_config->edge_cell_num;

    int *h_flag_cells_covered = new int[tot_cells_query * capacity_cache_query];
    memset(h_flag_cells_covered, 0, sizeof(int) * tot_cells_query * capacity_cache_query);
    int *d_flag_cells_covered;
    res = cudaMalloc((void **) &d_flag_cells_covered, sizeof(int) * tot_cells_query * capacity_cache_query);
    cudaMemcpyAsync(d_flag_cells_covered, h_flag_cells_covered, sizeof(int) * tot_cells_query * capacity_cache_query, \
            cudaMemcpyHostToDevice, stream0);
    sizecnt += sizeof(int) * tot_cells_query * capacity_cache_query;

    int *d_cnt_queries_per_cell;
    res = cudaMalloc((void **) &d_cnt_queries_per_cell, sizeof(int) * tot_cells_query * capacity_cache_query);
    sizecnt += sizeof(int) * tot_cells_query * capacity_cache_query;
    cudaMemsetAsync(d_cnt_queries_per_cell, 0, sizeof(int) * tot_cells_query * capacity_cache_query, stream0);
    if (res != 0) {
        cout << "Error Point 0!" << endl;
        return;
    }

    int total_update = p_config->max_obj_num * p_config->round_num;
    int total_query = p_config->max_query_num;
    int len_seg_cache_query = p_config->len_seg_cache_query;
    int left_update_afterskip = total_update - p_config->query_skip_round_num * p_config->max_obj_num;
    int tmp_i = 0;
    if (left_update_afterskip > p_config->buffer_block_size) {
        tmp_i = (int) ((double) p_config->buffer_block_size *
                       ((double) p_config->max_query_num / (double) left_update_afterskip)) + 1;
    } else {
        tmp_i = p_config->max_query_num;
    }
#if SEG_CACHE == 0
    int len_node_query =
            tmp_i / len_seg_cache_query + 1 + tot_cells_query + 26 * (p_config->thread_query_num / 32 + 1) * 5;
#else
    int len_node_query = tot_cells_query + 5;
#endif

    int *d_query_mtx_pool;
    res = cudaMalloc((void **) &d_query_mtx_pool,
                     capacity_cache_query * len_node_query * len_seg_cache_query * sizeof(int));
    if (res != 0) {
        cout << "Query Cache::Error Point 0!" << endl;
        return;
    }
    cudaMemsetAsync(d_query_mtx_pool, 0, capacity_cache_query * len_node_query * len_seg_cache_query * sizeof(int),
                    stream0);
    sizecnt += capacity_cache_query * len_node_query * len_seg_cache_query * sizeof(int);

    MemElement *h_me_query_pool = new MemElement[(capacity_cache_query + 1) * len_node_query];
    for (int i = 0; i < len_node_query; ++i) {
        h_me_query_pool[2 * len_node_query + i].lock = 0;
        h_me_query_pool[2 * len_node_query + i].next = -1;
        //h_me_query_pool[len_node_query+i].lock = 0;
        //h_me_query_pool[len_node_query+i].next = -1;
        //h_me_query_pool[i].lock = 0;
        //h_me_query_pool[i].next = -1;
    }
    MemElement *d_me_query_pool;
    cudaMalloc((void **) &d_me_query_pool, sizeof(MemElement) * (capacity_cache_query + 1) * len_node_query);
    sizecnt += sizeof(MemElement) * (capacity_cache_query + 1) * len_node_query;
    cudaMemcpyAsync(d_me_query_pool, h_me_query_pool, sizeof(MemElement) * (capacity_cache_query + 1) * len_node_query,
                    cudaMemcpyHostToDevice, stream0);
    MemElementCollection<int> *h_mec_q = new MemElementCollection<int>[2];
    for (int i = 0; i < 2; ++i) {
        int *q_mec_cnts;
        cudaMalloc((void **) &q_mec_cnts, sizeof(int) * len_node_query);
        sizecnt += sizeof(int) * len_node_query;
        h_mec_q[i].cnt = q_mec_cnts;
        int *q_mec_lasts;
        cudaMalloc((void **) &q_mec_lasts, sizeof(int) * tot_cells_query);
        sizecnt += sizeof(int) * tot_cells;
        h_mec_q[i].last = q_mec_lasts;
        h_mec_q[i].LEN = len_seg_cache_query;
        h_mec_q[i].mes = &d_me_query_pool[i * len_node_query];
        h_mec_q[i].pool = &d_query_mtx_pool[i * len_node_query * len_seg_cache_query];
    }
    MemElementCollection<int> *d_mec_q;
    cudaMalloc((void **) &d_mec_q, sizeof(MemElementCollection<int>) * 2);
    sizecnt += sizeof(MemElementCollection<int>) * 2;
    cudaMemcpyAsync(d_mec_q, h_mec_q, sizeof(MemElementCollection<int>) * 2, cudaMemcpyHostToDevice, stream0);
    int *d_query_idx;
    res = cudaMalloc((void **) &d_query_idx, capacity_cache_query * tot_cells_query * sizeof(int));
    if (res != 0) {
        cout << "Query Cache::Error Point 3.1!" << endl;
        return;
    }
    cudaMemsetAsync(d_query_idx, 0, capacity_cache_query * tot_cells_query * sizeof(int), stream0);
    sizecnt += capacity_cache_query * tot_cells_query * sizeof(int);


/*	int *d_queries_per_cell;
	res = cudaMalloc((void **)&d_queries_per_cell, sizeof(int) * tot_cells_query * len_seg_cache_query * capacity_cache_query);
	sizecnt += sizeof(int) * tot_cells_query * len_seg_cache_query * capacity_cache_query;
	cout << "len_seg_cache_query: " << len_seg_cache_query << endl;
	if (res != 0)
	{
		cout << "Error Point 1!" << endl;
		cout << "p_config->buffer_block_size: " << p_config->buffer_block_size << endl;
		cout << "p_config->max_query_num: " << p_config->max_query_num << endl;
		cout << "p_config->max_obj_num: " << p_config->max_obj_num << endl;
		cout << "p_config->round_num: " << p_config->round_num << endl;
		cout << "p_config->query_skip_round_num: " << p_config->query_skip_round_num << endl;

		cout << "tmp_i: " << tmp_i << endl;

		cout << "tot_cells_query: " << tot_cells_query << endl;
		cout << "len_seg_cache_query: " << len_seg_cache_query << endl;
		cout << "capacity_cache_query: " << capacity_cache_query << endl;

		cout << (sizeof(int) * tot_cells_query * len_seg_cache_query * capacity_cache_query) / (1024 * 1024.0) << "MiB" << endl;
		return;
	}
*/

    int *d_idx_cells_covered;
    res = cudaMalloc((void **) &d_idx_cells_covered, sizeof(int) * tot_cells_query * capacity_cache_query);
    sizecnt += sizeof(int) * tot_cells_query * capacity_cache_query;
    if (res != 0) {
        cout << "Error Point 2!" << endl;
        return;
    }

    int *d_bound_btw_cell;
    res = cudaMalloc((void **) &d_bound_btw_cell, sizeof(int) * (tot_cells_query + 2) * capacity_cache_query);
    sizecnt += sizeof(int) * (tot_cells_query + 2) * capacity_cache_query;
    if (res != 0) {
        cout << "Error Point 4!" << endl;
        return;
    }


    QueryType *d_buffer_block_query;
    res = cudaMalloc((void **) &d_buffer_block_query, sizeof(QueryType) * tmp_i * capacity_cache_query);
    sizecnt += sizeof(QueryType) * tmp_i * capacity_cache_query;
    if (res != 0) {
        cout << "Error Point 7!" << endl;
        return;
    }

    QueryQNode * h_arr_node_query = new QueryQNode[capacity_cache_query];
    for (int i = 0; i < capacity_cache_query; i++) {
        h_arr_node_query[i].tot_cells_covered = 0;
        h_arr_node_query[i].flag_cells_covered = &d_flag_cells_covered[i * tot_cells_query];

        h_arr_node_query[i].cnt_queries_per_cell = &d_cnt_queries_per_cell[i * tot_cells_query];
        //h_arr_node_query[i].queries_per_cell = &d_queries_per_cell[i * tot_cells_query * len_seg_cache_query];
        h_arr_node_query[i].idx_cells_covered = &d_idx_cells_covered[i * tot_cells_query];

        h_arr_node_query[i].bound_btw_cell = &d_bound_btw_cell[i * (tot_cells_query + 1)];

        h_arr_node_query[i].buffer_block_size_query = tmp_i;
        h_arr_node_query[i].buffer_block_query = &d_buffer_block_query[i * tmp_i];

        h_arr_node_query[i].q_size = len_node_query;
        h_arr_node_query[i].mtx_query_idx = d_query_idx;
        h_arr_node_query[i].mtx_query_nodes = &d_mec_q[i];
        h_arr_node_query[i].mtx_query_nodes_bak = &d_me_query_pool[2 * len_node_query];
        int *h_avail_queue = new int[len_node_query];//memleak
        for (int j = 0; j < len_node_query; j++) {
            h_avail_queue[j] = j;
        }
        int *d_avail_queue;
        res = cudaMalloc((void **) &d_avail_queue, sizeof(int) * len_node_query);
        if (res != 0) {
            cout << "Output::Error Point 8.0!" << endl;
        }
        cudaMemcpyAsync(d_avail_queue, h_avail_queue, sizeof(int) * len_node_query, cudaMemcpyHostToDevice, stream0);
        sizecnt += sizeof(int) * len_node_query;
        CircularQueue * h_queue_free;
        cudaMallocHost(&h_queue_free, sizeof(CircularQueue));
        h_queue_free->capacity = len_node_query;
        h_queue_free->avail_idx_bkt = d_avail_queue;
        h_queue_free->cnt_elem = len_node_query;
        h_queue_free->head = 0;
        h_queue_free->rear = 0;
        CircularQueue * d_queue_free;
        res = cudaMalloc((void **) &d_queue_free, sizeof(CircularQueue));
        if (res != 0) {
            cout << "Output::Error Point 8.1!" << endl;
        }
        cudaMemcpyAsync(d_queue_free, h_queue_free, sizeof(CircularQueue), cudaMemcpyHostToDevice, stream0);
        sizecnt += sizeof(CircularQueue);
        h_arr_node_query[i].fqueue_query = d_queue_free;
    }

    QueryQNode * d_arr_node_query;
    cudaMalloc((void **) &d_arr_node_query, sizeof(QueryQNode) * capacity_cache_query);
    cudaMemcpyAsync(d_arr_node_query, h_arr_node_query, sizeof(QueryQNode) * capacity_cache_query, \
            cudaMemcpyHostToDevice, stream0);
    sizecnt += sizeof(QueryQNode) * capacity_cache_query;

    QueryCacheArea * h_req_cache_query;
    cudaMallocHost(&h_req_cache_query, sizeof(QueryCacheArea));
    h_req_cache_query->array = d_arr_node_query;
    QueryCacheArea * d_req_cache_query;
    cudaMalloc((void **) &d_req_cache_query, sizeof(QueryCacheArea));
    cudaMemcpyAsync(d_req_cache_query, h_req_cache_query, sizeof(QueryCacheArea), \
            cudaMemcpyHostToDevice, stream0);
    sizecnt += sizeof(QueryCacheArea);

    cout << "Memory size for cache area of queries(MiB): " << sizecnt / (1024 * 1024.0) << endl << endl;



    // -------------------------------------------------------------------------------------------------
    cout << endl << "Initializing Memory for Output... " << endl;

    int *cnt_obs_per_req;
    cudaHostAlloc((void **) &cnt_obs_per_req, sizeof(int) * p_config->max_query_num, cudaHostAllocMapped);
    memset(cnt_obs_per_req, 0, sizeof(int) * p_config->max_query_num);

    int *d_cnt_obs_per_req;
    res = cudaMalloc((void **) &d_cnt_obs_per_req, sizeof(int) * p_config->max_query_num);

    cudaMemcpyAsync(d_cnt_obs_per_req, cnt_obs_per_req, sizeof(int) * p_config->max_query_num, \
                cudaMemcpyHostToDevice, stream2);
    cudaMemsetAsync(d_cnt_obs_per_req, 0, sizeof(int) * p_config->max_query_num, stream2);
    SimObject *d_obs_output;
    res = cudaMalloc((void **) &d_obs_output, sizeof(SimObject) * p_config->max_obj_num);

    if (res != 0) {
        cout << "Output::Error Point 0!" << endl;
    }

    cout << "Memory size for Output (MiB): " << \
            (sizeof(SimObject) * p_config->max_obj_num + sizeof(int) * p_config->max_query_num) / (1024 * 1024.0) \
 << endl << endl << endl;




    //---------- for queryDispatch multiqueue ---------------------
    int d_avail_qd_obj = 0;
    if (left_update_afterskip > p_config->buffer_block_size) {
        d_avail_qd_obj = 128 + 26 * (pinned_p_config->thread_analysis_num / 32 + 1) \
 + p_config->len_multiqueue + \
            (int) ((1.0f / p_config->qt_size) * (double) p_config->buffer_block_size * p_config->max_query_num / \
            (double) left_update_afterskip);
    } else {
        d_avail_qd_obj = 128 + 26 * (pinned_p_config->thread_analysis_num / 32 + 1) \
 + p_config->len_multiqueue + (int) ((1.0f / p_config->qt_size) * p_config->max_query_num);
    }
    MemItem<QueryType> *h_qd_obj_pool = new MemItem<QueryType>[d_avail_qd_obj];
    MemItem<QueryType> *d_qd_obj_pool;
    cudaMalloc((void **) &d_qd_obj_pool, sizeof(MemItem<QueryType>) * d_avail_qd_obj);

    QueryType *d_qd_query_type_pool;
    cudaMalloc((void **) &d_qd_query_type_pool, sizeof(QueryType) * d_avail_qd_obj * p_config->qt_size);
    cudaMemsetAsync(d_qd_query_type_pool, 0, sizeof(QueryType) * d_avail_qd_obj * p_config->qt_size, stream0);
    int *d_qd_anchor_pool;
    cudaMalloc((void **) &d_qd_anchor_pool, sizeof(int) * d_avail_qd_obj * p_config->qt_size);
    cudaMemsetAsync(d_qd_anchor_pool, 0, sizeof(int) * d_avail_qd_obj * p_config->qt_size, stream0);
    for (int i = 0; i < d_avail_qd_obj; i++) {
        h_qd_obj_pool[i].pool = d_qd_query_type_pool + i * p_config->qt_size;
        h_qd_obj_pool[i].cache_anchor = d_qd_anchor_pool + i * p_config->qt_size;
    }
    cudaMemcpyAsync(d_qd_obj_pool, h_qd_obj_pool, sizeof(MemItem<QueryType>) * d_avail_qd_obj, cudaMemcpyHostToDevice,
                    stream0);

    h_mm->mm_qd_query_type_pool.ptr = (void *) d_qd_query_type_pool;
    h_mm->mm_qd_query_type_pool.len = d_avail_qd_obj * p_config->qt_size;
    h_mm->mm_qd_query_type_pool.bsize = sizeof(QueryType) * d_avail_qd_obj * p_config->qt_size;

    h_mm->mm_qd_anchor_pool.ptr = (void *) d_qd_anchor_pool;
    h_mm->mm_qd_anchor_pool.len = d_avail_qd_obj * p_config->qt_size;
    h_mm->mm_qd_anchor_pool.bsize = sizeof(int) * d_avail_qd_obj * p_config->qt_size;

    h_mm->mm_qd_obj_pool.ptr = (void *) d_qd_obj_pool;
    h_mm->mm_qd_obj_pool.len = d_avail_qd_obj;
    h_mm->mm_qd_obj_pool.bsize = sizeof(MemItem<QueryType>) * d_avail_qd_obj;

    // About Circular Queue
    int d_avail_idx_anchor = d_avail_qd_obj;
    int *h_avail_idx_anchor_queue = new int[d_avail_idx_anchor];
    for (int i = 0; i < d_avail_idx_anchor; i++) {
        h_avail_idx_anchor_queue[i] = i;
    }
    int *d_avail_idx_anchor_queue;
    cudaMalloc((void **) &d_avail_idx_anchor_queue, sizeof(int) * d_avail_idx_anchor);
    cudaMemcpyAsync(d_avail_idx_anchor_queue, h_avail_idx_anchor_queue, sizeof(int) * d_avail_idx_anchor,
                    cudaMemcpyHostToDevice, stream0);
    CircularQueue * h_queue_idx_anchor_free;
    cudaMallocHost(&h_queue_idx_anchor_free, sizeof(CircularQueue));
    h_queue_idx_anchor_free->capacity = d_avail_idx_anchor;
    h_queue_idx_anchor_free->avail_idx_bkt = d_avail_idx_anchor_queue;
    h_queue_idx_anchor_free->cnt_elem = d_avail_idx_anchor;
    h_queue_idx_anchor_free->head = 0;
    h_queue_idx_anchor_free->rear = 0;
    CircularQueue * d_queue_idx_anchor_free;
    res = cudaMalloc((void **) &d_queue_idx_anchor_free, sizeof(CircularQueue));
    if (res != 0) {
        cout << "Output::Error Point 8.5!" << endl;
    }
    cudaMemcpyAsync(d_queue_idx_anchor_free, h_queue_idx_anchor_free, sizeof(CircularQueue), cudaMemcpyHostToDevice,
                    stream0);


    cout << "d_avail_qd_obj: " << d_avail_qd_obj << endl;
    cout << "Memory size for QueryDispatch Multiqueue (MiB): " << \
                (sizeof(MemItem<QueryType>) * d_avail_qd_obj + sizeof(int) * d_avail_qd_obj) / (1024 * 1024.0) \
 << endl << endl << endl;



    //-------------memory for wrap--------------
    int holdersize = 1024 * (20 + 16);
    //int* h_place_holder = new int[holdersize];
    //memset(&h_place_holder, 0, sizeof(int) * holdersize);
    int *d_place_holder;
    cudaMalloc((void **) &d_place_holder, sizeof(int) * holdersize);
    //cudaMemcpyAsync(d_place_holder, h_place_holder, sizeof(int) * holdersize, cudaMemcpyHostToDevice, stream0);
    cudaMemsetAsync(d_place_holder, 0, sizeof(int) * holdersize, stream0);
    cout << "Memory size for vwrap holder (MiB): " << \
                sizeof(int) * holdersize / (1024 * 1024.0) \
 << endl << endl << endl;



    // -----------------------------------------------------------------------------------
    cout << endl << "Initializing Memory for Grids... " << endl;

    Cell **h_cell = new Cell *[(p_config->edge_cell_num + 1) * 2];
    Cell *h_arr_cell;
    cudaMallocHost(&h_arr_cell, sizeof(Cell) * (tot_cells + 1) * 2);

    Cell **d_cell;
    Cell *d_arr_cell;

    cudaMalloc((void **) &d_cell, sizeof(Cell *) * (p_config->edge_cell_num + 1) * 2);
    cudaMalloc((void **) &d_arr_cell, sizeof(Cell) * (tot_cells + 1) * 2);

    Grid * h_index_A;
    Grid * h_index_B;
    cudaMallocHost(&h_index_A, sizeof(Grid));
    cudaMallocHost(&h_index_B, sizeof(Grid));

    Grid * d_index_A;
    Grid * d_index_B;

    cudaMalloc((void **) &d_index_A, sizeof(Grid));
    cudaMalloc((void **) &d_index_B, sizeof(Grid));

    int len_arr_bkts = p_config->len_arr_bkts_max;
    int arr_bkts_size = tot_cells + 1;

    int *h_arr_idx_bkt = new int[len_arr_bkts * arr_bkts_size * 2];

    memset(h_arr_idx_bkt, -1, sizeof(int) * len_arr_bkts * arr_bkts_size * 2);


    for (int i = 0; i < tot_cells; i++) {
        h_arr_idx_bkt[i * len_arr_bkts] = d_avail_bkt + i;
    }
    for (int i = 0; i < tot_cells; i++) {
        h_arr_idx_bkt[(i + arr_bkts_size) * len_arr_bkts] = d_avail_bkt + i;
    }

    cout << "h_arr_idx_bkt[111]: " << h_arr_idx_bkt[111] << endl;
    cout << "h_arr_idx_bkt[32]: " << h_arr_idx_bkt[32] << endl;
    int *d_arr_idx_bkt;
    res = cudaMalloc((void **) &d_arr_idx_bkt, sizeof(int) * len_arr_bkts * arr_bkts_size * 2);
    if (res != 0) {
        cout << "Output::Error Point 8.1!" << endl;
        return;
    }

    cout << "Memory size for Cell.arr_idx_bkts (MiB): " << \
                    (sizeof(int) * len_arr_bkts * arr_bkts_size * 2) / (1024 * 1024.0) \
 << endl << endl << endl;


//	cout << " cudaMemcpyAsync Point 1 " << endl;
    cudaMemcpyAsync(d_arr_idx_bkt, h_arr_idx_bkt, \
                sizeof(int) * len_arr_bkts * arr_bkts_size * 2, \
                                cudaMemcpyHostToDevice, stream0);


    h_mm->mms[2].ptr = (void *) d_arr_idx_bkt;
    h_mm->mms[2].toPtr = (void *) (d_arr_idx_bkt + arr_bkts_size * len_arr_bkts);
    h_mm->mms[2].checkpoint = d_arr_idx_bkt + arr_bkts_size * len_arr_bkts - 1;
    h_mm->mms[2].toCheckpoint = d_arr_idx_bkt + 2 * arr_bkts_size * len_arr_bkts - 1;
    h_mm->mms[2].len = arr_bkts_size * len_arr_bkts;
    h_mm->mms[2].bsize = sizeof(int) * len_arr_bkts * arr_bkts_size;


    //start freequeue;
    int *h_avail_idx_bkt_queue = new int[d_avail_bkt];
    for (int i = 0; i < d_avail_bkt; i++) {
        h_avail_idx_bkt_queue[i] = i;
        //h_avail_idx_bkt_queue[i] = i;
        //h_avail_idx_bkt_queue[i+d_avail_bkt] = i;
    }

    cout << "h_avail_idx_bkt_queue[99]: " << h_avail_idx_bkt_queue[99] << endl;
    int *d_avail_idx_bkt_queue;
    cudaMalloc((void **) &d_avail_idx_bkt_queue, sizeof(int) * d_avail_bkt);

    cudaMemcpyAsync(d_avail_idx_bkt_queue, h_avail_idx_bkt_queue, \
            sizeof(int) * d_avail_bkt, cudaMemcpyHostToDevice, stream0);

    CircularQueue * h_queue_bkts_free;
    cudaMallocHost(&h_queue_bkts_free, sizeof(CircularQueue));

    h_queue_bkts_free->capacity = d_avail_bkt;
    h_queue_bkts_free->avail_idx_bkt = d_avail_idx_bkt_queue;
    h_queue_bkts_free->cnt_elem = d_avail_bkt;
    h_queue_bkts_free->head = 0;
    h_queue_bkts_free->rear = 0;

    CircularQueue * d_queue_bkts_free;
    res = cudaMalloc((void **) &d_queue_bkts_free, sizeof(CircularQueue));

    if (res != 0) {
        cout << "Output::Error Point 8!" << endl;
        return;
    }

//	cout << " cudaMemcpyAsync Point 3 " << endl;
    cudaMemcpyAsync(d_queue_bkts_free, h_queue_bkts_free, sizeof(CircularQueue), cudaMemcpyHostToDevice, stream0);



    // end
//	h_index_A->cpuInit(p_config, h_cell, h_arr_cell, d_obs_pool, \
//			d_arr_bkts, d_arr_idx_bkt, len_arr_bkts, d_avail_bkt, 0);
//	h_index_B->cpuInit(p_config, h_cell + p_config->edge_cell_num, h_arr_cell + tot_cells, d_obs_pool,\
//			d_arr_bkts, d_arr_idx_bkt, len_arr_bkts, d_avail_bkt + tot_cells, tot_cells);
    h_index_A->cpuInit(p_config, h_cell, h_arr_cell, d_obs_pool_A, \
            d_arr_idx_bkt, len_arr_bkts, p_config->max_bucket_len, 0, 0);
    h_index_B->cpuInit(p_config, h_cell + p_config->edge_cell_num + 1, h_arr_cell + tot_cells + 1, d_obs_pool_B, \
            d_arr_idx_bkt, len_arr_bkts, p_config->max_bucket_len, 0, arr_bkts_size);

    cout << "Checking Grid..." << endl;
    cout << "Idx: " << h_index_A->cell[p_config->edge_cell_num / 2][p_config->edge_cell_num / 2].idx << "  " << \
            "Obs: " << h_index_B->cell[p_config->edge_cell_num / 2][p_config->edge_cell_num / 2].tot_obs << "  " << \
            "Subgrid: " << h_index_B->cell[p_config->edge_cell_num / 2][p_config->edge_cell_num / 2].subgrid << "  " << \
            "xmin: " << h_index_B->cell[p_config->edge_cell_num / 2][p_config->edge_cell_num / 2].rect.xmin << "  " << \
            endl;

    h_index_A->getDevicePointer(d_cell, d_arr_cell);
    //h_index_B->getDevicePointer(d_cell, d_arr_cell);
    h_index_B->getDevicePointer(d_cell + p_config->edge_cell_num + 1, d_arr_cell + tot_cells + 1);

    h_mm->mms[3].ptr = d_cell;
    h_mm->mms[3].toPtr = d_cell + p_config->edge_cell_num + 1;
    h_mm->mms[3].checkpoint = (int *) (d_cell + p_config->edge_cell_num);
    h_mm->mms[3].toCheckpoint = (int *) (d_cell + 2 * p_config->edge_cell_num + 1);
    h_mm->mms[3].len = p_config->edge_cell_num + 1;
    h_mm->mms[3].bsize = (p_config->edge_cell_num + 1) * sizeof(Cell *);

    h_mm->mms[4].ptr = d_arr_cell;
    h_mm->mms[4].toPtr = d_arr_cell + tot_cells + 1;
    h_mm->mms[4].checkpoint = &(d_arr_cell + tot_cells)->cnt_bkts;
    h_mm->mms[4].toCheckpoint = &(d_arr_cell + 2 * tot_cells + 1)->cnt_bkts;
    h_mm->mms[4].len = tot_cells + 1;
    h_mm->mms[4].bsize = (tot_cells + 1) * sizeof(Cell);


    // Caution!!! h_cell only used for initialzaion on CPU, should not be loaded to GPUs !!!
    //cudaMemcpyAsync(d_cell, h_cell, sizeof(Cell *) * p_config->edge_cell_num * 2, \
	//				cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_arr_cell, h_arr_cell, sizeof(Cell) * (tot_cells + 1) * 2, \
        cudaMemcpyHostToDevice, stream0);

    cudaMemcpyAsync(d_index_A, h_index_A, sizeof(Grid), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_index_B, h_index_B, sizeof(Grid), cudaMemcpyHostToDevice, stream0);


    cudaMemcpyAsync(d_mm, h_mm, sizeof(ManagedMemory), cudaMemcpyHostToDevice, stream0);


    // ------------------------------------------------------------------------------------
    cout << endl << "--- Launching Kernels ---" << endl;

    //load&process
    Message *message = new Message();
    message->mapsize = p_config->buffer_update_round;
    int *d_map;
    cudaMalloc((void **) &d_map, sizeof(int) * message->mapsize);
    int *h_map = new int[message->mapsize];
    for (int i = 0; i < message->mapsize; i++) {
        h_map[i] = 0;
    }
    cudaMemcpyAsync(d_map, h_map, sizeof(int) * message->mapsize, cudaMemcpyHostToDevice, stream0);
    message->d_map = d_map;
    message->h_map = h_map;
    message->d_ptr = dev_buffer_update;
    message->h_ptr = pinned_buffer_update;
    message->bsize = p_config->buffer_block_size;
    message->cursor = 0;
    message->limit = (long long int) (orig_buffer_update->buffer_len);
    message->stream_c = &stream3;
    message->start3 = &start3;
    message->stop3 = &stop3;
    message->elapsed0 = 0;
    message->time = 0;


    cout << "InitKernel()... " << endl << endl;
    InitKernel << < p_config->block_busy_num, p_config->thread_busy_num, 0, stream0 >> >
                                                                            (dev_p_config, d_obs_pool_A, d_obs_pool_B, d_sie_array, d_index_A, d_index_B, d_req_cache_update, d_req_cache_query, d_queue_bkts_free, d_place_holder, d_mm, d_map);
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

#ifdef linux
    pthread_create(&t1, NULL, copyData, (void *) message);
#else
    handle[HOST_COPY_DATA_THREAD_ID] = CreateThread(NULL, 0, copyData, (void *) message, 0, NULL);
#endif

    while (pinned_p_config->terminalFlag == 0) {
        cout << "current kernels' rate is " << pinned_p_config->block_analysis_num << " : " \
 << pinned_p_config->block_update_num << " : " \
 << pinned_p_config->block_query_num << endl;
        cout << "DistributorKernel()... " << endl;
        cudaEventRecord(start0, stream0);
#if SHARE_MEM == 1
        DistributorKernel<<<pinned_p_config->block_analysis_num, pinned_p_config->thread_analysis_num, \
            pinned_p_config->block_analysis_num * (pinned_p_config->thread_analysis_num+32) * sizeof(int),\
            stream0>>>\
                (dev_p_config, \
                dev_buffer_update, dev_cnt_update, \
                dev_buffer_query, dev_cnt_query, \
                 d_req_cache_update, d_req_cache_query, \
                d_index_A, d_index_B, d_queue_bkts_free, \
                d_qd_obj_pool, d_queue_idx_anchor_free, d_qd_query_type_pool, d_qd_anchor_pool, \
                d_place_holder, d_mm);
#else
        DistributorKernel << < pinned_p_config->block_analysis_num, pinned_p_config->thread_analysis_num, 0,
                stream0 >> > \
                (dev_p_config, \
                dev_buffer_update, dev_cnt_update, \
                dev_buffer_query, dev_cnt_query, \
                 d_req_cache_update, d_req_cache_query, \
                d_index_A, d_index_B, d_queue_bkts_free, \
                d_qd_obj_pool, d_queue_idx_anchor_free, d_qd_query_type_pool, d_qd_anchor_pool, \
                d_place_holder, d_mm);
#endif
        cudaEventRecord(stop0, stream0);

        cout << "UpdateKernel()... " << endl;
        cudaEventRecord(start1, stream1);
        UpdateKernel << < pinned_p_config->block_update_num, pinned_p_config->thread_update_num, 0, stream1 >> >
                                                                                                    (dev_p_config);
        cudaEventRecord(stop1, stream1);

        cout << "QueryKernel()... " << endl << endl;
        cudaEventRecord(start2, stream2);
        QueryKernel << < pinned_p_config->block_query_num, pinned_p_config->thread_query_num, 0, stream2 >> >
                                                                                                 (d_obs_output, d_cnt_obs_per_req);
        cudaEventRecord(stop2, stream2);

        //cudaDeviceSynchronize();
        cudaStreamSynchronize(stream0);
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        cudaEventSynchronize(stop0);
        cudaEventSynchronize(stop1);
        cudaEventSynchronize(stop2);
        cudaEventSynchronize(realstart);

        cudaMemcpyAsync(pinned_p_config, dev_p_config, sizeof(GConfig), \
                cudaMemcpyDeviceToHost, stream0);
        cudaStreamSynchronize(stream0);

        //cudaDeviceSynchronize();
        cudaEventElapsedTime(&elapsed0, start0, stop0);
        cudaEventElapsedTime(&elapsed1, start0, stop1);
        cudaEventElapsedTime(&elapsed2, start0, stop2);
        elapsed0_sum += elapsed0;
        elapsed1_sum += elapsed1;
        elapsed2_sum += elapsed2;

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }
    }


    cudaMemcpyAsync(complete_cnt_update, dev_cnt_update, sizeof(int), cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(complete_cnt_query, dev_cnt_query, sizeof(int), cudaMemcpyDeviceToHost, stream0);
    cudaMemcpy(cnt_obs_per_req, d_cnt_obs_per_req, sizeof(int) * p_config->max_query_num, \
            cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
#ifdef linux
    pthread_join(t1, NULL);
#else
    WaitForMultipleObjects(1, handle, TRUE, INFINITE);
#endif
    printf("\ncopydata clock: %f ms\n", message->elapsed0);

    cout << endl << "Updates: " << *complete_cnt_update << "\tQueries: " << *complete_cnt_query << endl;

    cout << endl << "Freeing memory ..." << endl;
    cudaFreeHost(complete_cnt_update);
    cudaFreeHost(complete_cnt_query);
    cudaFreeHost(pinned_buffer_update);
    cudaFreeHost(pinned_buffer_query);
    //cudaFreeHost(h_obs_pool);
    cudaFreeHost(h_sie_array);
    cudaFreeHost(h_req_cache_update);
    cudaFreeHost(h_req_cache_query);
    cudaFreeHost(h_arr_cell);
    cudaFreeHost(h_index_A);
    cudaFreeHost(h_index_B);
    cudaFreeHost(h_queue_bkts_free);
    cudaFree(dev_p_config);
    cudaFree(dev_buffer_update);
    cudaFree(dev_buffer_query);
    cudaFree(dev_cnt_update);
    cudaFree(dev_cnt_query);

    cout << endl << "=== Time Report ===" << endl;
    cout << "Distributor Kernel: " << elapsed0_sum << " ms" << endl;
    cout << "Update Kernel: " << elapsed1_sum << " ms" << endl;
    cout << "Query Kernel: " << elapsed2_sum << " ms" << endl;

    cout << endl << "Objects of each query..." << endl;
    for (int i = p_config->max_query_num * 1 / 4, tick = 0; i < p_config->max_query_num; i++) {
        if (cnt_obs_per_req[i] != 0) {
            cout << cnt_obs_per_req[i] << " ";
            tick++;
        }
        if (tick == 32) {
            break;
        }
    }
    cout << endl;

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    cout << endl << "Done!!!" << endl;
}


