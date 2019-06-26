#ifndef MEMPOOL_H_
#define MEMPOOL_H_

#include <stdio.h>

template<class DataType1>
class MemnodePool {
public:
    CircularQueue *free_queue;
    MemItem <DataType1> *pool;
    int pagesize;
public:
    inline bool init(int queuesize, int *d_avail_queue, int pagesize, DataType1 *origin_pool, int opoolsize) {
        CircularQueue *h_avail_queue = new CircularQueue();
        h_avail_queue->capacity = queuesize;
        h_avail_queue->avail_idx_bkt = d_avail_queue;
        h_avail_queue->cnt_elem = queuesize;
        h_avail_queue->head = 0;
        h_avail_queue->rear = 0;
        CircularQueue *d_avail_queue;
        delete h_avail_queue;
        return true;
    }

    inline int pop() {
        int cnt_elem = atomicAdd(&free_queue->cnt_elem, -1);
        int idx_in_queue = atomicAdd(&free_queue->head, 1);
        int idx_anchor_in_pool = free_queue->avail_idx_bkt[idx_in_queue % free_queue->capacity];
        if (free_queue->avail_idx_bkt[idx_anchor_in_pool] == -1) {
            printf("idx_anchor_in_pool empty error!");
            errorFlag = true;
            return -1;
        }
        atomicExch(&d_queue_idx_anchor_free->avail_idx_bkt[idx_in_queue], -1);
        MemItem <DataType1> *qn = &pool[idx_anchor_in_pool];
        atomicExch(&qn->id, idx_anchor_in_pool);
        atomicExch(&qn->cnt, 0);
        atomicExch(&qn->queuelen, 1);
        atomicExch(&qn->len, pagesize);
        atomicExch(&qn->next, -1);
        atomicExch(&qn->last, idx_anchor_in_pool);
        atomicExch(&qn->lock, 0);
        return idx_anchor_in_pool;
    }

    inline DataType1 *get(int idx) {
        return &pool[idx];
    }

    //clear
    __device__ inline bool init_free_queue() {
        for (int i = 0; i < free_queue->capacity; i++) {
            atomicExch(&free_queue->avail_idx_bkt[i], i);
        }
        atomicExch(&free_queue->cnt_elem, free_queue->capacity);
        atomicExch(&free_queue->head, 0);
        atomicExch(&free_queue->rear, 0);
        return true;
    }
};


template<class DataType2, class DataType3>
class Mempool2 {
public:

public:

};

#endif
