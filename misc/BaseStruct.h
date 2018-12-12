/************************************************************
* BaseStruct.h
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 10, 2014
* Description:
* Licence:
************************************************************/

#ifndef linux
#define NULL 0
#endif

// 0:	Enable dynamic memory allocation
// 1:	Fixed-size memory allocation
#ifndef SEG_CACHE
#define SEG_CACHE 0
#endif

#ifndef SHARE_MEM
#define SHARE_MEM 0
#endif

// 2000:	Preemption buckect-based query with xy optimization
// 20:		Preemption cell-based query
// 30:		Preemption query-based
// 3:		Stepping query-based
#ifndef QUERY_PATTERN
#define QUERY_PATTERN 2000
#endif

#ifndef CHECK_OBJ_BEFOREQUERY
#define CHECK_OBJ_BEFOREQUERY 0
#endif

#ifndef CHECK_SI
#define CHECK_SI 1
#endif

// 3:	Preemption update group-based
// 2:	Stepping update group-based
// 1:	Preemption update cell-based
// 0:	Stepping update cell-based
#ifndef DW
#define DW 1
#endif

// 1:	Enable multi-level queue
// 0:	Disable multi-level queue
#ifndef USE_MULTIQUEUE
#define USE_MULTIQUEUE 1
#endif

#ifndef USE_DPPROCESS
#define USE_DPPROCESS 0
#endif

#ifndef CHECK_UPDATE_MEMPOOL
#define CHECK_UPDATE_MEMPOOL 0
#endif

#ifndef CHECK_QUERY_MEMPOOL
#define CHECK_QUERY_MEMPOOL 0
#endif

// 1:	Enable rebalance
// 0:	Disable rebalance
#ifndef REBALANCE
#define REBALANCE 0
#endif

#ifndef UPDATE_DISPATCH_SEG
#define UPDATE_DISPATCH_SEG 5
#endif

#ifndef IGNORE_CNT
#define IGNORE_CNT 0
#endif


#ifndef BASESTRUCT_H_
#define BASESTRUCT_H_

struct UpdateType {
    int oid;
    float x;
    float y;
    float vx;
    float vy;
    float time;
};

struct UpdateTypeForSort {
    int id;
    UpdateType ut;
};

enum QueryStyle {
    UNKNOWN_TYPE = 0,
    AREA_STATIC = 1,
    AREA_MOVING = 2,
    kNN_STATIC = 3,
    kNN_MOVING = 4
};

struct QueryType {
    QueryStyle qs;
    int qid;
    float minX;         //若为范围查询，一下四个代表查询范围；若为KNN查询则用minX和minY代表查询点的坐标
    float minY;
    float maxX;
    float maxY;
    int k;              //KNN中的K值
    float t;
};

struct SimUpdate {
    int oid;
    float x;
    float y;
    float time;
};

struct SimObject {
    int oid;
    float x;
    float y;
    float time;
};

struct CircularQueue {
    int head;
    int rear;

    int capacity;
    int cnt_elem;

    int *avail_idx_bkt;
};

template<class T>
struct MemItem {
    int id;
    int cnt;
    int queuelen;
    int len;
    //QueueNode* next;
    //QueueNode* last;
    int next;
    int last;
    int lock;
    T *pool;
    int *cache_anchor;
};

struct MemElement {
    //int id;
    //int queuelen;
    int next;
    int lock;
};


template<class T>
struct MemElementCollection {
    int *cnt;
    int *last;
    int LEN;
    MemElement *mes;
    T *pool;
    int *cache_anchor;
    int globalCnt;
};


struct MMItem {
    int *checkpoint;
    int *toCheckpoint;
    void *ptr;
    int len;
    size_t bsize;
    void *toPtr;
};


struct ManagedMemory {
    int mmsLen;
    // mms[0]: d_obs_pool;
    // mms[1]: d_sie_array;
    // mms[2]: d_arr_idx_bkt;
    // mms[3]: d_cell;
    // mms[4]: d_arr_cell;
    MMItem mms[5];

    /* ignore */
    MMItem mm_qd_query_type_pool;
    MMItem mm_qd_anchor_pool;
    MMItem mm_qd_obj_pool;
};


#endif /* BASESTRUCT_H_ */
