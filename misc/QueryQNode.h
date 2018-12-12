/**********************************************************************
* QueryQNode.h
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Date: Jan 21, 2015 | 8:54:27 PM
* Description:*  
* Licence:*
**********************************************************************/


#ifndef QUERYQNODE_H_
#define QUERYQNODE_H_

#include "ObjBox.cuh"

class QueryQNode {
public:

    int tot_cells_covered;

    int *flag_cells_covered;

    volatile int *volatile idx_cells_covered;

    int *queries_per_cell;
    int *cnt_queries_per_cell;

    int *bound_btw_cell;


    int buffer_block_size_query;
    QueryType *buffer_block_query;
    int offset_buffer_query;

    int q_size;
    int *mtx_query_idx;
    MemElementCollection<int> *mtx_query_nodes;
    MemElement *mtx_query_nodes_bak;
    CircularQueue *fqueue_query;

public:
    QueryQNode(void) {
        tot_cells_covered = 0;
        flag_cells_covered = NULL;

        idx_cells_covered = NULL;

        queries_per_cell = NULL;
        cnt_queries_per_cell = NULL;

        bound_btw_cell = NULL;
        buffer_block_size_query = 0;
        buffer_block_query = NULL;
    }
};


#endif /* QUERYQNODE_H_ */
