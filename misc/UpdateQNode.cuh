/**********************************************************************
* UpdateQNode.h
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Nov 1, 2014 | 9:11:54 PM
* Description:*  
* Licence:*
**********************************************************************/

#ifndef UPDATEQNODE_H_
#define UPDATEQNODE_H_

#include <string.h>
#include "misc/Cell.cuh"
#include "misc/BaseStruct.h"

#ifdef linux
extern __device__ int len_seg_cache_update;
#else
extern "C" __device__ int len_seg_cache_update;
#endif


class UpdateQNode
{
public:

	int seg;
	int len;

	int num_cells;

	int d_size;
	int *mtx_delete_idx;
	//MemItem<int> *mtx_delete_node;
	MemElementCollection<int>* mtx_delete_nodes;
	int *mtx_delete_pool;
	CircularQueue* fqueue_delete;
	int *sum_d;

	int i_size;
	int *mtx_insert_idx;
	//MemItem<UpdateType> *mtx_insert_node;
	MemElementCollection<UpdateType>* mtx_insert_nodes;
	UpdateType *mtx_insert_pool;
	CircularQueue* fqueue_insert;
	int *sum_i;

	int f_size;
	int	*mtx_fresh_idx;
	//MemItem<UpdateType> *mtx_fresh_node;
	MemElementCollection<UpdateType>* mtx_fresh_nodes;
	UpdateType *mtx_fresh_pool;
	CircularQueue* fqueue_fresh;
	int *sum_f;

	int lock;
	//for memcpy
	MemElement* mtx_insert_nodes_bak;
	MemElement* mtx_delete_nodes_bak;

public:
	__device__ UpdateQNode(void)//deprecated
	{
		lock = 0;

		num_cells = gp_config->edge_cell_num * gp_config->edge_cell_num;
		seg = len_seg_cache_update;
		len = seg * num_cells;

		mtx_delete_idx = (int *)malloc(len * sizeof(int));
		mtx_insert_idx = (int *)malloc(len * sizeof(int));
		mtx_fresh_idx = (int *)malloc(len * sizeof(int));

		sum_i = new int[num_cells];
		memset(sum_i, 0, sizeof(int) * num_cells);
		sum_d = new int[num_cells];
		memset(sum_d, 0, sizeof(int) * num_cells);
		sum_f = new int[num_cells];
		memset(sum_f, 0, sizeof(int) * num_cells);

	}

	__device__  ~UpdateQNode(void)//deprecated
	{

	}

};


#endif /* UPDATEQNODE_H_ */
