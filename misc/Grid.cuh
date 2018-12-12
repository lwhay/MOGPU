/************************************************************
* Grid.h
* Copyright (c) Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 11, 2014*  2:34:09 PM* 
* Description:*  
* Licence:*
************************************************************/

#ifndef GRID_H_
#define GRID_H_

#include "misc/Cell.cuh"


class Grid
{
public:
	Rect rect;							//Grid所代表的矩形范围
    int cell_num;                      //Grid一个边上的CELL数，必须是2的幂
    Cell **cell;						//Grid: two dimensional cell
    Cell * arr_cell;
	ObjBox * obs_pool;
    int *arr_idx_bkt;
	int LEN_ARR_BKTS;
	int MAX_BKT_LEN;
	int offset_bkts;
	int offset_obspool;
	int cursor_update_wrap;
	int cursor_query_wrap;
	int block_read;
	int block_write;
public:
    Grid(void);
	Grid(int);
    void cpuInit(Config *p_config, Cell **_cell, Cell *_arr_cell, ObjBox *d_obs_pool, \
    		int *d_arr_idx_bkt, int _len_arr_bkts, int max_bkt_len, int _offset_in_obs_pool, int _offset_in_arr_bkts);
    void getDevicePointer(Cell **d_cell, Cell *d_arr_cell);
    __device__ void adjustPointer(void);

    //根据Cell编号获取Cell
    __device__ Cell *getCellByIdx(int idx);
    //根据行号和列号获取Cell
    __device__ Cell *getCellByRC(int i,int j);
    //根据横纵坐标获取Cell
    __device__  Cell *getCellByXY(float x,float y);
	__device__  Cell *getNearestCellByXY(float x, float y);
    __device__ void clear();

	__device__ int* getArrIdxBkt(int cellId);
	__device__ ObjBox* getBkt(int cellId, int bktId);
};

#endif /* GRID_H_ */
