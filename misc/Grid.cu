/************************************************************
* Grid.cu
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 11, 2014*  2:43:51 PM* 
* Description:*  
* Licence:*
************************************************************/

#include "Grid.cuh"
#include "config/Config.h"
#include "config/GConfig.h"
#include "device/DeviceGlobalVar.cuh"

Grid::Grid(void) {
    cell_num = -1;                      //Grid一个边上的CELL数，必须是2的幂
    cell = NULL;                        //二维CELL数组
    arr_cell = NULL;
    cursor_update_wrap = 0;
}

Grid::Grid(int num) {
    cell_num = num;
    arr_cell = (Cell * )
    malloc(sizeof(Cell) * cell_num);
    cursor_update_wrap = 0;
}

void Grid::cpuInit(Config *p_config, Cell **_cell, Cell *_arr_cell, ObjBox *d_obs_pool, \
        int *d_arr_idx_bkt, int _len_arr_bkts, int _max_bkt_len, int _offset_in_obs_pool, int _offset_in_arr_bkts) {
    MAX_BKT_LEN = _max_bkt_len;
    obs_pool = d_obs_pool;

    arr_idx_bkt = d_arr_idx_bkt;
    offset_bkts = _offset_in_arr_bkts;
    LEN_ARR_BKTS = _len_arr_bkts;
    offset_obspool = _offset_in_obs_pool;

    cell_num = p_config->edge_cell_num;
    //初始化CELL数组
    cell = _cell;
    arr_cell = _arr_cell;
    for (int i = 0; i < cell_num; i++) {
        cell[i] = arr_cell + i * cell_num;
    }
    //subgrid划分策略
    int subrow = cell_num, subcol = cell_num;

/*	if (cell_num > p_config->sub)
	{
		subrow = 128;
		subcol = 128;
	}*/

    int len_vgroup = cell_num * cell_num / (subrow * subcol);
    int subgrid = 0;
    rect.setCoverArea(p_config->region_xmin, p_config->region_ymin, \
            p_config->region_xmax, p_config->region_ymax);
    float piece_x = rect.width / (float) cell_num;
    float piece_y = rect.height / (float) cell_num;
    Rect tmp_rect;
    int i, j;
//	int k, l;
    for (i = 0; i < cell_num; i++) {
        for (j = 0; j < cell_num; j++) {

            cell[i][j].setIdx(i * cell_num + j);
            subgrid = cell[i][j].idx / len_vgroup;
            cell[i][j].setSubgrid(subgrid);

            tmp_rect.xmax = rect.xmin + (float) (j + 1) * piece_x;
            tmp_rect.xmin = rect.xmin + (float) j * piece_x;
            tmp_rect.ymax = rect.ymax - (float) i * piece_y;
            tmp_rect.ymin = rect.ymax - (float) (i + 1) * piece_y;
            cell[i][j].setRect(tmp_rect);

            cell[i][j].initBucket(p_config, d_obs_pool, d_arr_idx_bkt, \
                    _len_arr_bkts, _offset_in_obs_pool, _offset_in_arr_bkts);
        }
    }


}

void Grid::getDevicePointer(Cell **d_cell, Cell *d_arr_cell) {
    cell = d_cell;
    arr_cell = d_arr_cell;
}

__device__ void Grid::adjustPointer(void) {
    cell = NULL;
//	for (int i = 0; i < cell_num; i++)
//	{
//		cell[i] = arr_cell + i * cell_num;
//	}

}


__device__ int *Grid::getArrIdxBkt(int cellId) {
    return arr_idx_bkt + (offset_bkts + cellId) * LEN_ARR_BKTS;
}

__device__ ObjBox
*

Grid::getBkt(int cellId, int bktoffset) {
    return obs_pool + getArrIdxBkt(cellId)[bktoffset] * MAX_BKT_LEN;
}


__device__ Cell
*

Grid::getCellByIdx(int idx) {
    return &cell[idx / cell_num][idx % cell_num];
}

__device__ Cell
*

Grid::getCellByRC(int i, int j) {
//    return &cell[i][j];
    return &arr_cell[i * cell_num + j];
}

__device__ Cell
*

Grid::getCellByXY(float x, float y) {
    if (x < rect.xmin || x > rect.xmax || y > rect.ymax || y < rect.ymin) {
        return NULL;
    }

    int x_idx = (x - rect.xmin) * cell_num / rect.width;
    if (x_idx == cell_num)
        x_idx = cell_num - 1;
    int y_idx = (rect.ymax - y) * cell_num / rect.height;
    if (y_idx == cell_num)
        y_idx = cell_num - 1;
    return arr_cell + y_idx * cell_num + x_idx;

}

__device__ Cell
*

Grid::getNearestCellByXY(float x, float y) {

    if (x > rect.xmax) x = rect.xmax;
    if (x < rect.xmin) x = rect.xmin;
    if (y > rect.ymax) y = rect.ymax;
    if (y < rect.ymin) y = rect.ymin;


    int x_idx = (x - rect.xmin) * cell_num / rect.width;
    if (x_idx == cell_num)
        x_idx = cell_num - 1;
    int y_idx = (rect.ymax - y) * cell_num / rect.height;
    if (y_idx == cell_num)
        y_idx = cell_num - 1;
    return arr_cell + y_idx * cell_num + x_idx;

}


__device__ void Grid::clear() {
}


