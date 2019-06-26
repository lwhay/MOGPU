/************************************************************
* Cell.h
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 11, 2014*  2:10:42 PM*
* Description:*  
* Licence:*
************************************************************/

#ifndef CELL_H_
#define CELL_H_

#include "Rect.cuh"
#include "ObjBox.cuh"
#include "config/Config.h"

class Cell {
public:
    int idx;
    int subgrid;
    Rect rect;

    int len_bkt;
    int idx_bkt_init;

    int tot_obs;

    int tot_obs_top;
    //ObjBox *obs;

    int memfencedelay;

    int len_arr_bkts;
    //ObjBox **arr_bkts;
    //int *arr_idx_bkt;
    int cnt_bkts;

public:

    Cell(void);

    ~Cell(void);

    void setIdx(int _idx);

    void setRect(Rect &_rect);

    void setSubgrid(int _subgrid);

    void initBucket(Config *p_config, ObjBox *d_obs_pool, \
            int *d_arr_idx_bkt, int _len_arr_bkts, int _offset_in_obs_pool, int _offset_in_arr_bkts_pool);

    __device__ void writeLock(void);

    __device__ void writeUnlock(void);

    __device__ void readLock(void);

    __device__ void readUnlock(void);

};

#endif /* CELL_H_ */
