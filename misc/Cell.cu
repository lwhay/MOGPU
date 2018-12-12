/************************************************************
* Cell.cpp
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 11, 2014*  2:20:35 PM* 
* Description:*  
* Licence:*
************************************************************/

#include "Cell.cuh"
#include "config/GConfig.h"

Cell::Cell(void) {
    idx = -1;
    subgrid = -1;

    len_bkt = -1;
    idx_bkt_init = -1;
    cnt_bkts = 0;

    tot_obs = 0;

    tot_obs_top = 0;
    //obs = NULL;


    memfencedelay = 0;

    len_arr_bkts = 0;
    //arr_bkts = NULL;
    //arr_idx_bkt = NULL;
}

Cell::~Cell(void) {
}

void Cell::initBucket(Config *p_config, ObjBox *d_obs_pool, \
        int *d_arr_idx_bkt, int _len_arr_bkts, int _offset_in_obs_pool, int _offset_in_arr_bkts) {
    len_bkt = p_config->max_bucket_len;
    tot_obs = 0;

    idx_bkt_init = _offset_in_obs_pool + idx;

    cnt_bkts = 1;

    len_arr_bkts = _len_arr_bkts;
}

void Cell::setIdx(int _idx) {
    idx = _idx;
}

void Cell::setRect(Rect &_rect) {
    rect = _rect;
}


void Cell::setSubgrid(int _subgrid) {
    subgrid = _subgrid;
}


__device__ void Cell::readLock(void) {
    ;
}

__device__ void Cell::readUnlock(void) {
    ;
}

__device__ void Cell::writeLock(void) {
    ;
}

__device__ void Cell::writeUnlock(void) {
    ;
}




