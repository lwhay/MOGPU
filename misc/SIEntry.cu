/************************************************************
* SIEntry.cu
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 11, 2014*  4:00:20 PM* 
* Description: Derived from Location.cpp
* Licence:*
************************************************************/

#include "SIEntry.cuh"


SIEntry::SIEntry() {
    idx_cell = -1;
    idx_bkt = -1;
    idx_obj = -1;
    memfence_delay = 0;
}

__device__ int memfence_delay_step = 5;//32step


__device__ void SIEntry::init(void) {

}

__device__ void SIEntry::clear(void) {

}


__device__ void SIEntry::rwLock() {
}

__device__ void SIEntry::rwUnLock() {

}

__device__ bool SIEntry::tryLock() {
    return true;
}




