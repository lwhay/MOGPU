/************************************************************
* SIEntry.h
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 11, 2014*  3:05:39 PM* 
* Description: Derived form Location.h
* Licence:*
************************************************************/

#ifndef SIENTRY_H_
#define SIENTRY_H_

#include "ObjBox.cuh"
#include "misc/Cell.cuh"

class SIEntry {
public:
    int idx_cell;
    int idx_bkt;
    int idx_obj;
    int memfence_delay;

public:
    SIEntry();

    __device__ void init(void);

    __device__ int getIdx(void);

    __device__ void clear(void);

    __device__ void rwLock();

    __device__ void rwUnLock();

    __device__ bool tryLock();
};

#ifdef linux
// Pre-allocated memory for second index
extern __device__ SIEntry
*
sie_array;
extern __device__ int pitch_sie;
extern __device__ int memfence_delay_step;
#else
// Pre-allocated memory for second index
extern "C" __device__ SIEntry *sie_array;
extern "C" __device__ int pitch_sie;
extern "C" __device__ int memfence_delay_step;
#endif
#endif /* SIENTRY_H_ */
