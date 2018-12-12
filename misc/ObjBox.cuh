/************************************************************
* ObjBox.h
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 13, 2014*  9:27:59 AM* 
* Description: Cloned from MoveObj.h
* Licence:*
************************************************************/

#ifndef OBJBOX_H_
#define OBJBOX_H_

#include "misc/BaseStruct.h"

class ObjBox
{
public:

	int		oid;
	float	x;
	float	y;
//	float   vx;
//	float   vy;
	float	time;


public:

    ObjBox(void);
	__device__ void flushByUpdate(UpdateType *update_ins);
	__device__ void flushByObj(ObjBox *cand);
	__device__ bool isInRegion(float xmin, float ymin, \
			float xmax, float ymax);
	__device__ void clear(void);
};

#ifdef linux
// Pre-allocated memory for buckets
extern __device__ unsigned int counter_bucket;
//extern __device__ ObjBox *obs_pool;
extern __device__ ObjBox *obs_pool_A;
extern __device__ ObjBox *obs_pool_B;
#else
// Pre-allocated memory for buckets
extern "C" __device__ unsigned int counter_bucket;
//extern __device__ ObjBox *obs_pool;
extern "C" __device__ ObjBox *obs_pool_A;
extern "C" __device__ ObjBox *obs_pool_B;
#endif

#endif /* OBJBOX_H_ */
