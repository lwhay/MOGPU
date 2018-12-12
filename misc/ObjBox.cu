/************************************************************
* ObjBox.cu
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 13, 2014*  9:29:53 AM* 
* Description: Cloned from MoveObj.cpp
* Licence:*
************************************************************/

#include "ObjBox.cuh"


ObjBox::ObjBox(void)
{
    oid = -1;     //表示当前对象无效
    x = y = -1;
//    vx = vy = -1;
    time = -1;
}

__device__ void  ObjBox::flushByUpdate(UpdateType *update_ins)
{
	oid = update_ins->oid;
	x = update_ins->x;
	y = update_ins->y;
//	vx = update_ins->vx;
//	vy = update_ins->vy;
	time = update_ins->time;
}

__device__ void ObjBox::flushByObj(ObjBox *cand)
{
	oid = cand->oid;
	x = cand->x;
	y = cand->y;
//	vx = cand->vx;
//	vy = cand->vy;
	time = cand->time;
}

__device__ bool ObjBox::isInRegion(float xmin, float ymin, \
		float xmax, float ymax)
{
    if (x >= xmin && x< xmax && y >= ymin && y < ymax)
        return true;
    else
        return false;
}

__device__ void ObjBox::clear(void)
{
	oid = -1;     //表示当前对象无效
	x = y = -1;
//	vx = vy = 0;
	time = 0;
}




