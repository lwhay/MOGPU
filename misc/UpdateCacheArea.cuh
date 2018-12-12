/**********************************************************************
* UpdateCacheArea.h
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Date: Dec 30, 2014 | 2:54:32 PM
* Description:*  
* Licence:*
**********************************************************************/


#ifndef UPDATECACHEAREA_H_
#define UPDATECACHEAREA_H_

#include "UpdateQNode.cuh"


class UpdateCacheArea
{
public:
	int token0;
	int token1;
	int cnt0;
	int cnt1;
	UpdateQNode *array;

public:
	__device__ UpdateCacheArea(void)
	{
		token0 = 0;
		token1 = 0;
		cnt0 = 0;
		cnt1 = 0;
		array = new UpdateQNode[2];
	}

};


#endif /* UPDATECACHEAREA_H_ */
