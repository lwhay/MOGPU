/**********************************************************************
* QueryCacheArea.h
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Date: Jan 21, 2015 | 9:09:38 PM
* Description:*  
* Licence:*
**********************************************************************/


#ifndef QUERYCACHEAREA_H_
#define QUERYCACHEAREA_H_

#include "QueryQNode.h"

class QueryCacheArea
{
public:
	int token0;
	int token1;
	int cnt0;
	int cnt1;
	QueryQNode *array;

public:
	QueryCacheArea(void)
	{
		token0 = 0;
		token1 = 0;
		cnt0 = 0;
		cnt1 = 0;
		array = NULL;
	}
};


#endif /* QUERYCACHEAREA_H_ */
