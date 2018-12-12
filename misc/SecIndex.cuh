/************************************************************
* SecIndex.h
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 11, 2014*  4:16:26 PM* 
* Description:*  
* Licence:*
************************************************************/

#ifndef SECINDEX_H_
#define SECINDEX_H_

//#include <hash_map>

#include "config/GConfig.h"
#include "SIEntry.cuh"


//using namespace __gnu_cxx;

class SecIndex
{
public:
    SIEntry *index;
public:
    __device__ SecIndex(void)
    {
    	index = NULL;
    }
    __device__ ~SecIndex(void)
    {
//    	delete[] index;
    }


    __device__ SIEntry *getPtrSIE(int oid)
    {
        return &index[oid];
    }

    __device__ void clear()
    {
//        index.clear();
    }
};


#endif /* SECINDEX_H_ */
