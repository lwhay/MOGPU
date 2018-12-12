/************************************************************
* DeriveStruct.h
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 11, 2014*  5:12:36 PM* 
* Description:*  
* Licence:*
************************************************************/

#ifndef DERIVESTRUCT_H_
#define DERIVESTRUCT_H_

#include "misc/BaseStruct.h"
#include "misc/Cell.cuh"

typedef struct _tag_update_entity_
{
    UpdateType update_ins;
    Cell *p_cell;
}UpdateEntity;

#endif /* DERIVESTRUCT_H_ */
