/**********************************************************************
* QueryKernel.h
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Oct 28, 2014 | 02:18:59 PM
* Description:*  
* Licence:*
**********************************************************************/


#ifndef QUERYKERNEL_H_
#define QUERYKERNEL_H_

#include "misc/BaseStruct.h"

//__global__ void QueryKernel(SimObject *d_obs_output, int *d_cnt_obs_per_req, QueryType *dev_buffer_query);
__global__ void QueryKernel(SimObject *d_obs_output, int *d_cnt_obs_per_req);


#endif /* QUERYKERNEL_H_ */
