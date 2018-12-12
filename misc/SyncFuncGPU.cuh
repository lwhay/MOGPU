/**********************************************************************
* SyncFuncGPU.h
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Date: Feb 11, 2015 | 3:28:25 PM
* Description:*  
* Licence:*
**********************************************************************/


#ifndef SYNCFUNCGPU_H_
#define SYNCFUNCGPU_H_

__device__ void sync_func_query(void);
__device__ void sync_func_update(void);
__device__ void sync_func_dist(void);

#endif /* SYNCFUNCGPU_H_ */
