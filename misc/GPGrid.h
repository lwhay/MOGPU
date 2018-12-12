/**********************************************************************
* GPGrid.h
* Copyright @ Cloud Computing Lab, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Oct 23, 2014 | 10:10:10 PM
* Description:*  
* Licence:*
**********************************************************************/


#ifndef GPGRID_H_
#define GPGRID_H_

#ifndef linux
#define HOST_COPY_DATA_THREAD_ID    0
#define HOST_COPY_DATA_THREAD_NUM   1
#endif

void launchGPGrid(void);


#endif /* GPGRID_H_ */
