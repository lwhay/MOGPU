/************************************************************
* GConfig.h
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 15, 2014*  8:34:01 PM* 
* Description:*  
* Licence:*
************************************************************/

#ifndef GCONFIG_H_
#define GCONFIG_H_

#include "misc/BaseStruct.h"

typedef struct _tag_g_config_ {
    int period_len;                 //统计周期时长
    int period_num;                 //统计周期个数
    // region
    float region_xmin;
    float region_xmax;
    float region_ymin;
    float region_ymax;
    // data
    int max_obj_num;                //最大对象数量
    int round_num;                  //最大更新数量
    int query_skip_round_num;
    int max_query_num;              //最大查询数量
    int gaussian_data;              //是否使用高斯分布的数据
    int uniform_data;               //是否使用均匀分布的数据
    int hotspot_num;                //使用高斯分布时热点个数
    //cell
    int edge_cell_num;              //单边cell数量
    //_struct
    int max_bucket_len;   // >32!!!          //bkt最大长度
    int buffer_block_size;          //分析段长度
    //thread
    int block_analysis_num;
    int thread_analysis_num;        //分析线程数量
    int block_update_num;
    int thread_update_num;          //更新线程数量
    int block_query_num;           //
    int thread_query_num;           //查询线程数量
    int block_busy_num;
    int thread_busy_num;
    //scheme
    int single_stream_mutual_access;    //单数据流互斥访问
    int single_stream_para_access;    //单数据流并行访问
    //int multi_stream_parall_access;      //多数据流并行访问
    //int single_stream_parall_access_asynch;    //单数据流并行异步访问
    //int update_query_same_thread;    //更新和查询合并
    int open_correct_test;

    //other struct
    int len_seg_cache_update;
    int side_len_vgroup;
    int len_seg_cache_query;

    int len_seg_multiqueue;
    int len_multiqueue;
    int qt_size;

    int clockRate;
    int len_arr_bkts_max;
    int buffer_update_round;

    int terminalFlag;
} GConfig;

extern GConfig *host_p_config;
extern GConfig *dev_p_config;
// config information on GPU
extern __device__ GConfig
*
gp_config;

__global__ void copyKernel(GConfig * dev_p_gconfig);

void initHostGConfig(void);

int get_GPU_Rate(void);


#endif /* GCONFIG_H_ */
