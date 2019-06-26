/************************************************************
* GConfig.cu
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 15, 2014*  8:35:37 PM* 
* Description:*  
* Licence:*
************************************************************/
#include "Config.h"
#include "GConfig.h"
#include "device/DeviceGlobalVar.cuh"
#include <iostream>

using namespace std;


GConfig *host_p_config = new GConfig;
GConfig *dev_p_config;

// config information on GPU
__device__ GConfig
*
gp_config;

int get_GPU_Rate(void) {
    cudaDeviceProp deviceProp;//CUDA����Ĵ洢GPU���ԵĽṹ��
    cudaGetDeviceProperties(&deviceProp, 0);//CUDA���庯��
    cout << "wrap size: " << deviceProp.warpSize << endl;
    return deviceProp.clockRate;
}


void initHostGConfig(void) {
    Config *p_config = Config::getInstance();
    host_p_config = (GConfig *) malloc(sizeof(GConfig));
    host_p_config->period_len = p_config->period_len;
    host_p_config->period_num = p_config->period_num;
    // region
    host_p_config->region_xmin = p_config->region_xmin;
    host_p_config->region_xmax = p_config->region_xmax;
    host_p_config->region_ymin = p_config->region_ymin;
    host_p_config->region_ymax = p_config->region_ymax;
    // data
    host_p_config->max_obj_num = p_config->max_obj_num;
    host_p_config->round_num = p_config->round_num;
    host_p_config->query_skip_round_num = p_config->query_skip_round_num;
    host_p_config->max_query_num = p_config->max_query_num;
    host_p_config->gaussian_data = p_config->gaussian_data;
    host_p_config->uniform_data = p_config->uniform_data;
    host_p_config->hotspot_num = p_config->hotspot_num;
    //cell
    host_p_config->edge_cell_num = p_config->edge_cell_num;
    //_struct
    host_p_config->max_bucket_len = p_config->max_bucket_len;
    host_p_config->buffer_block_size = p_config->buffer_block_size;
    //thread
    host_p_config->block_analysis_num = p_config->block_analysis_num;
    host_p_config->thread_analysis_num = p_config->thread_analysis_num;
    host_p_config->block_update_num = p_config->block_update_num;
    host_p_config->thread_update_num = p_config->thread_update_num;
    host_p_config->block_query_num = p_config->block_query_num;
    host_p_config->thread_query_num = p_config->thread_query_num;
    host_p_config->block_busy_num = p_config->block_busy_num;
    host_p_config->thread_busy_num = p_config->thread_busy_num;
    //scheme
    host_p_config->single_stream_mutual_access = p_config->single_stream_mutual_access;
    host_p_config->single_stream_para_access = p_config->single_stream_para_access;
    //int multi_stream_parall_access;
    //int single_stream_parall_access_asynch;
    //int update_query_same_thread;
    //host_p_config->open_correct_test = p_config->open_correct_test;

    //other struct
    host_p_config->len_seg_cache_update = p_config->len_seg_cache_update;
    host_p_config->side_len_vgroup = p_config->side_len_vgroup;
    host_p_config->len_seg_cache_query = p_config->len_seg_cache_query;

    host_p_config->len_seg_multiqueue = p_config->len_seg_multiqueue;
    host_p_config->len_multiqueue = p_config->len_multiqueue;
    host_p_config->qt_size = p_config->qt_size;

    host_p_config->len_arr_bkts_max = p_config->len_arr_bkts_max;
    host_p_config->clockRate = get_GPU_Rate();

    host_p_config->terminalFlag = 0;
    host_p_config->buffer_update_round = p_config->buffer_update_round;

    cout << "Status of host_p_config: " << (host_p_config->max_obj_num == 0 ? "No" : "Yes") << endl << endl;
}


