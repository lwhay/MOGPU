/************************************************************
* Buffer.h
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 11, 2014*  4:46:58 PM* 
* Description:*  
* Licence:*
************************************************************/

#ifndef BUFFER_H_
#define BUFFER_H_

#include <stdio.h>
#include "config/Config.h"
#include "BaseStruct.h"
#include "config/GConfig.h"

template<class DataType>
class Buffer
{
public:
    DataType *req_buffer;
    int buffer_len;
    int current_pos;
//    pthread_spinlock_t spin_lock;
    DataType *tmp;

public:
    Buffer(void)
    {
        req_buffer = NULL;
        buffer_len = 0;
        current_pos = 0;
        tmp = NULL;
    }
    ~Buffer()
    {
        if(req_buffer){
            delete []req_buffer;
        }
    }

    void initBuffer()
    {
        Config *p_config = Config::getInstance();
        char fpath[50];
        char fname[20];
        if(sizeof(DataType) == sizeof(UpdateType))
        {
            //更新数据获取
            buffer_len = p_config->max_obj_num * p_config->round_num;
            sprintf(fname, "update.dat");
        }
        else if(sizeof(DataType) == sizeof(QueryType))
        {
             //查询数据获取
             buffer_len = p_config->max_query_num;
             sprintf(fname, "query.dat");
        }

        if(p_config->gaussian_data == 1)
            sprintf(fpath,"..//data//gaussian//%d//%dK//%s", \
            		p_config->hotspot_num,p_config->max_obj_num/1000,fname);
        else if(p_config->uniform_data == 1)
            sprintf(fpath,"..//data//uniform//%dK//%s", \
            		p_config->max_obj_num / 1000,fname);
		else 
			sprintf(fpath,"..//data//roadmap//%dK//%s", \
            		p_config->max_obj_num / 1000,fname);
        //判断是否存在文件
        FILE *fp = fopen(fpath, "rb");
        if(fp && sizeof(DataType) == sizeof(UpdateType))
        {
			req_buffer = new DataType[buffer_len];
			//fread(req_buffer, buffer_len * sizeof(DataType), 1, fp);
			for(int i = 0;i<buffer_len;i++){
                fscanf(fp,"%d %f %f %f %f %f\n",&(((UpdateType*)(&(req_buffer[i])))->oid),&(((UpdateType*)(&(req_buffer[i])))->x),&(((UpdateType*)(&(req_buffer[i])))->y),&(((UpdateType*)(&(req_buffer[i])))->vx),&(((UpdateType*)(&(req_buffer[i])))->vy),&(((UpdateType*)(&(req_buffer[i])))->time));
            }
			fclose(fp);
			fp = NULL;

        }else if(fp && sizeof(DataType) == sizeof(QueryType)){
			req_buffer = new DataType[buffer_len];
			for(int i = 0;i<buffer_len;i++){
				fscanf(fp,"%d %f %f %f %f %f\n",&(((QueryType*)(&(req_buffer[i])))->qid),&(((QueryType*)(&(req_buffer[i])))->minX),&(((QueryType*)(&(req_buffer[i])))->minY),&(((QueryType*)(&(req_buffer[i])))->maxX),&(((QueryType*)(&(req_buffer[i])))->maxY),&(((QueryType*)(&(req_buffer[i])))->t));
			}
			fclose(fp); fp = NULL;
		}
        else
        {
            printf("Data file does not exist!\n");
            exit(404);
        }
    }

    DataType *getData(int block_size=1)
    {

    	DataType *tmp = NULL;
        if(current_pos + block_size <= buffer_len)
        {
            tmp = req_buffer + current_pos;
            current_pos += block_size;
        }

        return tmp;
    }

    DataType *getHostBuffer(void)
    {
    	return req_buffer;
    }

    void clear()
    {
        delete []req_buffer;
        req_buffer = NULL;
    }
};


#endif /* BUFFER_H_ */
