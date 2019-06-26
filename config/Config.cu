/************************************************************
* Config.cu
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 9, 2014
* Description:
* Licence:
************************************************************/

#include <iostream>
//#include <string>
#include <stdio.h>
#include <stdlib.h>
//#include <stdlib.h>
#include "Config.h"

#define TIXML_USE_STL

#include "./tinyxml/tinyxml.h"
#include "./tinyxml/tinystr.h"

using namespace std;

Config *Config::p_config = NULL;
char config_file[128];

Config::Config(void) {
    // config.xml file in the same dir

    sprintf(config_file, "config.xml");
    FILE *fpc = fopen(config_file, "r");
    if (!fpc) {
        cout << "Config file not found: config.xml " << endl;
        exit(1);
    } else {
        fclose(fpc);
    }
    readSettings();
}

void Config::writeSettings(void) {
    // To be continued ...
//	cout << "---- Writing Config Data ----" << endl;
//	TiXmlDocument *doc = new TiXmlDocument();
//	TiXmlElement *root = new TiXmlElement("config");
//	doc->LinkEndChild(root);
//
//	TiXmlElement *period = new TiXmlElement("period");
//	root->LinkEndChild(period);
//	TiXmlElement *periodLen = new TiXmlElement("period_len");

}

void Config::readSettings(void) {

    cout << "---- Reading Config Data ----" << endl;

    TiXmlDocument *doc = new TiXmlDocument();
    doc->LoadFile(config_file);
    // [root config]
    TiXmlElement *rootElement = doc->RootElement();

    // [period]
    TiXmlElement *token = rootElement->FirstChildElement();
    TiXmlElement *period = token;
    period = period->FirstChildElement();
    period_len = atoi(period->GetText());
    cout << "period_len = " << period_len << endl;

    period = period->NextSiblingElement();
    period_num = atoi(period->GetText());
    cout << "period_num = " << period_num << endl;

    // [region]
    token = token->NextSiblingElement();
    TiXmlElement *region = token;
    region = region->FirstChildElement();
    region_xmin = atoi(region->GetText());
    cout << endl << "region_xmin = " << region_xmin << endl;

    region = region->NextSiblingElement();
    region_xmax = atoi(region->GetText());
    cout << "region_xmax = " << region_xmax << endl;

    region = region->NextSiblingElement();
    region_ymin = atoi(region->GetText());
    cout << "region_ymin = " << region_ymin << endl;

    region = region->NextSiblingElement();
    region_ymax = atoi(region->GetText());
    cout << "region_ymax = " << region_ymax << endl;

    // [data]
    token = token->NextSiblingElement();
    TiXmlElement *data = token;
    data = data->FirstChildElement();
    max_obj_num = atoi(data->GetText());
    cout << endl << "max_obj_num = " << max_obj_num << endl;

    data = data->NextSiblingElement();
    round_num = atoi(data->GetText());
    cout << "round_num = " << round_num << endl;

    data = data->NextSiblingElement();
    query_skip_round_num = atoi(data->GetText());
    cout << "query_skip_round_num = " << query_skip_round_num << endl;

    data = data->NextSiblingElement();
    max_query_num = atoi(data->GetText());
    cout << "max_query_num = " << max_query_num << endl;

    data = data->NextSiblingElement();
    gaussian_data = atoi(data->GetText());
    cout << "gaussian_data = " << gaussian_data << endl;

    data = data->NextSiblingElement();
    uniform_data = atoi(data->GetText());
    cout << "uniform_data = " << uniform_data << endl;

    data = data->NextSiblingElement();
    hotspot_num = atoi(data->GetText());
    cout << "hotspot_num = " << hotspot_num << endl;

    // [cell]  moved to last
//	token = token->NextSiblingElement();
//	TiXmlElement *cell = token;
//	cell = cell->FirstChildElement();
//	edge_cell_num = atoi(cell->GetText());
//	cout << endl << "edge_cell_num = " << edge_cell_num << endl;

    // [_struct]
    token = token->NextSiblingElement();
    TiXmlElement *_struct = token;
    _struct = _struct->FirstChildElement();
    max_bucket_len = atoi(_struct->GetText());
    cout << endl << "max_bucket_len = " << max_bucket_len << endl;

    _struct = _struct->NextSiblingElement();
    buffer_block_size = atoi(_struct->GetText());
    cout << "buffer_block_size = " << buffer_block_size << endl;

    // [thread]
    token = token->NextSiblingElement();
    TiXmlElement *thread = token;
    thread = thread->FirstChildElement();
    block_analysis_num = atoi(thread->GetText());
    cout << endl << "block_analysis_num = " << \
            block_analysis_num << endl;

    thread = thread->NextSiblingElement();
    thread_analysis_num = atoi(thread->GetText());
    cout << "thread_analysis_num = " << thread_analysis_num << endl;

    thread = thread->NextSiblingElement();
    block_update_num = atoi(thread->GetText());
    cout << "block_update_num = " << block_update_num << endl;

    thread = thread->NextSiblingElement();
    thread_update_num = atoi(thread->GetText());
    cout << "thread_update_num = " << thread_update_num << endl;

    thread = thread->NextSiblingElement();
    block_query_num = atoi(thread->GetText());
    cout << "block_query_num = " << block_query_num << endl;

    thread = thread->NextSiblingElement();
    thread_query_num = atoi(thread->GetText());
    cout << "thread_query_num = " << thread_query_num << endl;

    thread = thread->NextSiblingElement();
    block_busy_num = atoi(thread->GetText());
    cout << "block_busy_num = " << block_busy_num << endl;

    thread = thread->NextSiblingElement();
    thread_busy_num = atoi(thread->GetText());
    cout << "thread_busy_num = " << thread_busy_num << endl;

    // [scheme]
    token = token->NextSiblingElement();
    TiXmlElement *scheme = token;
    scheme = scheme->FirstChildElement();
    single_stream_mutual_access = atoi(scheme->GetText());
    cout << endl << "single_stream_mutual_access = " << \
            single_stream_mutual_access << endl;

    scheme = scheme->NextSiblingElement();
    single_stream_para_access = atoi(scheme->GetText());
    cout << "single_stream_para_access = " << single_stream_para_access \
 << endl;

    scheme = scheme->NextSiblingElement();
    open_correct_test = atoi(scheme->GetText());
    cout << "open_correct_test = " << open_correct_test << endl;


    // [parm]
    token = token->NextSiblingElement();
    TiXmlElement *params = token;
    params = params->FirstChildElement();
    //queryDispatch multiqueue
    len_seg_multiqueue = atoi(params->GetText());
    cout << endl << "len_seg_multiqueue = " << len_seg_multiqueue << endl;

    params = params->NextSiblingElement();
    len_multiqueue = atoi(params->GetText());
    cout << "len_multiqueue = " << len_multiqueue << endl;

    params = params->NextSiblingElement();
    qt_size = atoi(params->GetText());
    cout << "qt_size = " << qt_size << endl;

    params = params->NextSiblingElement();
    len_seg_cache_update = atoi(params->GetText());
    cout << "len_seg_cache_update = " << \
            len_seg_cache_update << endl;

    params = params->NextSiblingElement();
    side_len_vgroup = atoi(params->GetText());
    cout << "side_len_vgroup = " << side_len_vgroup << endl;

    params = params->NextSiblingElement();
    edge_cell_num = atoi(params->GetText());
    cout << "edge_cell_num = " << edge_cell_num << endl;

    params = params->NextSiblingElement();
    len_seg_cache_query = atoi(params->GetText());
    cout << "len_seg_cache_query = " << len_seg_cache_query << endl;

    len_arr_bkts_max = (int) ((128.0f / (double) edge_cell_num) * 512);

    buffer_update_round = 2;

    cout << endl << "========= Bottom of the list =========" << endl;
}


Config *Config::getInstance(void) {
    if (p_config == NULL) {
        p_config = new Config;
    }
    return p_config;
}



