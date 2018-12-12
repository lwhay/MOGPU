/************************************************************
* main.cpp / main.cu
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 9, 2014,  9:28:15 AM
* Description:
* Licence:
************************************************************/

#include <iostream>
#include "config/Config.h"
#include "misc/GPGrid.h"

using namespace std;

int main(void) {
    cout << "==== SoloGPU! ====" << endl;

    cout << "\n==== Config Parameters ====" << endl;
    Config *config_ins = Config::getInstance();

    cout << "\n=== Launch GPGrid... ===" << endl;

    launchGPGrid();

    return 0;
}
