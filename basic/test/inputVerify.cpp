//
// Created by lwh on 19-6-22.
//

#include <iostream>
#include <string>
#include <unordered_map>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include "BaseStruct.h"

using namespace std;

size_t size = 10000000;

void verifyInit(char *path) {
    FILE *fp = fopen(path, "rt");
    unordered_map<int, int> map;
    UpdateType *ut = new UpdateType[size];
    //fread(ut, sizeof(UpdateType), size, fp);
    for (int i = 0; i < size; i++) {
        fscanf(fp, "%d%f%f%f%f%f\n", &ut[i].oid, &ut[i].x, &ut[i].y, &ut[i].vx, &ut[i].vy, &ut[i].time);
        if (map.find(ut[i].oid) == map.end()) {
            map.insert(make_pair(ut[i].oid, 1));
        } else {
            map.find(ut[i].oid)->second++;
        }
        //cout << ut[i].oid << endl;
    }
    int maxid, maxfq = 0;
    int minid, minfq = 1 << 30;
    for (pair<int, int> p : map) {
        if (p.second > maxfq) {
            maxid = p.first;
            maxfq = p.second;
        }
        if (p.second < minfq) {
            minid = p.first;
            minfq = p.second;
        }
    }
    cout << minid << "<->" << minfq << " : " << maxid << "<->" << maxfq << endl;
    fclose(fp);
}

void verifyUpdate(char *path) {
    FILE *fp = fopen(path, "rt");
    unordered_map<int, int> map;
    UpdateType *ut = new UpdateType[size];
    //fread(ut, sizeof(UpdateType), size, fp);
    for (int i = 0; i < size; i++) {
        fscanf(fp, "%d%f%f%f%f%f\n", &ut[i].oid, &ut[i].x, &ut[i].y, &ut[i].vx, &ut[i].vy, &ut[i].time);
        if (map.find(ut[i].oid) == map.end()) {
            map.insert(make_pair(ut[i].oid, 1));
        } else {
            map.find(ut[i].oid)->second++;
        }
        //cout << ut[i].oid << endl;
    }
    int maxid, maxfq = 0;
    int minid, minfq = 1 << 30;
    for (pair<int, int> p : map) {
        if (p.second > maxfq) {
            maxid = p.first;
            maxfq = p.second;
        }
        if (p.second < minfq) {
            minid = p.first;
            minfq = p.second;
        }
    }
    cout << minid << "<->" << minfq << " : " << maxid << "<->" << maxfq << endl;
    fclose(fp);
}

void verifyQuery(char *path) {

}

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "Command path" << endl;
        exit(0);
    }
    string path(argv[1]);
    size = std::atol(argv[2]);
    switch (path.at(path.find_last_of("/") + 1)) {
        case 'i' :
            cout << "Verifying init: " << argv[1] << endl;
            verifyInit(argv[1]);
            break;
        case 'q' :
            cout << "Verifying query " << argv[1] << endl;
            verifyQuery(argv[1]);
            break;
        case 'u' :
            cout << "Verifying update " << argv[1] << endl;
            verifyUpdate(argv[1]);
            break;
        default:
            cout << "Error on verification" << endl;
    }
}