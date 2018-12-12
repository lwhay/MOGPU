/************************************************************
* Rect.h
* Copyright (c) B701, CS, Wuhan University
* Author: Chundan Wei
* Email: danuno@qq.com
* Version: 1.0
* Date: Aug 11, 2014,  2:06:33 PM*
* Description:*
* Licence:
************************************************************/

#ifndef RECT_H_
#define RECT_H_


class Rect
{
public:
	float xmin;
	float ymin;
	float xmax;
	float ymax;
    float height;
    float width;

public:
    Rect(void)
    {
        xmin = ymin = xmax = ymax = 0;
        height = width = 0;
    }

    Rect(float _xmin,float _ymin,float _xmax,float _ymax)
    {
        xmin = _xmin;
        ymin = _ymin;
        xmax = _xmax;
        ymax = _ymax;
        height = ymax - ymin;
        width = xmax - xmin;
    }

    ~Rect(void)
    {
        ;
    }

    void setCoverArea(float _xmin, float _ymin, \
    		float _xmax, float _ymax)
    {
        xmin = _xmin;
        ymin = _ymin;
        xmax = _xmax;
        ymax = _ymax;
        height = ymax - ymin;
        width = xmax - xmin;
    }

    __device__ float getCenterX()
    {
        return (xmin + xmax) / 2;
    }

    __device__ float getCenterY()
    {
        return (ymin + ymax) / 2;
    }

    __device__ bool isCovering(float x, float y)
    {
        if (x >= xmin && x <= xmax && y <= ymax && y >= ymin)
            return true;
        else
            return false;
    }
};


#endif /* RECT_H_ */
