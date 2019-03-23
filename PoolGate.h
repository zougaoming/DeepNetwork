//
//  PoolGate.h
//  GateC
//
//  Created by Steven on 20/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef PoolGate_h
#define PoolGate_h

#include <stdio.h>
#include "Matrix.h"
#include "Gate.h"
typedef struct PoolGateParamS{
    Matrix *input;
    unsigned int filter;
    unsigned int strids;
    Matrix *_output;
    Matrix *dz;
    Matrix *bz_x;
    Matrix *bz_y;
    unsigned int outputH;
    unsigned int outputW;
    unsigned int fNum;
    unsigned int channel_in;
    unsigned int isBackward;
    ParamLink * link;
}PoolGateParam;
//PoolGateParam p
Matrix* PoolGate(PoolGateParam *p);
#endif /* PoolGate_h */
