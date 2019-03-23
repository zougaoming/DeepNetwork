//
//  CnnGate.h
//  GateC
//
//  Created by Steven on 22/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef CnnGate_h
#define CnnGate_h
#include <stdio.h>
#include "Matrix.h"
#include "Gate.h"
typedef struct CnnGateParamS{
    Matrix *input;
     int filter;
     int strids;
     int panding;
    Matrix *_output;
    Matrix *dz;
    Matrix * weight;
    Matrix * bias;
    Matrix * dx;
    Matrix * dbias;
    Matrix * dw;
     int outputH;
     int outputW;
     int N;
     int channel_in;
     int channel_out;
    
     int isBackward;
    ParamLink * link;
}CnnGateParam;
void CnnGate(CnnGateParam *p);
#endif /* CnnGate_h */
