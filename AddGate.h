//
//  AddGate.h
//  GateC
//
//  Created by Steven on 21/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef AddGate_h
#define AddGate_h

#include <stdio.h>
#include "Matrix.h"
#include "Gate.h"
typedef struct AddGateParamS{
    Matrix *input1;
    Matrix *input2;
    unsigned dim;
    unsigned int N;
    unsigned int C;
    Matrix *_output;
    Matrix *dz;
    Matrix *dinput1;
    Matrix *dinput2;
    unsigned int isBackward;
    ParamLink * link;
}AddGateParam;
void AddGate(AddGateParam *p);
#endif /* AddGate_h */
