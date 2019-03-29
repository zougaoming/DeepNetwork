//
//  AddGate.c
//  GateC
//
//  Created by Steven on 21/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "AddGate.h"
void AddGate_Forward(AddGateParam *p)
{
    if(compareMatrix_Shape(p->input1,p->input2) == 0){printf("m1.ndim != m2.ndim");return ;}
    p->dim = getMatrixNdim(p->input1);
    p->N = p->input1->dshape.shape[0];
    p->C = p->input1->dshape.shape[1];
    p->_output = addSecondOrderMatrixs(p->input1,p->input2);
    
}
void AddGate_Backward(AddGateParam *p)
{
    p->_output = p->dz;
}
void AddGate(AddGateParam *p)
{
    if(p->isBackward == 1)
        AddGate_Backward(p);
    else
    {
        //PoolGate *newp;
        AddGate_Forward(p);
    }
   }