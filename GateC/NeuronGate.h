//
//  NeuronGate.h
//  GateC
//
//  Created by Steven on 1/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef NeuronGate_h
#define NeuronGate_h
#include "Matrix.h"
#include "Gate.h"
#include <stdio.h>
typedef struct NeuronGateParamS{
    Matrix *input;
    Matrix *_output;
    Matrix *dz;
    Matrix * weight;
    Matrix * bias;
    Matrix * dx;
    Matrix * dbias;
    Matrix * dw;
    Activator_Forward forward;
    Activator_Backward backward;
}NeuronGateParam;
//void CnnGate(CnnGateParam *p);
void Neuron_Forward(NeuronGateParam *p);
void Neuron_Backward(NeuronGateParam *p);
#endif /* NeuronGate_h */
