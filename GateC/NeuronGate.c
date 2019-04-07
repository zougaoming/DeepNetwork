//
//  NeuronGate.c
//  GateC
//
//  Created by Steven on 1/4/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "NeuronGate.h"
void Neuron_Forward(NeuronGateParam *p){
    transposeSecondOrderMatrix(p->weight);
    Matrix* m = mulSecondOrderMatrixs(p->weight,p->input);
    for(int i = 0;i< m->dshape.shape[2];i++)
    {
        for(int j = 0;j< m->dshape.shape[3];j++)
        {
            *(m->array + i * m->dshape.shape[3] + j) += *(p->bias->array + i * p->bias->dshape.shape[3]);
        }
    }
    p->_output = copyMatrix(m);
    p->forward(p->_output);
    destroyMatrix(m);
    
}
void Neuron_Backward(NeuronGateParam *p){
    p->dw = zeros_like(p->weight);
    p->dbias = zeros_like(p->bias);
    p->dx = zeros_like(p->input);
    
    p->backward(p->dz,p->_output);
    transposeSecondOrderMatrix(p->input);
    p->dw = mulSecondOrderMatrixs(p->dz,p->input);
    transposeSecondOrderMatrix(p->dw);

    p->dx = mulSecondOrderMatrixs(p->weight,p->dz);
    double sum = 0;
    for(int i = 0;i<p->dz->dshape.shape[2];i++)
    {
        sum = 0;
        for(int j = 0;j<p->dz->dshape.shape[3];j++)
        {
            sum += *(p->dz->array + i * p->dz->dshape.shape[3] + j);
        }
        *(p->dbias->array + i * p->dbias->dshape.shape[3]) = sum;
    }
    //printShape(p->dbias);
    //p->dbias = p->dz;//modifyMatrixElem(p->dbias, 0, 0, 0, 1, getMatrixSum(p->dz));
}