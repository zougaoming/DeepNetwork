//
//  ReluActivatior.c
//  GateC
//
//  Created by Steven on 22/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//
#include <math.h>
#include "Activatior.h"
void ReluActivator_Forward(Matrix *m)
{
    getMaximumMatrix(m,0);
}
void ReluActivator_Backward(Matrix *dz,Matrix *output)
{
    for(int i = 0;i<dz->length;i++)
    {
        if(*(output->array + i) <= .0)
            *(dz->array + i) = (double).0;
    }
}
void IdentityActivator_Forward(Matrix *m)
{
    ;
}
void IdentityActivator_Backward(Matrix *dz,Matrix *output)
{
    ;
}

void SigmoidActivator_Forward(Matrix *m)
{
    doWise(m,1,Wise_Sign,Wise_exp,Wise_Add,Wise_todiv,NULL);
}
void SigmoidActivator_Backward(Matrix *dz,Matrix *output)
{
    doWise(output,1,Wise_SigmoidBackward,NULL);
    dotSecondOrderMatrixs2(dz, output);
}
void TanhActivator_Forward(Matrix *m)
{
    doWise(m,1,Wise_TanhForward,NULL);
}
void TanhActivator_Backward(Matrix *dz,Matrix *output)
{
    doWise(output,1,Wise_TanhBackward,NULL);
    dotSecondOrderMatrixs2(dz, output);
}
void SoftmaxActivator_Forward(Matrix *m)
{
    int z = m->dshape.shape[3];
    int y = m->dshape.shape[2] * z ;
    int x = m->dshape.shape[1] * y;
    int index = 0;
    double sum=0,max=0,tmp;
    for(int i = 0;i<m->dshape.shape[0];i++)
    {
        max = getMatrixMax(m,i);
        sum = 0;
        index = i * x;
        for(int j = 0;j<x;j++)
        {
            tmp = exp(*(m->array + index + j) - max);
            *(m->array + i * x + j) = tmp;
            sum += tmp;
        }
        for(int j = 0;j<x;j++)
        {
            *(m->array + index + j) /= sum;
        }
    }
}
void SoftmaxActivator_Backward(Matrix *dz,Matrix *output)
{
    
    doWise(output,1,Wise_TanhBackward,NULL);
    dotSecondOrderMatrixs2(dz, output);
}



