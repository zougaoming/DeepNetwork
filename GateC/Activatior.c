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
    int n = m->dshape.shape[0];
    int c = m->dshape.shape[1];
    int h = m->dshape.shape[2];
    int w = m->dshape.shape[3];
    double sum;
    for(int i = 0;i<n;i++)
    {
        sum = getMatrixSumbyDim(m,3,i,0);
        for(int j = 0;j<c;j++)
        {
            for(int k = 0;k<h;k++)
            {
                for(int l = 0;l<w;l++)
                {
                    
                }
            }
        }
    }
}
void SoftmaxActivator_Backward(Matrix *dz,Matrix *output)
{
    
    doWise(output,1,Wise_TanhBackward,NULL);
    dotSecondOrderMatrixs2(dz, output);
}



