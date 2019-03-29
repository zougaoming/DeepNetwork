//
//  CnnGate.c
//  GateC
//
//  Created by Steven on 22/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "CnnGate.h"
void Conv(Matrix *input,Matrix *weight,Matrix *i_a,Matrix* result,int outputH,int outputW,int strids)
{
    
    for(int h=0;h<outputH;h++)
    {
        for(int w=0;w<outputW;w++)
        {
            getSecondOrderSubMatrix2(input,i_a,h * strids,w*strids);
            dotSecondOrderMatrixs2(i_a,weight);
            double tmp = getMatrixSum(i_a);
            //printf("C h=%d,w=%d,sum=%f\n",h,w,tmp);
            modifyMatrixElem(result, 0, 0, h, w, tmp);
        }
    }
}

void Backward(CnnGateParam *p)
{

    ReluActivator_Backward(p->dz,p->_output);
    //printarray(p->dz);
    p->dw = zeros_like(p->weight);
    p->dbias = zeros_like(p->bias);
    p->dx = zeros_like(p->input);
    
    int channel_out = p->weight->dshape.shape[0];
    int channel_in = p->weight->dshape.shape[1];
    int N = p->input->dshape.shape[0];
    int fliter = p->weight->dshape.shape[2];
    int zp = fliter - p->panding -1;
    Matrix *padded_delta = copyMatrix(p->dz);
    PandingMatrix4D(padded_delta,zp);
    int inputH = p->input->dshape.shape[2];
    int inputW = p->input->dshape.shape[3];
    int outputW = getOutputSize(inputW,p->strids,fliter,p->panding);
    int outputH = getOutputSize(inputH,p->strids,fliter,p->panding);
    
    Dshape panded_shape;
    initDshapeInt(&panded_shape, 0, 0, padded_delta->dshape.shape[2], padded_delta->dshape.shape[3]);
    Matrix *panded_convw = creatMatrixFromValue(0, panded_shape);
    
    Dshape out_tmpShape;
    initDshapeInt(&out_tmpShape,0,0,outputH,outputW);
    Matrix *out_tmp = creatZerosMatrix(out_tmpShape);
    Matrix *out_tmp2 = copyMatrix(out_tmp);
    Dshape input2Dshape;
    initDshapeInt(&input2Dshape, 0, 0,inputH,inputW);
    Matrix *input2D = creatMatrixFromValue(0, input2Dshape);
    Matrix *result2 = creatMatrixFromValue(0, input2Dshape);
    Dshape flitershape;
    initDshapeInt(&flitershape, 0, 0,fliter,fliter);
    Matrix *w_a = creatMatrixFromValue(0, flitershape);
    Matrix *w_a2 = copyMatrix(w_a);
    Matrix *result = creatMatrixFromValue(0, flitershape);
    for(int n=0;n<N;n++)
    {
        for(int inc = 0;inc < channel_in;inc++)
        {
            for(int outc = 0;outc < channel_out;outc++)
            {
                get2dim(p->input,input2D,n,inc);
                get2dim(p->dz,out_tmp,n,outc);
                Conv(input2D,out_tmp,out_tmp2,result,fliter,fliter,p->strids);
                addSecondOrderMatrixsby2d(p->dw,result,outc,inc);

                get2dim(p->weight,w_a,outc,inc);
                rot90Matrix(w_a,2);
                get2dim(padded_delta,panded_convw,n,outc);//out_tmp  的大小，应该加上PANDING
                Conv(panded_convw, w_a,w_a2,result2, inputH,inputW,p->strids);
                addSecondOrderMatrixsby2d(p->dx,result2,n,inc);

            }
        }
        for(int outc = 0;outc < channel_out;outc++)
        {
            double dbias;
            getMatrixElem(p->dbias, 0, 0, 0, outc, &dbias);
            dbias += getMatrixSumbyDim(p->dz,2,n,outc);
            modifyMatrixElem(p->dbias, 0, 0, 0, outc, dbias);
        }
    }
    destroyMatrix(panded_convw);
    destroyMatrix(padded_delta);
    destroyMatrix(out_tmp);
    destroyMatrix(out_tmp2);
    destroyMatrix(input2D);
    destroyMatrix(result);
    destroyMatrix(result2);
    destroyMatrix(w_a);
    destroyMatrix(w_a2);
}


void Forward(CnnGateParam *p)
{
    
    int channel_out = p->weight->dshape.shape[0];
    int channel_in = p->weight->dshape.shape[1];
    int N = p->input->dshape.shape[0];
    int fliter = p->weight->dshape.shape[2];
    int inputH = p->input->dshape.shape[2];
    int inputW = p->input->dshape.shape[3];
    int outputW = getOutputSize(inputW,p->strids,fliter,p->panding);
    int outputH = getOutputSize(inputH,p->strids,fliter,p->panding);
    PandingMatrix4D(p->input,p->panding);
    Dshape outputShape;
    initDshapeInt(&outputShape,N,channel_out,outputH,outputW);
    Matrix * _output = creatZerosMatrix(outputShape);
    
    Dshape out_tmpShape;
    initDshapeInt(&out_tmpShape,0,0,outputH,outputW);
    Matrix *out_tmp = creatZerosMatrix(out_tmpShape);
    Matrix *result = copyMatrix(out_tmp);
    Dshape input2Dshape;
    initDshapeInt(&input2Dshape, 0, 0,inputH + 2*p->panding,inputW + 2*p->panding);
    Matrix *input2D = creatMatrixFromValue(0, input2Dshape);
    Dshape flitershape;
    initDshapeInt(&flitershape, 0, 0,fliter,fliter);
    Matrix *w_a = creatMatrixFromValue(0, flitershape);
    Matrix *i_a = copyMatrix(w_a);
    for(int n = 0;n< N;n++)
    {
        for(int c = 0;c < channel_out;c++)
        {
            setZeros(out_tmp);
            for(int inc=0;inc < channel_in;inc++)
            {
                get2dim(p->input,input2D,n,inc);
                get2dim(p->weight,w_a,c,inc);
                Conv(input2D, w_a,i_a,result, outputH,outputW,p->strids);
                addSecondOrderMatrixs2(out_tmp,result);
            }
            double b;
            getMatrixElem(p->bias, 0, 0, 0, c, &b);
            kAddMatrix(out_tmp, b);
            setMatrixArray(_output,out_tmp,n,c);
        }
    }
    ReluActivator_Forward(_output);
    p->_output = _output;
    printf("%f\n",get_mempool_usage());
    destroyMatrix(i_a);
    destroyMatrix(w_a);
    destroyMatrix(input2D);
    destroyMatrix(out_tmp);
    destroyMatrix(result);
    //MemoryPool_Clear();
    printf("%f\n",get_mempool_usage());
    return ;
    
    
}
/*
void CnnGate(CnnGateParam *p)
{
    if(p->isBackward == 1)
    {
        Backward(p);
    }
    else
    {
        //PoolGate *newp;
        Forward(p);
    }
}
*/