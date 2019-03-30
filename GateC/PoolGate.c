//
//  PoolGate.c
//  GateC
//
//  Created by Steven on 20/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "PoolGate.h"



double calc_pool(Matrix *input,char * type,int index,int f,int c,Matrix * bz_x,Matrix * bz_y)
{
    double tmp = 0.0;
    double max = 0.0;
    //getMatrixElem(input,0,0,0,0,&max);
    max = *(input->array);
    int y = input->dshape.shape[2];
    int z = input->dshape.shape[3];
    
    int bz = bz_x->dshape.shape[3];
    int by = bz_x->dshape.shape[2] * bz;
    //int bx = bz_x->dshape.shape[1] * by;
    int index2 = f * by + c * bz + index;
    for(int i=0;i< y;i++)
    {
        for(int j = 0;j < z;j++)
        {
            tmp = *(input->array + i * z  + j);
            //getMatrixElem(input,0,0,i,j,&tmp);
            if(tmp > max)
            {
                max = tmp;
                *(bz_x->array + index2) = i;
                *(bz_y->array + index2) = y;
                //modifyMatrixElem(bz_x,0,f,c,index,i);
                //modifyMatrixElem(bz_y,0,f,c,index,j);
            }
        }
    }
    return max;
}

Matrix* PoolGate_Backward(PoolGateParam *p)
{
    int step = p->strids;
    int fliters = p->filter;
    Matrix* result = zeros_like(p->input);
    int N = p->input->dshape.shape[0];
    int channel = p->input->dshape.shape[1];
    if(N == 0)N = 1;
    if(channel == 0)channel = 1;
    int inputW = p->input->dshape.shape[3];
    int inputH = p->input->dshape.shape[2];
    //printShape(p->input);
    
    int outputW = getOutputSize(inputW,step,fliters,0);
    int outputH = getOutputSize(inputH,step,fliters,0);
    int index = 0;
    int indexbz = 0;
    int bz = p->bz_x->dshape.shape[3];
    int by = p->bz_x->dshape.shape[2] * bz;
    int bx = p->bz_x->dshape.shape[1] * by;
    for(int f =0;f<N;f++)
    {
        for(int c =0;c<channel;c++)
        {
            index = 0;
            indexbz = f * by + c * bz;
            for(int h = 0;h < outputH;h++)
            {
                for(int w = 0;w < outputW;w++)
                {
                    
                    double tmp = *(p->bz_x->array + indexbz + index);
                    //getMatrixElem(p->bz_x,0,f,c,index,&tmp);
                    int xh = h * p->strids + (int)tmp;
                    tmp = *(p->bz_y->array + indexbz + index);
                    //getMatrixElem(p->bz_y,0,f,c,index,&tmp);
                    int xw = w * p->strids + (int)tmp;
                
                    //*(result->array + )
                    getMatrixElem(p->dz,f,c,h,w,&tmp);
                    modifyMatrixElem(result,f,c,xh,xw,tmp);
                    index ++;
                }
            }
        }
    }
    
    return result;

}

Matrix* PoolGate_Forward(PoolGateParam *p)
{
    int step = p->strids;
    int fliters = p->filter;
    int N = p->input->dshape.shape[0];
    int channel = p->input->dshape.shape[1];
    if(N == 0)N = 1;
    if(channel == 0)channel = 1;
    int inputW = p->input->dshape.shape[3];
    int inputH = p->input->dshape.shape[2];
    //printShape(p->input);
    
    int outputW = getOutputSize(inputW,step,fliters,0);
    int outputH = getOutputSize(inputH,step,fliters,0);
    //printf("C N->%d,C=%d,outputW->%d,outputH->%d",p->fNum,p->channel_in, p->outputW,p->outputH);
    Dshape outputShape;
    initDshapeInt(&outputShape,N,channel,outputH,outputW);
    p->_output = creatZerosMatrix(outputShape);
    
    Dshape bzshape;
    initDshapeInt(&bzshape,0,N,channel,outputW * outputH);
    p->bz_x =creatZerosMatrix(bzshape);
    p->bz_y = zeros_like(p->bz_x);
    Dshape input2Dshape;
    initDshapeInt(&input2Dshape, 0, 0,inputH,inputW);
    Matrix *input2D = creatMatrixFromValue(0, input2Dshape);
    Dshape flitershape;
    initDshapeInt(&flitershape, 0, 0,fliters,fliters);
    Matrix *i_a = creatMatrixFromValue(0, flitershape);
    int index = 0;
    for(int f =0;f<N;f++)
    {
        //printf("f=%d\n",f);
        for(int c =0;c<channel;c++)
        {
            get2dim(p->input,input2D,f,c);
            //printarray(input2D);
            index = 0;
            for(int h = 0;h < outputH;h++)
            {
                for(int w = 0;w < outputW;w++)
                {
                    //printf("f=%d,c=%d,h=%d,w=%d,index=%d\n,",f,c,h,w,index);
                    int starth = h * step;
                    int startw = w * step;
                    
                    getSecondOrderSubMatrix2(input2D,i_a,starth,startw);
                    //printarray(i_a);
                    if(!i_a)printf("i_a is NULL!");
                    double res = calc_pool(i_a,p->type,index,f,c,p->bz_x,p->bz_y);
                    modifyMatrixElem(p->_output,f,c,h,w,res);
                    index ++;
                }
            }
        }
    }
    //printShape(p->_output);
    destroyMatrix(input2D);
    destroyMatrix(i_a);
    return p->_output;
}