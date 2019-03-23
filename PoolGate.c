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
    getMatrixElem(input,0,0,0,0,&max);
    for(int i=0;i<input->dshape.shape[2];i++)
    {
        for(int j = 0;j < input->dshape.shape[3];j++)
        {
            getMatrixElem(input,0,0,i,j,&tmp);
            if(tmp > max)
            {
                max = tmp;
                modifyMatrixElem(bz_x,0,f,c,index,i);
                modifyMatrixElem(bz_y,0,f,c,index,j);
            }
        }
    }
    return max;
}

Matrix* PoolGate_Backward(PoolGateParam *p)
{
    Matrix* result = zeros_like(p->input);
    int index = 0;
    for(int f =0;f<p->fNum;f++)
    {
        for(int c =0;c<p->channel_in;c++)
        {
            index = 0;
            for(int h = 0;h < p->outputH;h++)
            {
                for(int w = 0;w < p->outputW;w++)
                {
                    double tmp;
                    getMatrixElem(p->bz_x,0,f,c,index,&tmp);
                    int xh = h * p->strids + (int)tmp;
                    getMatrixElem(p->bz_y,0,f,c,index,&tmp);
                    int xw = w * p->strids + (int)tmp;
                
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
    p->fNum = p->input->dshape.shape[0];
    p->channel_in = p->input->dshape.shape[1];
    if(p->fNum == 0)p->fNum = 1;
    if(p->channel_in == 0)p->channel_in = 1;
    int inputW = p->input->dshape.shape[3];
    int inputH = p->input->dshape.shape[2];
    //printShape(p->input);
    
    p->outputW = getOutputSize(inputW,step,fliters,0);
    p->outputH = getOutputSize(inputH,step,fliters,0);
    printf("C N->%d,C=%d,outputW->%d,outputH->%d",p->fNum,p->channel_in, p->outputW,p->outputH);
    Dshape outputShape;
    initDshapeInt(&outputShape,p->fNum,p->channel_in,p->outputH,p->outputW);
    p->_output = creatZerosMatrix(outputShape);
    
    Dshape bzshape;
    initDshapeInt(&bzshape,0,p->fNum,p->channel_in,p->outputW * p->outputH);
    p->bz_x =creatZerosMatrix(bzshape);
    p->bz_y = zeros_like(p->bz_x);
    Dshape input2Dshape;
    initDshapeInt(&input2Dshape, 0, 0,inputH,inputW);
    Matrix *input2D = creatMatrixFromValue(0, input2Dshape);
    Dshape flitershape;
    initDshapeInt(&flitershape, 0, 0,fliters,fliters);
    Matrix *i_a = creatMatrixFromValue(0, flitershape);
    int index = 0;
    for(int f =0;f<p->fNum;f++)
    {
        //printf("f=%d\n",f);
        for(int c =0;c<p->channel_in;c++)
        {
            get2dim(p->input,input2D,f,c);
            //printarray(input2D);
            index = 0;
            for(int h = 0;h < p->outputH;h++)
            {
                for(int w = 0;w < p->outputW;w++)
                {
                    //printf("f=%d,c=%d,h=%d,w=%d,index=%d\n,",f,c,h,w,index);
                    int starth = h * step;
                    int startw = w * step;
                    
                    getSecondOrderSubMatrix2(input2D,i_a,starth,startw);
                    //printarray(i_a);
                    if(!i_a)printf("i_a is NULL!");
                    double res = calc_pool(i_a,'M',index,f,c,p->bz_x,p->bz_y);
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

Matrix* PoolGate(PoolGateParam *p)
{
    if(p->isBackward == 1)
        return PoolGate_Backward(p);
    else
    {
        //PoolGate *newp;
        return PoolGate_Forward(p);
    }
}