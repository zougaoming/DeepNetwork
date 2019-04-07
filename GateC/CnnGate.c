//
//  CnnGate.c
//  GateC
//
//  Created by Steven on 22/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "CnnGate.h"
#include  <dlfcn.h>
void Conv(Matrix *input,Matrix *weight,Matrix *i_a,Matrix* result,int outputH,int outputW,int strids)
{
    double tmp;
    int z = i_a->dshape.shape[3];
    int inputz = input->dshape.shape[3];
    setZeros(result);
    for(int h=0;h<outputH;h++)
    {
        for(int w=0;w<outputW;w++)
        {
            int resulti = h * result->dshape.shape[3] + w;
            int starti = w * strids + h * strids * (inputz);
            
            for(int i=0;i<i_a->dshape.shape[2];i++){
                for(int j=0;j<z;j++){
                    int bindex = i * z + j;
                    tmp = *(input->array + starti + i*inputz+j);
                    tmp *= *(weight->array + bindex);
                    *(result->array + resulti) += tmp;
                }
            }
            
            
            //getSecondOrderSubMatrix2(input,i_a,h * strids,w*strids);
            //dotSecondOrderMatrixs2(i_a,weight);
            //double tmp = getMatrixSum(i_a);
            //printf("C h=%d,w=%d,sum=%f\n",h,w,tmp);
            //modifyMatrixElem(result, 0, 0, h, w, tmp);
        }
    }
}

void Conv4D(Matrix *input,Matrix *weight,Matrix *i_a,Matrix* result,int outputH,int outputW,int strids,int index00,int index01,int index10,int index11)
{
    double tmp;
    int input_lx,input_ly,input_lz;
    input_lz = input->dshape.shape[3];
    input_ly = input->dshape.shape[2] * input_lz;
    input_lx = input->dshape.shape[1] * input_ly;
    int input_starti = index00 * input_lx + index01 * input_ly;
    
    
    int z = weight->dshape.shape[3];
    int y = weight->dshape.shape[2] * z;
    int x = weight->dshape.shape[1] * y;
    int weight_starti = index10 * x + index11 * y;
    setZeros(result);
    for(int h=0;h<outputH;h++)
    {
        for(int w=0;w<outputW;w++)
        {
            int resulti = h * result->dshape.shape[3] + w;
            int starti = w * strids + h * strids * input_lz + input_starti;
            
            for(int i=0;i<weight->dshape.shape[2];i++){
                for(int j=0;j<z;j++){
                    int bindex = i * z + j + weight_starti;
                    tmp = *(input->array + starti + i*input_lz+j);
                    tmp *= *(weight->array + bindex);
                    *(result->array + resulti) += tmp;
                }
            }
            
            
            //getSecondOrderSubMatrix2(input,i_a,h * strids,w*strids);
            //dotSecondOrderMatrixs2(i_a,weight);
            //double tmp = getMatrixSum(i_a);
            //printf("C h=%d,w=%d,sum=%f\n",h,w,tmp);
            //modifyMatrixElem(result, 0, 0, h, w, tmp);
        }
    }
}

void Backward(CnnGateParam *p)
{

    p->backward(p->dz,p->_output);
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
    

    Dshape out_tmpShape;
    initDshapeInt(&out_tmpShape,0,0,outputH,outputW);
    //Matrix *out_tmp = creatZerosMatrix(out_tmpShape);
    Matrix *out_tmp2 = creatZerosMatrix(out_tmpShape);
    Dshape input2Dshape;
    initDshapeInt(&input2Dshape, 0, 0,inputH,inputW);
    //Matrix *input2D = creatMatrixFromValue(0, input2Dshape);
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

                Conv4D(p->input, p->dz,out_tmp2,result,fliter,fliter,p->strids,n,inc,n,outc);
                addSecondOrderMatrixsby2d(p->dw,result,outc,inc);

                get2dim(p->weight,w_a,outc,inc);
                rot90Matrix(w_a,2);
                Conv4D(padded_delta, w_a,w_a2,result2, inputH,inputW,p->strids,n,outc,0,0);
                addSecondOrderMatrixsby2d(p->dx,result2,n,inc);

            }
        }
        for(int outc = 0;outc < channel_out;outc++)
        {
            double dbias;
            dbias = getMatrixElem(p->dbias, 0, 0, 0, outc);
            dbias += getMatrixSumbyDim(p->dz,2,n,outc);
            modifyMatrixElem(p->dbias, 0, 0, 0, outc, dbias);
        }
    }
    destroyMatrix(padded_delta);
    destroyMatrix(out_tmp2);
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

    Dshape flitershape;
    initDshapeInt(&flitershape, 0, 0,fliter,fliter);
    Matrix *i_a = creatMatrixFromValue(0, flitershape);
    for(int n = 0;n< N;n++)
    {
        for(int c = 0;c < channel_out;c++)
        {
            setZeros(out_tmp);
            for(int inc=0;inc < channel_in;inc++)
            {
                Conv4D(p->input, p->weight,i_a,result, outputH,outputW,p->strids,n,inc,c,inc);
                addSecondOrderMatrixs2(out_tmp,result);
            }
            kAddMatrix(out_tmp, getMatrixElem(p->bias, 0, 0, 0, c));
            setMatrixArray(_output,out_tmp,n,c);
        }
    }
    p->forward(_output);
    p->_output = _output;
    
    destroyMatrix(i_a);
    destroyMatrix(out_tmp);
    destroyMatrix(result);
    return ;
    
    
}