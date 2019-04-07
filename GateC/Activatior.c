//
//  ReluActivatior.c
//  GateC
//
//  Created by Steven on 22/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//
#include <math.h>
#include <string.h>
#include "Activatior.h"
#include  <dlfcn.h>

/*
pActivatorFuncLink AFuncLinkHead = NULL;
pActivatorFuncLink AFuncLinkTail = NULL;
void createAFuncLink(char* name,Activator_Forward forward,Activator_Backward backward)
{
    if(AFuncLinkHead == NULL)
    {
        AFuncLinkHead = (ActivatorFuncLink*)MemoryPool_Alloc(sizeof(ActivatorFuncLink));
        
        AFuncLinkHead->next = NULL;
        AFuncLinkHead->name=name;
        AFuncLinkHead->backward = backward;
        AFuncLinkHead->forward = forward;
        AFuncLinkTail = AFuncLinkHead;
    }
    else{
        pActivatorFuncLink pNew = (pActivatorFuncLink)MemoryPool_Alloc(sizeof(ActivatorFuncLink));    //    为节点分配空间
        pNew->name=name;
        pNew->backward = backward;
        pNew->forward = forward;
        AFuncLinkTail->next = pNew;                        //将最后一个节点的指针指向下一个新的节点
        pNew->next = NULL;                            //将新节点中的指针置为空
        AFuncLinkTail = pNew;                                //将新节点赋给最后的一个节点
    }

}

void initActivator()
{
    createAFuncLink("ReluActivator",ReluActivator_Forward,ReluActivator_Backward);
    createAFuncLink("IdentityActivator",IdentityActivator_Forward,IdentityActivator_Backward);
    createAFuncLink("SigmoidActivator",SigmoidActivator_Forward,SigmoidActivator_Backward);
    createAFuncLink("TanhActivator",TanhActivator_Forward,TanhActivator_Backward);
    createAFuncLink("SoftmaxActivator",SoftmaxActivator_Forward,SoftmaxActivator_Backward);
}
pActivatorFuncLink getActivator(char* name)
{
    if(AFuncLinkHead == NULL) initActivator();
    pActivatorFuncLink activefunc = AFuncLinkHead;
    while(activefunc != NULL)
    {
        if(strcmp(activefunc->name,name) == 0)
        {
            return activefunc;
        }
    }
    return NULL;
}
*/

char* strjoin(char *s1, char *s2)
{
    char *result = MemoryPool_Alloc(strlen(s1)+strlen(s2)+1);//+1 for the zero-terminator
    //in real code you would check for errors in malloc here
    if (result == NULL) exit (1);
    
    strcpy(result, s1);
    strcat(result, s2);
    
    return result;
}

Activator_Forward getForward(char* name)
{
    name = strjoin(name, "_Forward");
    Activator_Forward func = dlsym(RTLD_DEFAULT,name);
    MemoryPool_Free(name);
    return func;
}
Activator_Backward getBackward(char* name)
{
    name = strjoin(name, "_Backward");
    Activator_Backward Func = dlsym(RTLD_DEFAULT,name);
    MemoryPool_Free(name);
    return Func;
}

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
void  SoftmaxActivator_Backward(Matrix *dz,Matrix *output)
{
    Dshape gradShape;
    initDshapeInt(&gradShape, output->dshape.shape[0], output->dshape.shape[1], output->dshape.shape[2], output->dshape.shape[3]);
    Matrix *grad = creatMatrixFromValue(0, gradShape);
    Dshape HWshape;
    initDshapeInt(&HWshape, 0, 0, output->dshape.shape[2], output->dshape.shape[3]);
    Matrix *HWOutput = creatMatrixFromValue(0, HWshape);
    Matrix *HW0 = creatMatrixFromValue(0, HWshape);
    Matrix *HW1 = creatMatrixFromValue(0, HWshape);
    int z = output->dshape.shape[3];
    int y = output->dshape.shape[2] * z;
    int x = output->dshape.shape[1] * y;
    int index = 0;
    for(int i = 0;i<output->dshape.shape[0];i++)
    {
        index = i * x;
        for(int c0 = 0;c0<output->dshape.shape[1];c0++)
        {
            get2dim(output,HWOutput,i,c0);
            for(int c1 = 0;c1<output->dshape.shape[1];c1++)
            {
                get2dim(output,HW1,i,c1);
                dotSecondOrderMatrixs2(HW1,HWOutput);
                
                if(c1 == c0)
                {
                    subSecondOrderMatrixs2(HW1,HWOutput);
                }
            
                doWise(HW1,-1,Wise_Sign,NULL);
                //setMatrixArray(grad,HW1,c0,c1);
                get2dim(dz,HW0,i,c1);
                dotSecondOrderMatrixs2(HW0,HW1);
                addSecondOrderMatrixsby2d(grad,HW0,i,c0);
            }
        }
    }
    dz = grad;
    
    destroyMatrix(HWOutput);
    destroyMatrix(HW0);
    destroyMatrix(HW1);
    //printf("t->%f\n",get_mempool_usage());
    //return grad;
}





