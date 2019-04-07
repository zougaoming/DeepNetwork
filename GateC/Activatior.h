//
//  ReluActivatior.h
//  GateC
//
//  Created by Steven on 22/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef ReluActivatior_h
#define ReluActivatior_h

#include <stdio.h>
#include "Matrix.h"

typedef void (*Activator_Forward)(Matrix*);
typedef void (*Activator_Backward)(Matrix*,Matrix*);
void ReluActivator_Forward(Matrix *m);
void ReluActivator_Backward(Matrix *dz,Matrix *output);
void IdentityActivator_Forward(Matrix *m);
void IdentityActivator_Backward(Matrix *dz,Matrix *output);
void TanhActivator_Forward(Matrix *m);
void TanhActivator_Backward(Matrix *dz,Matrix *output);
void SigmoidActivator_Forward(Matrix *m);
void SigmoidActivator_Backward(Matrix *dz,Matrix *output);
void SoftmaxActivator_Forward(Matrix *m);
void SoftmaxActivator_Backward(Matrix *dz,Matrix *output);

/*
typedef struct ActivatorFuncLinks
{
    char * name;
    struct ActivatorFuncLinks *next;
    Activator_Backward backward;
    Activator_Forward forward;
    
}ActivatorFuncLink,*pActivatorFuncLink;
pActivatorFuncLink getActivator(char* name);
*/
Activator_Forward getForward(char* name);
Activator_Backward getBackward(char* name);
#endif /* ReluActivatior_h */
