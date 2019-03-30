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
void ReluActivator_Forward(Matrix *m);
void ReluActivator_Backward(Matrix *dz,Matrix *output);
void IdentityActivator_Forward(Matrix *m);
void IdentityActivator_Backward(Matrix *dz,Matrix *output);
void SigmoidActivator_Forward(Matrix *m);
void SigmoidActivator_Backward(Matrix *dz,Matrix *output);
void SoftmaxActivator_Forward(Matrix *m);
#endif /* ReluActivatior_h */
