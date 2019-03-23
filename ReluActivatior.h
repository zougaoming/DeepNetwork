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
void ActiveForward(Matrix *m);
void ActiveBackward(Matrix *dz,Matrix *output);
#endif /* ReluActivatior_h */
