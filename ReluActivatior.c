//
//  ReluActivatior.c
//  GateC
//
//  Created by Steven on 22/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "ReluActivatior.h"
void ActiveForward(Matrix *m)
{
    getMaximumMatrix(m,0);
}
void ActiveBackward(Matrix *dz,Matrix *output)
{
    double  tmp;
    for(int i = 0;i<dz->length;i++)
    {
        tmp = *(output->array + i);
        if(tmp < .0)
            *(dz->array + i) = (double).0;
    }
}