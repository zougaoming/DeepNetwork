//
//  test.c
//  GateC
//
//  Created by Steven on 19/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "test.h"
#include "numpy/ndarraytypes.h"
Matrix * pyArray2Matrix_t(PyArrayObject* array)
{
    
    int len = 1;
    Dshape dshape;
    for(int i = 0;i<4;i++)
        dshape.shape[i] = 0;
    for(int i = 0;i<array->nd;i++)
    {
        dshape.shape[i + (4 - array->nd )]= array->dimensions[i];
        len *= array->dimensions[i];
    }
    //initDshape(&dshape, array->dimensions, array->nd);
    Matrix *m = creatAsMatrixFromDatas(array->data,len,dshape);
    //printarray(m);
    return m;
}