//
//  GateC.h
//  GateC
//
//  Created by Steven on 18/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef GateC_h
#define GateC_h
#include <stdio.h>
#include "numpy/ndarrayobject.h"
#include "Matrix.h"
#include "PoolGate.h"
#include "AddGate.h"
#include "CnnGate.h"
/*
 
 int nd：Numpy Array数组的维度。
 int *dimensions ：Numpy Array 数组每一维度数据的个数。
 int *strides：Numpy Array 数组每一维度的步长。
 char *data： Numpy Array 中指向数据的头指针。

 */


Matrix * pyArray2Matrix(PyArrayObject* array);
PyArrayObject * matrix2pyArray(Matrix* array);
void printArray(PyArrayObject * array);

#endif /* GateC_h */
