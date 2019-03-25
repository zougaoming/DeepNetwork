//
//  main.c
//  GateC
//
//  Created by Steven on 18/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include <stdio.h>
#include "PoolGate.h"
#include "Matrix.h"
PoolGateParam p;
int main(int argc, const char * argv[]) {
    Dshape shape;
    initDshapeInt(&shape, 2, 5, 8, 8);
    Matrix* m = creatZerosMatrix(shape);
    p.input = m;
    p.filter = 2;
    p.strids = 2;
    Matrix*result = PoolGate(&p);
    printarray(result);
    printf("here");
}
