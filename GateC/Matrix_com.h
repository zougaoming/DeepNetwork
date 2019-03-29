//
//  Matrix_com.h
//  GateC
//
//  Created by Steven on 28/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef Matrix_com_h
#define Matrix_com_h
#include "Matrix.h"
#include <stdio.h>
#include <math.h>
typedef double(*WiseFunc)(double,double);
double Wise_exp(double p1,double p2);
double Wise_Add(double p1,double p2);
double Wise_Sign(double p1,double p2);
double Wise_todiv(double p1,double p2);
double Wise_Mul(double p1,double p2);
double Wise_SigmoidBackward(double p1,double p2);
double Wise_TanhForward(double p1,double p2);
double Wise_TanhBackward(double p1,double p2);
double Wise_SoftmaxForward(double p1,double p2);
//WiseFunc Wise_exp;
#endif /* Matrix_com_h */
