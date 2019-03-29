//
//  Matrix_com.c
//  GateC
//
//  Created by Steven on 28/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "Matrix_com.h"
double Wise_Add(double p1,double p2)
{
    return p1 + p2;
}
double Wise_Mul(double p1,double p2)
{
    return p1 * p2;
}
double Wise_Sign(double p1,double p2)
{
    return -p1;
}
double Wise_todiv(double p1,double p2)
{
    if(p1 != 0) return 1/p1;
    else return 0;
}
double Wise_div(double p1,double p2)
{
    if(p2 != 0) return p1/p2;
    else return p1;
}
double Wise_exp(double p1,double p2)
{
    return exp(p1);
    
}
double Wise_SigmoidBackward(double p1,double p2)
{
    return p1*(1-p1);
}
double Wise_TanhForward(double p1,double p2)
{
    return 2.0 / (1.0 + exp(-2 * p1)) - 1.0;
}
double Wise_TanhBackward(double p1,double p2)
{
    return (1 - p1 * p1);
}

double Wise_SoftmaxForward(double p1,double p2)
{
    //double shifted_x = p1 - p2;
    return exp(p1 - p2);
}
