//
//  Gate.h
//  GateC
//
//  Created by Steven on 21/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#ifndef Gate_h
#define Gate_h

#include <stdio.h>
#include "Activatior.h"
typedef struct Params{
    void* p;
    char * name;
    struct Params *next;
}ParamLink;
int isExits(char* name);
ParamLink* findParamByName(char* name,void *p);
int getOutputSize(int M,int S,int F,int P);
#endif /* Gate_h */
