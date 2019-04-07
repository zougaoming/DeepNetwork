//
//  Gate.c
//  GateC
//
//  Created by Steven on 21/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "Gate.h"
#include <string.h>
ParamLink *Link = NULL;
void createLink(ParamLink *lastLink,char *name,void *newp){
    if(lastLink == NULL)
    {
        Link = (ParamLink*)malloc(sizeof(ParamLink));
        Link->name = name;
        Link->p = newp;
        Link->next = NULL;
    }
    else
    {
        ParamLink * newLink = (ParamLink*)malloc(sizeof(ParamLink));
        newLink->name = name;
        newLink->p = newp;
        newLink->next = NULL;
        lastLink->next = newLink;
    }
    
}
int isExits(char* name)
{
    ParamLink * oldLink = NULL;
    ParamLink *cur = Link;
    while(cur != NULL)
    {
        //printf("%s\n",name);
        //printf("cur->name=%s\n",cur->name);
        if(name && strcmp(name, cur->name) == 0)
        {
            //printf("hasFind");
            return 1;
        }
        oldLink = cur;
        cur = cur->next;
    }
    return 0;
}
ParamLink* findParamByName(char* name,void *p){
    if(!Link){
        createLink(Link,name,p);
    }
    
    ParamLink * oldLink = NULL;
    ParamLink *cur = Link;
    while(cur != NULL)
    {
        if(strcmp(name, cur->name) == 0)
        {
            return cur;
        }
        oldLink = cur;
        cur = cur->next;
    }
    createLink(oldLink,name,p);
    return oldLink->next;
}

int getOutputSize(int M,int S,int F,int P)
{
    return (int)((M - F + 2 * P) / S + 1);
}