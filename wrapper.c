//
//  wrapper.c
//  GateC
//
//  Created by Steven on 18/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "wrapper.h"
#include <Python.h>
#include <stdlib.h>
#include "GateC.h"
#include "Gate.h"
#include <string.h>
/*
typedef struct  {
    int row, col;
    float *element;
    unsigned char init;
}Mat;
*/



static PyObject *py_PoolGate(PyObject *self,PyObject * args)
{
    
    PyObject* p_object;
    char* p_name;
    PoolGateParam p;
    if(!PyArg_ParseTuple(args,"Oiiis",&p_object,&(p.filter),&(p.strids),&(p.isBackward),&p_name))
    {
        return NULL;
    }
    ParamLink* plink = NULL;
    
    if((plink = isExits(p_name)) == 0)
    {
        PoolGateParam *newp = (PoolGateParam*)malloc(sizeof(PoolGateParam));
        memcpy(newp,&p,sizeof(PoolGateParam));
        plink = findParamByName(p_name,newp);
    }
    plink = findParamByName(p_name,&p);
    PoolGateParam *curp = (PoolGateParam *)(plink->p);
    curp->isBackward = p.isBackward;
    if(curp->isBackward == 1)curp->dz = pyArray2Matrix((PyArrayObject*)p_object);
    else curp->input = pyArray2Matrix((PyArrayObject*)p_object);
    PyArrayObject *result =  matrix2pyArray(PoolGate(curp));
    return (PyObject *)Py_BuildValue("O",result);
}

static PyObject *py_CnnGate(PyObject *self,PyObject * args)
{
    
    PyObject* p_object;
    PyObject* p_weight;
    PyObject* p_bias;
    char* p_name;
    CnnGateParam p;
    if(!PyArg_ParseTuple(args,"OOOiiis",&p_object,&(p_weight),&(p_bias),&(p.strids),&(p.panding),&(p.isBackward),&p_name))
    {
        return NULL;
    }
    printf("isBackward=%d",p.isBackward);
    ParamLink* plink = NULL;
    
    
    
    if((plink = isExits(p_name)) == 0)
    {
        printf("no Exits!\n");
        PoolGateParam *newp = (PoolGateParam*)malloc(sizeof(PoolGateParam));
        memcpy(newp,&p,sizeof(PoolGateParam));
        plink = findParamByName(p_name,newp);
    }
    
    plink = findParamByName(p_name,&p);
    CnnGateParam *curp = (CnnGateParam *)(plink->p);
    curp->isBackward = p.isBackward;
    printf("from weight\n");
    printArrayShape((PyArrayObject*)p_weight);
    curp->weight = pyArray2Matrix((PyArrayObject*)p_weight);
    printShape(curp->weight);
    curp->bias = pyArray2Matrix((PyArrayObject*)p_bias);

    if(curp->isBackward == 1)
    {   printf("is Backward here");
        curp->dz = pyArray2Matrix((PyArrayObject*)p_object);
    }
    else curp->input = pyArray2Matrix((PyArrayObject*)p_object);
    printShape(curp->weight);
    CnnGate(curp);
    printShape(curp->weight);
    if(curp->isBackward == 1)
    {
        PyArrayObject *dw =  matrix2pyArray(curp->dw);
        PyArrayObject *dx =  matrix2pyArray(curp->dx);
        PyArrayObject *dbias =  matrix2pyArray(curp->dbias);
        return (PyObject *)Py_BuildValue("OOO",dw,dbias,dx);
    }
    else
    {
        
        PyArrayObject *result =  matrix2pyArray(curp->_output);
        printf("here");
        return (PyObject *)Py_BuildValue("O",result);
    }
}
static PyObject *py_AddGate(PyObject *self,PyObject * args)
{
    
    PyObject *obj1;
    PyObject *obj2;
    PyObject* p_name;
    AddGateParam p;
    if(!PyArg_ParseTuple(args,"O|Ois",&obj1,&obj2,&(p.isBackward),&p_name))
    {
        return NULL;
    }
    ParamLink* plink = findParamByName((char *)p_name,&p);
    AddGateParam* curp = (AddGateParam*)(plink->p);
    if(curp->isBackward == 1)curp->dz = pyArray2Matrix((PyArrayObject*)obj1);
    else{
        curp->input1 = pyArray2Matrix((PyArrayObject*)obj1);
        curp->input2 = pyArray2Matrix((PyArrayObject*)obj2);
    }
    AddGate(curp);
    PyArrayObject *result = matrix2pyArray(curp->_output);
    return (PyObject *)Py_BuildValue("O",result);
/*s
    const char * buffer;
    PyObject *obj;
    int buffer_len = sizeof(Mat);
    if(!PyArg_ParseTuple(args,"O",&obj))
    {
        return NULL;
    }
    PyObject_AsCharBuffer(obj, &buffer, &buffer_len);
    Mat *mat = (Mat *)buffer;
    printf("mat->col = %d",mat->col);
    printf("mat->row = %d",mat->row);
    printf("mat->element = %s",mat->element);
    printf("mat->col = %c",mat->init);
    return NULL;//(PyObject*)Py_BuildValue("O", obj);
    
    
    PyObject *pList;
    char pItem;
    Py_ssize_t n;
    int i;
    char *buffer;
    int buffer_len = 5;
    
    if (!PyArg_ParseTuple(args, "O!", &pList)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be a list.");
        return NULL;
    }
    PyObject_AsCharBuffer(pList, &buffer, &buffer_len);
    printf("%f",buffer[0]);
    return (PyObject *)Py_BuildValue("i",test(buffer));
    
    PyObject *s;
    if(!(PyArg_ParseTuple(args,"O",&s))){
        return NULL;
    }
    PyObject *iter = PyObject_GetIter(s);
    if (!iter) {
        // error not iterator
    }
    PyObject *next;
    while (1) {
        next = PyIter_Next(iter);
        if (!next) {
            // nothing left in the iterator
            break;
        }
        
        if (!PyFloat_Check(next)) {
            // error, we were expecting a floating point value
        }
        
        // do something with foo
    }
    //int ret = ;
    return (PyObject *)Py_BuildValue("i",test(iter));
     */
}
//添加模块数组(注意是PyMethodDef,不要错写成PyMethondDef)
//定义对应的方法名，后面Python调用的时候就用这里面的方法名调用
static PyMethodDef py_test1Methods[] = {
    {"PoolGate",py_PoolGate,METH_VARARGS},
    {"AddGate",py_AddGate,METH_VARARGS},
    {"CnnGate",py_CnnGate,METH_VARARGS},
    
    {NULL,NULL},
};
static struct PyModuleDef ExtestModule =
{
    PyModuleDef_HEAD_INIT,
    "GateC", NULL, -1, py_test1Methods
};
void init_numpy2()
{
    Py_Initialize();
    {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");  } }

}

PyMODINIT_FUNC PyInit_GateC(void)
{
    //printf("this");
    Py_Initialize();
    if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); }
    return PyModule_Create(&ExtestModule);
}
