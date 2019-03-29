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

static PyObject *py_PoolGate_backward(PyObject *self,PyObject * args)
{
    
    PyObject* p_dz;
    PyObject* p_output;
    PyObject* p_input;
    PyObject* bz_x;
    PyObject* bz_y;
    PoolGateParam p;
    if(!PyArg_ParseTuple(args,"O|O|O|O|O|iis",&p_input,&p_output,&p_dz,&bz_x,&bz_y,&(p.filter),&(p.strids),&(p.type)))
    {
        return NULL;
    }
    p.input = pyArray2Matrix((PyArrayObject*)p_input);
    p._output = pyArray2Matrix((PyArrayObject*)p_output);
    p.dz = pyArray2Matrix((PyArrayObject*)p_dz);
    p.bz_x = pyArray2Matrix((PyArrayObject*)bz_x);
    p.bz_y = pyArray2Matrix((PyArrayObject*)bz_y);
    PyArrayObject *result =  matrix2pyArray(PoolGate_Backward(&p));
    
    return (PyObject *)Py_BuildValue("O",result);
}

static PyObject *py_PoolGate(PyObject *self,PyObject * args)
{
    
    PyObject* p_object;
    PoolGateParam p;
    if(!PyArg_ParseTuple(args,"Oiis",&p_object,&(p.filter),&(p.strids),&(p.type)))
    {
        return NULL;
    }
    printf("here4");
    p.input = pyArray2Matrix((PyArrayObject*)p_object);
    PoolGateParam *curp = &p;
    PyArrayObject *result =  matrix2pyArray(PoolGate_Forward(curp));
    if(strcmp(p.type, "MAX") == 0)
    {
        PyArrayObject* bz_x = matrix2pyArray(curp->bz_x);
        PyArrayObject* bz_y = matrix2pyArray(curp->bz_y);
        return (PyObject *)Py_BuildValue("OOO",result,bz_x,bz_y);
    }
    return (PyObject *)Py_BuildValue("O",result);
}
static PyObject *py_CnnGate_backward(PyObject *self,PyObject * args)
{
    
    PyObject* p_input;
    PyObject* p_output;
    PyObject *p_dz;
    PyObject* p_weight;
    PyObject* p_bias;
    CnnGateParam p;
    if(!PyArg_ParseTuple(args,"O|O|O|O|Oii",&p_input,&p_output,&p_dz,&(p_weight),&(p_bias),&(p.strids),&(p.panding)))
    {
        return NULL;
    }
    p.input = pyArray2Matrix((PyArrayObject*)p_input);
    p._output = pyArray2Matrix((PyArrayObject*)p_output);
    p.weight = pyArray2Matrix((PyArrayObject*)p_weight);
    p.bias = pyArray2Matrix((PyArrayObject*)p_bias);
    p.dz = pyArray2Matrix((PyArrayObject*)p_dz);
    Backward(&p);
    PyArrayObject *dw =  matrix2pyArray(p.dw);
    PyArrayObject *dx =  matrix2pyArray(p.dx);
    PyArrayObject *dbias =  matrix2pyArray(p.dbias);
    return (PyObject *)Py_BuildValue("OOO",dw,dbias,dx);
    
}

static PyObject *py_CnnGate(PyObject *self,PyObject * args)
{
    
    PyObject* p_object;
    PyObject* p_weight;
    PyObject* p_bias;
    CnnGateParam p;
    if(!PyArg_ParseTuple(args,"O|O|Oii",&p_object,&(p_weight),&(p_bias),&(p.strids),&(p.panding)))
    {
        return NULL;
    }

    CnnGateParam *curp = &p;
    //curp->isBackward = p.isBackward;
    curp->weight = pyArray2Matrix((PyArrayObject*)p_weight);
    curp->bias = pyArray2Matrix((PyArrayObject*)p_bias);
    curp->input = pyArray2Matrix((PyArrayObject*)p_object);
    printf("here1");
    Forward(curp);
    printf("here2");
    PyArrayObject *result =  matrix2pyArray(curp->_output);
    //Py_DECREF(p_object);
    //Py_DECREF(p_weight);
    //Py_DECREF(p_bias);
    printf("here3");
    PyObject *  t = (PyObject *)Py_BuildValue("O",result);
    printf("here4");
    return t;
}
static PyObject *py_test(PyObject *self,PyObject * args)
{
    
    PyObject* p_test;
    PyObject* p_test2;
    if(!PyArg_ParseTuple(args,"O|O",&p_test,&p_test2))
    {
        return NULL;
    }
    Matrix * test = pyArray2Matrix((PyArrayObject *)p_test);
    Matrix * test2 = pyArray2Matrix((PyArrayObject *)p_test2);
    TanhActivator_Backward(test,test2);
    PyArrayObject *result =  matrix2pyArray(test);
    
    return (PyObject *)Py_BuildValue("O",result);
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
    {"PoolGate_Backward",py_PoolGate_backward,METH_VARARGS},
    {"AddGate",py_AddGate,METH_VARARGS},
    {"CnnGate",py_CnnGate,METH_VARARGS},
    {"CnnGate_Backward",py_CnnGate_backward,METH_VARARGS},
    {"test",py_test,METH_VARARGS},
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
    getMemoryPool();
    Py_Initialize();
    if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); }
    return PyModule_Create(&ExtestModule);
}
