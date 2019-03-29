//
//  GateC.c
//  GateC
//
//  Created by Steven on 18/3/19.
//  Copyright © 2019年 Steven. All rights reserved.
//

#include "GateC.h"
#include "numpy/ndarraytypes.h"
Matrix * pyArray2Matrix(PyArrayObject* array)
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
    Matrix *m = creatMatrixFromDatas(array->data,len,dshape);
    return m;
}
void printArray(PyArrayObject * array)
{
    //int len = 1;
    //for(int i = 0;i<array->nd;i++)
    //{
        //len *= array->dimensions[i];
    //}
    for(int i = 0;i<array->dimensions[0];i++)
    {
        for(int j = 0;j<array->dimensions[1];j++)
        {
            for(int k = 0;k<array->dimensions[2];k++)
            {
                for(int l=0;l<array->dimensions[3];l++)
                {
                    printf("index=(%d,%d) -> %f\n",i,j,*(double *)((double *)array->data + l*array->dimensions[3]+k*array->dimensions[2] + i * array->strides[0]+ j * array->strides[1]));
                }
            }
            
        }
        
    }
}
void printArrayShape(PyArrayObject * array)
{
    printf("(");
    for(int i = 0;i<array->nd;i++)
    {
        printf("%d,",array->dimensions[i]);
    }
    printf(")");

}
void init_numpy(){
    Py_Initialize();
{if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");  } }
}

void printPyArrayObject_dimensions(PyArrayObject *array)
{
    for(int i=0;i<array->nd;i++)
        printf("ndim=%d i=%d dimensions=%d\n",array->nd,i,array->dimensions[i]);
}
PyArrayObject * matrix2pyArray(Matrix* array)
{
    
    init_numpy();
    int ndim = 0;
    for(int i = 0;i<4;i++)
    {
        if(array->dshape.shape[i] > 0)
        {
            ndim = 4 - i;
            break;
        }
    }
    //printShape(array);
    npy_intp dimensions[ndim];
    for(int i = 0;i<ndim;i++)
    {
        dimensions[i] = array->dshape.shape[4- ndim + i];
    }
    
    for(int i = 0;i < ndim;i++)
    {
        //printf("%d -> %d;",i,dimensions[i]);
    }
    
    PyArrayObject * outArray = (PyArrayObject *)PyArray_SimpleNew(ndim, dimensions, NPY_DOUBLE);
    if(!outArray){printf("matrix2pyArray->new PyArrayObject error!") ;}
    outArray->flags |= NPY_ARRAY_OWNDATA;
    //printPyArrayObject_dimensions(outArray);
    double* ptr = (double *)outArray->data;
    
    for(int i = 0;i<array->length;i++)
    {
        ptr[i] = *((double*)(array->array) + i);
        
    }
    destroyMatrix(array);
    return outArray;

}
/*
void* findParambyName(void* p,char * name)
{
    //for(int i = 0;i < )
}
 */



