#incoding:utf-8
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#模块名
MOD = 'GateC'
#资源（要编译和链接的代码文件）
source = ['GateC.c','wrapper.c','Matrix.c','test.c','PoolGate.c','AddGate.c','Gate.c','CnnGate.c','ReluActivatior.c']

#调用setup函数,编译和链接
setup(name=MOD,ext_modules=[Extension(MOD,sources=source)])