#coding=gbk
import GateC
import numpy as np
import struct
from ctypes import *

isBackward = 1
data1 = np.array([[2,2,3],[4,5,6]],dtype=np.double)
#buf = struct.pack("P",byref(data))
data2 = np.array([[1,2,3],[4,5,6]],dtype=np.double)
data3 = np.array([[1,2,3],[4,5,6]],dtype=np.double)
result = GateC.AddGate(data1,data2,0,"a")

print("hhh")




input = np.array([[1,2,3,4],
				  [4,5,6,5],
				  [7,8,9,0],
				  [1,2,3,4]],dtype=np.double)
#input = np.random.uniform(1,2,size=(2,1,6,6))
result = GateC.PoolGate(input,2,2,0,"b")
print(result)

#result2 = GateC.PoolGate(result,2,2,1)
#print(result2)

result = GateC.AddGate(data1,data2,isBackward,"a")
print("ccc")
result = GateC.PoolGate(input,2,2,isBackward,"b")
print("ddd")
result = GateC.PoolGate(input,2,2,0,"c")
result = GateC.PoolGate(input,2,2,1,"c")