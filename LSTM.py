#!/usr/bin/env Python
# coding=utf-8
import numpy as np
from ActiveFunc import TanhActivator,softmax3,SigmoidActivator
from Gate import NeuronGate,AddGate,MulGate,InOutGate,ConcateGate,CopyGate,Gate
from Json2Network import JsonModel
from Optimizer import *

class LstmParam:

	def __init__(self, ModelFile='./gojs/model.txt', inputsize=0, outputsize=0, keepDropout=0, Optimizer=None, rate=0.05, ActiveFunc=None):
		self.ModelFile = ModelFile
		self.keepDropout = keepDropout
		self.inputsize = inputsize
		self.outputsize = outputsize
		self.rate = rate
		self.ActiveFunc = ActiveFunc
		if Optimizer is None:
			self.Optimizer = SGDandMomentumOptimizer(self)
		else:
			self.Optimizer = Optimizer

class LSTM_layer:
	def __init__(self,lstmParam,state_size,output_size,index,jsonModel=None):
		self.index = index
		self.param = lstmParam
		self.jsonModel = jsonModel
		self.Gate = Gate('', self)
		self.Gates = jsonModel.getGatesAndParam(self)


		#self.prev_s = np.zeros((state_size, 1))
		#self.prev_h = np.zeros((output_size,1))
		#self.dprev_s = np.zeros_like(self.prev_s)
		#self.dOutput = np.zeros_like(self.prev_h)
		self.cache = 'prev_h=None,prev_s=None'
	def forward(self,input,prev_h=None,prev_s=None):
		if prev_h is None:prev_h = self.prev_h
		if prev_s is None:prev_s = self.prev_s
		setattr(self, self.jsonModel.input, input)

		for db in self.jsonModel.DB:#fromDB
			key = 'o' + self.jsonModel.key2bz(db['key'])
			value = db['text']
			setattr(self,key,eval(value))

		for g in self.Gates:#forward
			str = 'self.' + g.Key + '.forward()'
			#print(g.Value)
			eval(str)

		for db in self.jsonModel.Output:#toDB
			value = 'self.o' + self.jsonModel.key2bz(db['key'])
			key = db['text']
			setattr(self, key, eval(value))
		return self.Output,self.prev_s

	def backward(self,dOutput=None,dprev_s=None):
		if dOutput is None:doutput = self.dOutput
		if dprev_s is None:dprev_s = self.dprev_s
		for db in self.jsonModel.Output:  # fromDB
			key = 'do' + self.jsonModel.key2bz(db['key'])
			value = 'd' + db['text']
			setattr(self, key, eval(value))
		self.Gate.runBackward()#backward

		for db in self.jsonModel.DB:#toDB
			value = 'self.do' + self.jsonModel.key2bz(db['key'])
			key = 'd' + db['text']
			setattr(self, key, eval(value))

		return self.dOutput,self.dprev_s

	def setOutputDelta(self,delta):
		self.dOutput = delta

	def Dropout(self, x, level):
		if level == 0:
			return x
		if level < 0. or level >= 1:  # level是概率值，必须在0~1之间
			raise ValueError('Dropout level must be in interval [0, 1[.')
		retain_prob = 1. - level

		# 我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
		# 硬币 正面的概率为p，n表示每个神经元试验的次数
		# 因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
		random_tensor = np.random.binomial(n=1, p=retain_prob,
										   size=x.shape)  # 即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
		x *= random_tensor
		# x /= retain_prob
		np.true_divide(x, retain_prob, out=x, casting='unsafe')
		return x


class LSTM:
	def __init__(self, hidden_size):
		self.lstmParam = LstmParam()
		self.jsonModel = JsonModel(self.lstmParam)
		self.jsonModel.run()

		self.input_size = self.lstmParam.inputsize
		self.hidden_size = hidden_size
		self.output_size = self.lstmParam.outputsize
		self.newInputSize = self.input_size + self.output_size
		self.Layers = []


	def setInput(self,X):
		self.Input = X
		self.len = len(X)
		for i in range(self.len):
			l = LSTM_layer(self.lstmParam, self.hidden_size, self.output_size,i,self.jsonModel)
			self.Layers.append(l)

	def forward(self):
		index = 0
		prev_h = prev_s = None
		for l in self.Layers:
			prev_h,prev_s = l.forward(self.Input[index],prev_h,prev_s)
			print(index, '->', l.Output)
			index += 1

		#return self.Layers[-1].output


	def backward(self,Y):
		index = self.len-1
		dprev_h = softmax3().backward(self.Layers[-1].Output, Y[-1])  # self.Layers[-1].dprev_c
		self.Layers[-1].setOutputDelta(dprev_h)
		dprev_s = np.zeros((self.hidden_size,1))
		for l in self.Layers[::-1]:
			dprev_h, dprev_s = l.backward(dprev_h,dprev_s)
			index -= 1


def data_set():
	x = [np.array([[1], [0], [1]]),
         np.array([[2], [3], [4]]),np.array([[2], [3], [4]]),np.array([[2], [3], [4]])]
	d = np.array([[[.123], [.2],[.3], [.2],[0.5]]])
	return x,d

class Param:
	def __init__(self,isize,hsize,osize,rate=0.05):

		self.w0 = np.random.uniform(-0.001,0.001,size=(isize,hsize))
		self.w1 = np.random.uniform(-0.001,0.001,size=(hsize,osize))
		self.b0 = np.random.uniform(-0.001,0.001,size=(hsize,1))
		self.b1 = np.random.uniform(-0.001, 0.001, size=(osize, 1))
		self.activeFunc = SigmoidActivator()
		self.rate=rate
class bp:
	def __init__(self,param):
		self.param = param
		self.Gate = Gate('',self)
		self.NeuronGate0 = NeuronGate(self.Gate, Input='input', W='w0',
									   bias='b0', o='o0', activeFunc=SigmoidActivator())
		self.NeuronGate1 = NeuronGate(self.Gate, Input='o0', W='w1',
									  bias='b1', o='o1', activeFunc=None)

	def forward(self,X):
		self.input = X
		self.NeuronGate0.forward()
		self.NeuronGate1.forward()
		return self.o1

	def backward(self,delta):
		self.do1 =  delta
		self.Gate.runBackward()

if __name__ == '__main__':

	'''
	param = Param(3,8,5)
	network = bp(param=param)
	x = np.array([[1],[2],[3]])
	y = np.array([[1],[.2],[3],[.4],[5]])
	for e in range(200):
		t = network.forward(x)
		delta = t - y
		print(t)
		network.backward(delta)
		#network.update()
	'''



	#Model().productLayer()

	l = LSTM(5)
	x, d = data_set()
	l.setInput(x)
	for e in range(150):

		l.forward()
		l.backward(d)






